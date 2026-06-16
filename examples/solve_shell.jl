# Example: solve the 3D shell Laplace problem u = 1/r on a cubed-sphere shell
# with an inner excision hole, two ways —
#
#   (1) FINITE shell  : inner Dirichlet u = 1/R1, outer Robin fall-off
#                       ∂ₙu + u/R2 = 0  (the "step 1" system);
#   (2) COMPACTIFIED  : outer boundary mapped to spatial infinity i⁰ with exact
#                       Dirichlet u = 0 there  (the "step 2" system).
#
# It prints progress + summary statistics for both, then visualizes the finite
# solution and its pointwise error on the equatorial plane z = 0 as standalone
# SVG heatmaps (no plotting-package dependency).
#
# Run from the package root with the examples environment:
#   julia --project=examples examples/solve_shell.jl
# (the examples environment `dev`s HexCGSEM and HexMeshes).

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, dof_coords, boundary_dofs, assemble_robin,
                solve_dirichlet, gather!, nlocal
using HexMeshes
using HexMeshes: interpolate_field, nv
using SparseArrays: nnz
using StaticArrays: SVector
using Printf

const OUTDIR = joinpath(@__DIR__, "output")

# ----------------------------------------------------------------------
# small progress helpers
phase(label) = (printstyled("\n▶ ", label, "\n"; color = :cyan, bold = true); flush(stdout))
info(msg) = (println("    ", msg); flush(stdout))

# ----------------------------------------------------------------------
# Solve one case. Returns a NamedTuple with the solution and statistics.
function solve_case(; compactified::Bool, R1, R2, M, M_r, p)
    uex(x) = 1 / sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    phase(compactified ? "Compactified shell — outer boundary at i⁰ (exact Dirichlet)" :
                          "Finite shell — outer Robin fall-off ∂ₙu + u/R2 = 0")

    t_mesh = @elapsed mesh = compactified ?
        make_compactified_shell_mesh(Float64, R1, M; M_r = M_r, inner_bc = :dirichlet) :
        make_radial_shell_mesh(Float64, R1, R2, M; M_r = M_r, outer_bc = :sommerfeld, inner_bc = :dirichlet)
    info(@sprintf("mesh: %d elements, %d vertices  (%.3f s)", mesh.Ne, nv(mesh), t_mesh))

    refel = ReferenceElement(Float64, p)
    dof = DofHandler(mesh, p)
    qr = QuadratureRule(refel, 2p)
    tq = TensorQuadrature(refel, qr, Val(3))
    info(@sprintf("degree p = %d,  global DOFs = %d", p, dof.ndofs))

    t_asm = @elapsed K = assemble_stiffness(dof, mesh, tq)
    info(@sprintf("stiffness assembled: %d nonzeros  (%.3f s)", nnz(K), t_asm))

    Xg = dof_coords(dof, mesh, refel)
    dvals = Dict{Int, Float64}()
    local K_total, b
    if compactified
        K_total = K
        b = zeros(dof.ndofs)
        for d in boundary_dofs(dof, mesh; tags = (Int8(2),))   # inner sphere
            dvals[d] = uex(Xg[d])
        end
        for d in boundary_dofs(dof, mesh; tags = (Int8(1),))   # i⁰ face
            dvals[d] = 0.0
        end
    else
        t_rob = @elapsed begin
            Krob, brob = assemble_robin(dof, mesh, refel, qr; tags = (Int8(7),),
                                        a = x -> 1 / R2, g = x -> 0.0)
        end
        info(@sprintf("Robin surface term assembled  (%.3f s)", t_rob))
        K_total = K + Krob
        b = brob
        for d in boundary_dofs(dof, mesh; tags = (Int8(2),))   # inner sphere
            dvals[d] = uex(Xg[d])
        end
    end

    t_solve = @elapsed u = solve_dirichlet(K_total, b, dvals, dof.ndofs)
    info(@sprintf("solved (sparse Cholesky)  (%.3f s)", t_solve))

    # Error statistics over the finite-coordinate DOFs (the i⁰ DOFs sit at r = ∞).
    linf = 0.0
    umax = 0.0
    for g in eachindex(u)
        isfinite(Xg[g][1]) || continue
        ue = uex(Xg[g])
        linf = max(linf, abs(u[g] - ue))
        umax = max(umax, abs(ue))
    end
    info(@sprintf("L∞ error = %.3e   (relative %.3e)", linf, linf / umax))

    return (; mesh, refel, dof, u, uex, R1, R2, compactified,
            ndofs = dof.ndofs, nnz = nnz(K_total), t_solve, linf, rel = linf / umax)
end

# ----------------------------------------------------------------------
# Evaluate a solution on the equatorial plane z = 0 over an `n×n` grid covering
# [-R2, R2]². Points outside the shell annulus R1 ≤ r ≤ R2 are left as NaN.
function equatorial_grid(res; n = 161)
    (; mesh, refel, dof, u, uex, R1, R2) = res
    N = refel.p + 1
    uloc = Matrix{Float64}(undef, nlocal(dof), mesh.Ne)
    gather!(uloc, u, dof)
    uarr = reshape(uloc, N, N, N, mesh.Ne)
    xs = refel.nodes

    gx = collect(range(-R2, R2; length = n))
    Uh = fill(NaN, n, n)
    Err = fill(NaN, n, n)
    nmiss = 0
    for jy in 1:n, ix in 1:n
        x = gx[ix]
        y = gx[jy]
        r = hypot(x, y)
        (R1 <= r <= R2) || continue
        v = try
            interpolate_field(mesh, xs, uarr, SVector(x, y, 0.0); default = NaN)
        catch
            NaN
        end
        if isfinite(v)
            Uh[ix, jy] = v
            Err[ix, jy] = abs(v - uex(SVector(x, y, 0.0)))
        else
            nmiss += 1
        end
    end
    return gx, Uh, Err, nmiss
end

# ----------------------------------------------------------------------
# Minimal self-contained SVG heatmap (viridis-like colormap + colorbar).
# `Z[i, j]` corresponds to (gx[i], gx[j]); NaN cells are drawn light gray.
const _VIRIDIS = ((0.267, 0.005, 0.329), (0.231, 0.318, 0.545),
                  (0.128, 0.567, 0.551), (0.369, 0.789, 0.383), (0.993, 0.906, 0.144))
function _color(t)
    isfinite(t) || return "#eeeeee"
    t = clamp(t, 0.0, 1.0) * (length(_VIRIDIS) - 1)
    k = clamp(floor(Int, t), 0, length(_VIRIDIS) - 2)
    f = t - k
    a = _VIRIDIS[k + 1]
    b = _VIRIDIS[k + 2]
    rgb = ntuple(c -> round(Int, 255 * (a[c] + f * (b[c] - a[c]))), 3)
    return @sprintf("#%02x%02x%02x", rgb[1], rgb[2], rgb[3])
end

function save_heatmap(path, gx, Z, title; vmin = nothing, vmax = nothing)
    n = length(gx)
    vals = filter(isfinite, vec(Z))
    lo = vmin === nothing ? (isempty(vals) ? 0.0 : minimum(vals)) : vmin
    hi = vmax === nothing ? (isempty(vals) ? 1.0 : maximum(vals)) : vmax
    hi <= lo && (hi = lo + 1)
    pad = 55
    sz = 380
    cbw = 22
    W = pad + sz + 70 + cbw
    H = pad + sz + 20
    cw = sz / n
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $W $H" font-family="sans-serif">""")
    println(io, """<text x="$pad" y="28" font-size="16" font-weight="bold">$title</text>""")
    @inbounds for j in 1:n, i in 1:n
        v = Z[i, j]
        x = pad + (i - 1) * cw
        y = pad + (n - j) * cw          # flip so +y points up
        t = isfinite(v) ? (v - lo) / (hi - lo) : NaN
        println(io, @sprintf("""<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s"/>""",
                             x, y, cw + 0.6, cw + 0.6, _color(t)))
    end
    println(io, """<rect x="$pad" y="$pad" width="$sz" height="$sz" fill="none" stroke="#333"/>""")
    # colorbar
    cbx = pad + sz + 30
    nseg = 64
    for s in 0:(nseg - 1)
        y = pad + sz - (s + 1) * sz / nseg
        println(io, @sprintf("""<rect x="%.1f" y="%.2f" width="%d" height="%.2f" fill="%s"/>""",
                             cbx, y, cbw, sz / nseg + 0.6, _color(s / (nseg - 1))))
    end
    println(io, """<rect x="$cbx" y="$pad" width="$cbw" height="$sz" fill="none" stroke="#333"/>""")
    println(io, @sprintf("""<text x="%.1f" y="%.1f" font-size="11">%.3g</text>""", cbx + cbw + 4, pad + 6, hi))
    println(io, @sprintf("""<text x="%.1f" y="%.1f" font-size="11">%.3g</text>""", cbx + cbw + 4, pad + sz, lo))
    println(io, "</svg>")
    write(path, take!(io))
    return path
end

# ----------------------------------------------------------------------
function main()
    R1, R2 = 2.0, 20.0
    M, M_r, p = 3, 4, 6

    printstyled("\nCG-SEM Laplace on a cubed-sphere shell  (u = 1/r,  R1=$R1, R2=$R2)\n";
                color = :yellow, bold = true)

    fin = solve_case(; compactified = false, R1, R2, M, M_r, p)
    cmp = solve_case(; compactified = true, R1, R2, M, M_r, p)

    phase("Summary")
    @printf("  %-26s %10s %12s %12s %12s\n", "case", "DOFs", "solve [s]", "L∞ error", "rel error")
    for (name, r) in (("finite (Robin)", fin), ("compactified (i⁰)", cmp))
        @printf("  %-26s %10d %12.3f %12.3e %12.3e\n", name, r.ndofs, r.t_solve, r.linf, r.rel)
    end
    @printf("\n  compactification improves the L∞ error by %.1g×\n", fin.linf / cmp.linf)

    phase("Equatorial-plane visualization (finite shell, z = 0)")
    gx, Uh, Err, nmiss = equatorial_grid(fin; n = 161)
    info(@sprintf("evaluated %d×%d grid (%d interior misses left blank)", length(gx), length(gx), nmiss))
    mkpath(OUTDIR)
    logErr = map(e -> isfinite(e) && e > 0 ? log10(e) : NaN, Err)
    p1 = save_heatmap(joinpath(OUTDIR, "shell_solution_z0.svg"), gx, Uh, "u = 1/r  on  z = 0  (finite shell)")
    p2 = save_heatmap(joinpath(OUTDIR, "shell_error_z0.svg"), gx, logErr, "log₁₀ |error|  on  z = 0  (finite shell)")
    info("wrote $(relpath(p1))")
    info("wrote $(relpath(p2))")
    finite_vals = filter(isfinite, vec(Err))
    if !isempty(finite_vals)
        info(@sprintf("equatorial-plane error: max %.3e, mean %.3e", maximum(finite_vals),
                      sum(finite_vals) / length(finite_vals)))
    end
    println()
    return nothing
end

main()
