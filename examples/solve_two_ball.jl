# Example: solve the 3D binary-excision Laplace problem on the two_ball mesh
# (a ball minus two spherical holes) under three outer-boundary treatments —
#   finite + Dirichlet, finite + Sommerfeld fall-off Robin, and compactified i⁰ —
# so it offers both a finite outer boundary and a compactified domain.
#
# Manufactured harmonic "binary monopole" u = 1/|x − p₁| + 1/|x − p₂| with the
# 1/r singularities at the ball centres p₁,₂ = (±d/2, 0, 0) (excised ⇒ analytic
# on the domain; → 0 at infinity). Solve Δu = 0.
#
# Prints progress + a summary table and writes z = 0 slice heatmaps (the plane
# through both ball centres) of the solution and pointwise error.
#
# Run:  julia --project=examples examples/solve_two_ball.jl

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, dof_coords, boundary_dofs, assemble_robin,
                solve_dirichlet, solve_dirichlet_cg, gather!, nlocal
using HexMeshes
using HexMeshes: interpolate_field, nv, npatches
using SparseArrays: nnz
using StaticArrays: SVector
using Printf

include(joinpath(@__DIR__, "svg.jl"))

const OUTDIR = joinpath(@__DIR__, "output")

phase(label) = (printstyled("\n▶ ", label, "\n"; color = :cyan, bold = true); flush(stdout))
info(msg) = (println("    ", msg); flush(stdout))

# outer ∈ (:dirichlet, :sommerfeld, :compactified); solver ∈ (:cholesky, :cg).
# `:cg` applies the stiffness operator matrix-free (Jacobi-preconditioned CG) and
# never assembles the global matrix; the small Robin matrix is still assembled.
function solve_case(; outer, R1 = 1.0, R2 = 100.0, d = 10.0, M = 2, p = 3,
                    L = nothing, A = nothing, R_mid = nothing,
                    M_h = 2, M_b = 2, M_i = 2, M_s = 2, mode = :separated, solver = :cholesky)
    a = d / 2
    uex(x) = 1 / hypot(x[1] - a, x[2], x[3]) + 1 / hypot(x[1] + a, x[2], x[3])
    phase("Outer boundary: $outer")

    R2eff = outer === :compactified ? Inf : R2
    obc = outer === :sommerfeld ? :sommerfeld : :dirichlet
    t_mesh = @elapsed mesh = make_two_ball_mesh(Float64, R1, R2eff, d, M;
                                                L, A, R_mid,
                                                M_h, M_b, M_i, M_s, outer_bc = obc, mode)
    info(@sprintf("mesh: %d patches, %d elements, %d vertices  (%.3f s)",
                  npatches(mesh), mesh.Ne, nv(mesh), t_mesh))

    refel = ReferenceElement(Float64, p)
    dof = DofHandler(mesh, p)
    qr = QuadratureRule(refel, 2p)
    tq = TensorQuadrature(refel, qr, Val(3))
    info(@sprintf("degree p = %d,  global DOFs = %d", p, dof.ndofs))

    Xg = dof_coords(dof, mesh, refel)
    if outer === :sommerfeld
        Krob, brob = assemble_robin(dof, mesh, refel, qr; tags = (Int8(7),), a = x -> 1 / R2, g = x -> 0.0)
        b = zeros(dof.ndofs)
        dvals = Dict(g => uex(Xg[g]) for g in boundary_dofs(dof, mesh; tags = (Int8(8),)))
    else
        Krob = nothing
        brob = nothing
        b = zeros(dof.ndofs)
        dvals = Dict(g => (isfinite(Xg[g][1]) ? uex(Xg[g]) : 0.0) for g in boundary_dofs(dof, mesh))
    end

    niter = 0
    if solver === :cg
        t_solve = @elapsed ((u, stats) = solve_dirichlet_cg(dof, mesh, tq, b, dvals;
                                                            Krob = Krob, brob = brob, rtol = 1.0e-10))
        niter = stats.niter
        info(@sprintf("solved (matrix-free Jacobi-CG, %d iters, solved=%s)  (%.3f s)",
                      niter, stats.solved, t_solve))
    else
        t_asm = @elapsed K = assemble_stiffness(dof, mesh, tq)
        info(@sprintf("stiffness assembled: %d nonzeros  (%.3f s)", nnz(K), t_asm))
        Kt = Krob === nothing ? K : K + Krob
        bb = brob === nothing ? b : brob
        t_solve = @elapsed u = solve_dirichlet(Kt, bb, dvals, dof.ndofs)
        info(@sprintf("solved (sparse Cholesky)  (%.3f s)", t_solve))
    end

    linf = maximum(abs(u[g] - uex(Xg[g])) for g in eachindex(u) if isfinite(Xg[g][1]))
    info(@sprintf("L∞ error = %.3e", linf))
    return (; mesh, refel, dof, u, uex, d, R1, R2eff, ndofs = dof.ndofs, t_solve, niter, solver, linf, outer)
end

# Evaluate the solution and |error| on the z = 0 plane over an n×n grid covering
# the hole region [-W, W]²; points inside either ball (or outside a finite R2)
# are masked. Uses interpolate_field (point-location works on two_ball).
function slice_grid(res; n = 121)
    (; mesh, refel, dof, u, uex, d, R1, R2eff) = res
    a = d / 2
    N = refel.p + 1
    uloc = Matrix{Float64}(undef, nlocal(dof), mesh.Ne)
    gather!(uloc, u, dof)
    uarr = reshape(uloc, N, N, N, mesh.Ne)
    xs = refel.nodes
    W = a + 3 * R1
    gx = collect(range(-W, W; length = n))
    Uh = fill(NaN, n, n)
    Err = fill(NaN, n, n)
    for jy in 1:n, ix in 1:n
        x = gx[ix]
        y = gx[jy]
        (hypot(x - a, y) < R1 || hypot(x + a, y) < R1) && continue        # inside a ball
        (isfinite(R2eff) && hypot(x, y) > R2eff) && continue              # outside the outer sphere
        v = try
            interpolate_field(mesh, xs, uarr, SVector(x, y, 0.0); default = NaN)
        catch
            NaN
        end
        isfinite(v) || continue
        Uh[ix, jy] = v
        Err[ix, jy] = abs(v - uex(SVector(x, y, 0.0)))
    end
    return gx, Uh, Err
end

function main()
    mkpath(OUTDIR)
    printstyled("\nCG-SEM Laplace on the 3D two_ball (binary-excision) mesh\n"; color = :yellow, bold = true)
    printstyled("  u = 1/|x−p₁| + 1/|x−p₂|   (R1 = 1, d = 10; outer R = 100 or i⁰)\n"; color = :yellow)

    res_d = solve_case(; outer = :dirichlet)
    res_s = solve_case(; outer = :sommerfeld)
    res_c = solve_case(; outer = :compactified)

    # Same problems solved matrix-free with Jacobi-CG (no global matrix assembled).
    res_d_cg = solve_case(; outer = :dirichlet, solver = :cg)
    res_s_cg = solve_case(; outer = :sommerfeld, solver = :cg)
    res_c_cg = solve_case(; outer = :compactified, solver = :cg)

    phase("Summary")
    @printf("  %-26s %9s %12s %10s %13s\n", "outer boundary", "DOFs", "solver", "solve [s]", "L∞ error")
    for (name, r) in (("finite + Dirichlet", res_d), ("finite + Dirichlet", res_d_cg),
                      ("finite + Sommerfeld", res_s), ("finite + Sommerfeld", res_s_cg),
                      ("compactified (i⁰)", res_c), ("compactified (i⁰)", res_c_cg))
        tag = r.solver === :cg ? @sprintf("CG (%d it)", r.niter) : "Cholesky"
        @printf("  %-26s %9d %12s %10.3f %13.3e\n", name, r.ndofs, tag, r.t_solve, r.linf)
    end

    phase("z = 0 slice visualization (plane through both ball centres)")
    for (tag, r) in (("finite", res_d), ("compactified", res_c))
        gx, Uh, Err = slice_grid(r)
        logErr = map(e -> isfinite(e) && e > 0 ? log10(e) : NaN, Err)
        ps = save_heatmap(joinpath(OUTDIR, "two_ball_$(tag)_solution_z0.svg"), gx, Uh,
                          "u = 1/r₁ + 1/r₂ on z=0  ($tag)")
        pe = save_heatmap(joinpath(OUTDIR, "two_ball_$(tag)_error_z0.svg"), gx, logErr,
                          "log₁₀|error| on z=0  ($tag)")
        info("wrote $(relpath(ps))  and  $(relpath(pe))")
    end
    println()
    return nothing
end

main()
