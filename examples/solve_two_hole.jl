# Example: solve the 2D binary-excision Laplace problem on the two-hole meshes
# (a disk minus two circular holes) in both :separated (well-separated) and
# :touching (close) modes. Manufactured harmonic "binary monopole"
#   u = log|x − p₁| + log|x − p₂|
# whose log singularities sit at the two hole centres (so they are excised and u
# is analytic on the domain ⇒ spectral convergence). Solve Δu = 0 with Dirichlet
# u = u_exact on every boundary (both holes + outer circle).
#
# Prints progress + summary statistics and writes SVG scatter plots of the
# per-element nodal solution and pointwise error, zoomed to the hole region.
#
# Run from the package root:
#   julia --project=examples examples/solve_two_hole.jl

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, dof_coords, boundary_dofs, solve_dirichlet, solve_dirichlet_cg,
                node_coords, gather!, nlocal
using HexMeshes
using HexMeshes: nv, npatches
using SparseArrays: nnz
using Printf

include(joinpath(@__DIR__, "svg.jl"))

const OUTDIR = joinpath(@__DIR__, "output")

phase(label) = (printstyled("\n▶ ", label, "\n"; color = :cyan, bold = true); flush(stdout))
info(msg) = (println("    ", msg); flush(stdout))

# `solver = :cholesky` assembles the global stiffness matrix and factors it
# (sparse Cholesky); `solver = :cg` never assembles it — the operator is applied
# matrix-free and the system solved with Jacobi-preconditioned CG.
function solve_case(; mode, d, R1 = 1.0, R2 = 100.0, M = 4, p = 7, solver = :cholesky)
    p1 = (d / 2, 0.0)
    p2 = (-d / 2, 0.0)
    uex(x) = log(hypot(x[1] - p1[1], x[2] - p1[2])) + log(hypot(x[1] - p2[1], x[2] - p2[2]))
    phase(mode === :separated ? "Well-separated holes (d = $d, :separated)" :
                                "Close holes (d = $d, :touching)")

    t_mesh = @elapsed mesh = make_two_hole_mesh(Float64, R1, R2, d, M;
                                                M_h = 3, M_b = 3, M_i = 3, M_s = 4, mode = mode)
    info(@sprintf("mesh: %d patches, %d elements, %d vertices  (%.3f s)",
                  npatches(mesh), mesh.Ne, nv(mesh), t_mesh))

    refel = ReferenceElement(Float64, p)
    dof = DofHandler(mesh, p)
    tq = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(2))
    info(@sprintf("degree p = %d,  global DOFs = %d", p, dof.ndofs))

    Xg = dof_coords(dof, mesh, refel)
    dvals = Dict(g => uex(Xg[g]) for g in boundary_dofs(dof, mesh))   # Dirichlet on all boundaries
    niter = 0
    if solver === :cg
        t_solve = @elapsed ((u, stats) = solve_dirichlet_cg(dof, mesh, tq, zeros(dof.ndofs), dvals;
                                                            rtol = 1.0e-10))
        niter = stats.niter
        info(@sprintf("solved (matrix-free Jacobi-CG, %d iters)  (%.3f s)", niter, t_solve))
    else
        t_asm = @elapsed K = assemble_stiffness(dof, mesh, tq)
        info(@sprintf("stiffness assembled: %d nonzeros  (%.3f s)", nnz(K), t_asm))
        t_solve = @elapsed u = solve_dirichlet(K, zeros(dof.ndofs), dvals, dof.ndofs)
        info(@sprintf("solved (sparse Cholesky)  (%.3f s)", t_solve))
    end

    linf = maximum(abs(u[g] - uex(Xg[g])) for g in eachindex(u))
    info(@sprintf("L∞ error = %.3e", linf))
    return (; mesh, refel, dof, u, uex, ndofs = dof.ndofs, t_solve, niter, solver, linf)
end

# Per-node physical coords, solution value, and |error| (for scatter plots).
function nodal_fields(res)
    (; mesh, refel, dof, u, uex) = res
    X = node_coords(mesh, refel)
    ul = Matrix{Float64}(undef, nlocal(dof), mesh.Ne)
    gather!(ul, u, dof)
    xs = Float64[]
    ys = Float64[]
    vals = Float64[]
    err = Float64[]
    for e in 1:mesh.Ne, n in 1:nlocal(dof)
        P = X[n, e]
        push!(xs, P[1])
        push!(ys, P[2])
        push!(vals, ul[n, e])
        push!(err, abs(ul[n, e] - uex(P)))
    end
    return xs, ys, vals, err
end

function main()
    mkpath(OUTDIR)
    printstyled("\nCG-SEM Laplace on the 2D two-hole (binary-excision) meshes\n";
                color = :yellow, bold = true)
    printstyled("  u = log|x−p₁| + log|x−p₂|   (outer R = 100, holes r = 1)\n"; color = :yellow)

    res_sep = solve_case(; mode = :separated, d = 10.0)
    res_tou = solve_case(; mode = :touching, d = 4.0)

    # Same problems solved matrix-free with Jacobi-CG (no global matrix assembled).
    res_sep_cg = solve_case(; mode = :separated, d = 10.0, solver = :cg)
    res_tou_cg = solve_case(; mode = :touching, d = 4.0, solver = :cg)

    phase("Summary")
    @printf("  %-24s %9s %12s %10s %13s\n", "setup", "DOFs", "solver", "solve [s]", "L∞ error")
    for (name, r) in (("well-separated (d=10)", res_sep), ("well-separated (d=10)", res_sep_cg),
                      ("close (d=4)", res_tou), ("close (d=4)", res_tou_cg))
        tag = r.solver === :cg ? @sprintf("CG (%d it)", r.niter) : "Cholesky"
        @printf("  %-24s %9d %12s %10.3f %13.3e\n", name, r.ndofs, tag, r.t_solve, r.linf)
    end

    phase("Scatter-plot visualization (per-element nodes, zoomed on the holes)")
    for (tag, r, win) in (("separated", res_sep, 8.0), ("touching", res_tou, 5.0))
        xs, ys, vals, err = nodal_fields(r)
        logerr = map(e -> e > 0 ? log10(e) : NaN, err)   # Dirichlet nodes (err=0) drop out
        ps = save_scatter(joinpath(OUTDIR, "two_hole_$(tag)_solution.svg"),
                          xs, ys, vals, "u = log r₁ + log r₂  ($tag)";
                          xlim = (-win, win), ylim = (-win, win), r = 1.6)
        pe = save_scatter(joinpath(OUTDIR, "two_hole_$(tag)_error.svg"),
                          xs, ys, logerr, "log₁₀|error|  ($tag)";
                          xlim = (-win, win), ylim = (-win, win), r = 1.6)
        info("wrote $(relpath(ps))  and  $(relpath(pe))")
    end
    println()
    return nothing
end

main()
