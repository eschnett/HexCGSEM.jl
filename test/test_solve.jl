# M3 — first end-to-end solves with spectral (p-)convergence to high accuracy.
# 1D Poisson with a manufactured source (the simplest full-pipeline check),
# then 2D Laplace on the curvilinear annulus against analytic harmonic
# solutions (monopole log r and a decaying multipole). All Dirichlet here; the
# Robin/fall-off boundary is exercised in test_solve_robin.jl.

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, assemble_load, dof_coords, boundary_dofs, solve_dirichlet
using HexMeshes
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

_maxerr(u, uex, Xg) = maximum(abs(u[g] - uex(Xg[g])) for g in eachindex(u))

@testset "1D Poisson (Dirichlet), p-convergence" begin
    uex(x) = sin(2 * x[1]) + 0.3 * x[1]
    f(x) = 4 * sin(2 * x[1])            # −u'' = f
    M = 4
    errs = Float64[]
    for p in (2, 4, 6, 8)
        mesh = make_uniform_line(Float64, M, 0.0, 1.0)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        tq = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(1))
        K = assemble_stiffness(dof, mesh, tq)
        b = assemble_load(dof, mesh, tq, f)
        Xg = dof_coords(dof, mesh, refel)
        dvals = Dict(d => uex(Xg[d]) for d in boundary_dofs(dof, mesh))
        u = solve_dirichlet(K, b, dvals, dof.ndofs)
        push!(errs, _maxerr(u, uex, Xg))
        _progress("1D Poisson p=$p  err=$(errs[end])")
    end
    @test errs[end] < 1.0e-11
    @test errs[end] < errs[1] / 100      # spectral collapse
end

@testset "2D Laplace, monopole log r (annulus, Dirichlet), p-convergence" begin
    uex(x) = log(sqrt(x[1]^2 + x[2]^2))    # harmonic in 2D
    errs = Float64[]
    for p in (2, 4, 6, 8)
        mesh = make_annulus_mesh(Float64, 1.0, 2.0, 2; outer_bc = :dirichlet, inner_bc = :dirichlet)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        tq = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(2))
        K = assemble_stiffness(dof, mesh, tq)
        b = zeros(dof.ndofs)
        Xg = dof_coords(dof, mesh, refel)
        dvals = Dict(d => uex(Xg[d]) for d in boundary_dofs(dof, mesh))
        u = solve_dirichlet(K, b, dvals, dof.ndofs)
        push!(errs, _maxerr(u, uex, Xg))
        _progress("2D log r p=$p  err=$(errs[end])")
    end
    @test errs[end] < 1.0e-10
    @test errs[end] < errs[1] / 100
end

@testset "2D Laplace, decaying multipole (annulus, Dirichlet), p-convergence" begin
    k = 2
    uex(x) = real(complex(x[1], x[2])^(-k))   # harmonic; decays like r^{-k}
    errs = Float64[]
    for p in (2, 4, 6, 8, 10)
        mesh = make_annulus_mesh(Float64, 1.0, 2.0, 4; outer_bc = :dirichlet, inner_bc = :dirichlet)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        tq = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(2))
        K = assemble_stiffness(dof, mesh, tq)
        b = zeros(dof.ndofs)
        Xg = dof_coords(dof, mesh, refel)
        dvals = Dict(d => uex(Xg[d]) for d in boundary_dofs(dof, mesh))
        u = solve_dirichlet(K, b, dvals, dof.ndofs)
        push!(errs, _maxerr(u, uex, Xg))
        _progress("2D multipole k=$k p=$p  err=$(errs[end])")
    end
    @test errs[end] < 1.0e-10
    @test errs[end] < errs[1] / 100
end
