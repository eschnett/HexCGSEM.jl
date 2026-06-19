# Matrix-free Jacobi-preconditioned CG (`solve_dirichlet_cg`) must reproduce the
# assembled-and-factored path (`solve_dirichlet`): first that the matrix-free
# `StiffnessOperator` and `stiffness_diagonal` match `assemble_stiffness` and its
# diagonal to round-off, then that the full CG solve matches the Cholesky solve
# and reaches the same spectral accuracy on Dirichlet and Robin problems.

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, assemble_load, dof_coords, boundary_dofs, assemble_robin,
                solve_dirichlet, solve_dirichlet_cg, StiffnessOperator, stiffness_diagonal,
                scatter_add!
using HexMeshes
using LinearAlgebra: norm, mul!, diag
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

_maxerr(u, uex, Xg) = maximum(abs(u[g] - uex(Xg[g])) for g in eachindex(u))

@testset "matrix-free operator matches assembled K ($(Threads.nthreads()) threads)" begin
    @testset "$label (D=$D)" for (D, label, meshof) in (
        (2, "annulus", () -> make_annulus_mesh(Float64, 1.0, 2.0, 4)),
        (3, "two_ball separated", () -> make_two_ball_mesh(Float64, 1.0, 100.0, 10.0, 2;
                                                           M_h = 2, M_b = 2, M_i = 2, M_s = 2,
                                                           outer_bc = :dirichlet, mode = :separated)),
    )
        mesh = meshof()
        @testset "p=$p" for p in (3, 4)
            refel = ReferenceElement(Float64, p)
            dof = DofHandler(mesh, p)
            tq = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(D))
            _progress("D=$D $label p=$p  (Ne=$(mesh.Ne), ndofs=$(dof.ndofs))")

            K = assemble_stiffness(dof, mesh, tq)
            free = collect(1:dof.ndofs)
            op = StiffnessOperator(dof, mesh, tq; free = free)

            # Operator action equals K·x for random x (full free set = all DOFs).
            x = randn(dof.ndofs)
            y = similar(x)
            mul!(y, op, x)
            @test y ≈ K * x rtol = 1.0e-10

            # Matrix-free diagonal equals diag(K) exactly (same scatter).
            @test stiffness_diagonal(op) == diag(K)

            # Gather/scatter plumbing: scattering all-ones locals gives multiplicity.
            ones_local = ones(dof.nlocal, mesh.Ne)
            mult = zeros(Float64, dof.ndofs)
            scatter_add!(mult, ones_local, dof)
            @test mult == dof.multiplicity
        end
    end
end

# CG must match the Cholesky solution and the manufactured solution; report the
# iteration count and verify a true (unpreconditioned) backward error.
@testset "1D Poisson (Dirichlet): CG matches Cholesky, p-convergence" begin
    uex(x) = sin(2 * x[1]) + 0.3 * x[1]
    f(x) = 4 * sin(2 * x[1])
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
        uchol = solve_dirichlet(K, b, dvals, dof.ndofs)
        ucg, stats = solve_dirichlet_cg(dof, mesh, tq, b, dvals; rtol = 1.0e-12, verbose = false)
        push!(errs, _maxerr(ucg, uex, Xg))
        _progress("1D Poisson p=$p  niter=$(stats.niter)  err=$(errs[end])")
        @test stats.solved
        @test ucg ≈ uchol rtol = 1.0e-9
    end
    @test errs[end] < 1.0e-10
    @test errs[end] < errs[1] / 100
end

@testset "2D Laplace (annulus, Dirichlet): CG matches Cholesky, p-convergence" begin
    uex(x) = log(sqrt(x[1]^2 + x[2]^2))
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
        uchol = solve_dirichlet(K, b, dvals, dof.ndofs)
        ucg, stats = solve_dirichlet_cg(dof, mesh, tq, b, dvals; rtol = 1.0e-12, verbose = false)
        push!(errs, _maxerr(ucg, uex, Xg))
        _progress("2D log r p=$p  niter=$(stats.niter)  err=$(errs[end])")
        @test stats.solved
        @test ucg ≈ uchol rtol = 1.0e-9
    end
    @test errs[end] < 1.0e-9
    @test errs[end] < errs[1] / 100
end

@testset "2D Laplace (annulus, Robin fall-off): CG matches Cholesky" begin
    k = 2
    R1, R2 = 1.0, 2.0
    uex(x) = real(complex(x[1], x[2])^(-k))
    for p in (4, 8)
        mesh = make_annulus_mesh(Float64, R1, R2, 4; outer_bc = :sommerfeld, inner_bc = :dirichlet)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        qr = QuadratureRule(refel, 2p)
        tq = TensorQuadrature(refel, qr, Val(2))
        K = assemble_stiffness(dof, mesh, tq)
        Krob, brob = assemble_robin(dof, mesh, refel, qr; tags = (Int8(7),), a = x -> k / R2, g = x -> 0.0)
        Xg = dof_coords(dof, mesh, refel)
        dvals = Dict(d => uex(Xg[d]) for d in boundary_dofs(dof, mesh; tags = (Int8(2),)))
        uchol = solve_dirichlet(K + Krob, brob, dvals, dof.ndofs)
        ucg, stats = solve_dirichlet_cg(dof, mesh, tq, zeros(dof.ndofs), dvals;
                                        Krob = Krob, brob = brob, rtol = 1.0e-12, verbose = false)
        _progress("2D Robin multipole k=$k p=$p  niter=$(stats.niter)  err=$(_maxerr(ucg, uex, Xg))")
        @test stats.solved
        @test ucg ≈ uchol rtol = 1.0e-9
    end
end

@testset "singular system without Dirichlet or Robin is rejected" begin
    mesh = make_uniform_line(Float64, 4, 0.0, 1.0)
    refel = ReferenceElement(Float64, 3)
    dof = DofHandler(mesh, 3)
    tq = TensorQuadrature(refel, QuadratureRule(refel, 6), Val(1))
    @test_throws ArgumentError solve_dirichlet_cg(dof, mesh, tq, zeros(dof.ndofs), Dict{Int, Float64}())
end
