# M7 (STEP 1) — 3D Laplace on the spherical shell with an inner excision hole.
#   * monopole u = 1/r: inner Dirichlet u = 1/R1, outer matched Robin
#     ∂ₙu + u/R2 = 0 (exact for 1/r). Radial ⇒ fast spectral convergence.
#   * dipole u = z/r³: Dirichlet on both spheres (angular stress test).

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, dof_coords, boundary_dofs, assemble_robin, solve_dirichlet
using HexMeshes
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

@testset "3D shell, 1/r monopole (inner Dirichlet + outer Robin fall-off)" begin
    R1, R2 = 1.0, 2.0
    uex(x) = 1 / sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    errs = Float64[]
    for p in (2, 4, 6)
        mesh = make_radial_shell_mesh(Float64, R1, R2, 2; outer_bc = :sommerfeld, inner_bc = :dirichlet)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        qr = QuadratureRule(refel, 2p)
        tq = TensorQuadrature(refel, qr, Val(3))
        K = assemble_stiffness(dof, mesh, tq)
        Krob, brob = assemble_robin(dof, mesh, refel, qr; tags = (Int8(7),), a = x -> 1 / R2, g = x -> 0.0)
        Xg = dof_coords(dof, mesh, refel)
        dvals = Dict(d => uex(Xg[d]) for d in boundary_dofs(dof, mesh; tags = (Int8(2),)))
        u = solve_dirichlet(K + Krob, brob, dvals, dof.ndofs)
        push!(errs, maximum(abs(u[g] - uex(Xg[g])) for g in eachindex(u)))
        _progress("3D 1/r p=$p  err=$(errs[end])")
    end
    @test errs[end] < 1.0e-7
    @test errs[end] < errs[1] / 1000      # spectral collapse
end

@testset "3D shell, z/r³ dipole (Dirichlet), p-convergence" begin
    R1, R2 = 1.0, 2.0
    uex(x) = x[3] / (x[1]^2 + x[2]^2 + x[3]^2)^1.5
    errs = Float64[]
    for p in (2, 4, 6)
        mesh = make_radial_shell_mesh(Float64, R1, R2, 2; outer_bc = :dirichlet, inner_bc = :dirichlet)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        tq = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(3))
        K = assemble_stiffness(dof, mesh, tq)
        Xg = dof_coords(dof, mesh, refel)
        dvals = Dict(d => uex(Xg[d]) for d in boundary_dofs(dof, mesh))
        u = solve_dirichlet(K, zeros(dof.ndofs), dvals, dof.ndofs)
        push!(errs, maximum(abs(u[g] - uex(Xg[g])) for g in eachindex(u)))
        _progress("3D dipole p=$p  err=$(errs[end])")
    end
    @test errs[end] < 1.0e-5
    @test errs[end] < errs[1] / 100
end
