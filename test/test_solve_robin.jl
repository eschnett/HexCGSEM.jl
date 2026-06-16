# M3 (Robin) — the fall-off boundary. A homogeneous Robin a·u + ∂ₙu = 0 at the
# outer circle with a = k/R2, plus inner Dirichlet from the analytic solution,
# must recover the decaying multipole cos(kθ)/r^k (which satisfies exactly
# ∂_r u = −(k/r) u). This exercises the surface-integral assembly and the
# matched-fall-off BC that M7's outer sphere uses in 3D.

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, dof_coords, boundary_dofs, assemble_robin, solve_dirichlet
using HexMeshes
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

@testset "2D Laplace, homogeneous Robin fall-off recovers multipole" begin
    k = 2
    R1, R2 = 1.0, 2.0
    uex(x) = real(complex(x[1], x[2])^(-k))
    errs = Float64[]
    for p in (2, 4, 6, 8, 10)
        # outer = :sommerfeld ⇒ tag 7 (used here for Robin); inner = :dirichlet ⇒ tag 2.
        mesh = make_annulus_mesh(Float64, R1, R2, 4; outer_bc = :sommerfeld, inner_bc = :dirichlet)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        qr = QuadratureRule(refel, 2p)
        tq = TensorQuadrature(refel, qr, Val(2))
        K = assemble_stiffness(dof, mesh, tq)
        Krob, brob = assemble_robin(dof, mesh, refel, qr; tags = (Int8(7),),
                                    a = x -> k / R2, g = x -> 0.0)
        Kt = K + Krob
        Xg = dof_coords(dof, mesh, refel)
        dvals = Dict(d => uex(Xg[d]) for d in boundary_dofs(dof, mesh; tags = (Int8(2),)))
        u = solve_dirichlet(Kt, brob, dvals, dof.ndofs)
        push!(errs, maximum(abs(u[g] - uex(Xg[g])) for g in eachindex(u)))
        _progress("2D Robin multipole k=$k p=$p  err=$(errs[end])")
    end
    @test errs[end] < 1.0e-9
    @test errs[end] < errs[1] / 100
end
