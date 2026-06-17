# Elliptic solve on the 2D binary-excision (two-hole) meshes — a disk minus two
# circular holes (the 2D binary-black-hole excision domain), in both the
# :separated (well-separated holes) and :touching (close holes) modes.
#
# Manufactured harmonic "binary monopole" u = log|x − p₁| + log|x − p₂| with the
# log singularities at the two hole centres (so they are excised and u is
# analytic on the closed domain ⇒ spectral convergence). Solve Δu = 0 with
# Dirichlet u = u_exact on every boundary (both holes + outer circle).
#
# This is the integration test that exercises the dimension-generic DofHandler
# on BilinearQuad patches and irregular-valence vertices — neither of which the
# shell/annulus solves reach.

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, dof_coords, boundary_dofs, solve_dirichlet
using HexMeshes
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

# (mode, hole separation d, final-accuracy floor). d = 2 would be the degenerate
# tangent limit for R1 = 1 (:touching needs d > 2·R1), so the close case uses 4.
@testset "two-hole binary-excision Laplace ($mode, d=$d)" for (mode, d, tol) in (
    (:separated, 10.0, 1.0e-4),
    (:touching, 4.0, 5.0e-6),
)
    R1, R2 = 1.0, 100.0
    p1 = (d / 2, 0.0)
    p2 = (-d / 2, 0.0)
    uex(x) = log(hypot(x[1] - p1[1], x[2] - p1[2])) + log(hypot(x[1] - p2[1], x[2] - p2[2]))
    errs = Float64[]
    for p in (2, 4, 6)
        mesh = make_two_hole_mesh(Float64, R1, R2, d, 4;
                                  M_h = 3, M_b = 3, M_i = 3, M_s = 4, mode = mode)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        tq = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(2))
        K = assemble_stiffness(dof, mesh, tq)
        Xg = dof_coords(dof, mesh, refel)
        dvals = Dict(g => uex(Xg[g]) for g in boundary_dofs(dof, mesh))
        u = solve_dirichlet(K, zeros(dof.ndofs), dvals, dof.ndofs)
        push!(errs, maximum(abs(u[g] - uex(Xg[g])) for g in eachindex(u)))
        _progress("two-hole $mode d=$d p=$p  ndofs=$(dof.ndofs)  Linf=$(errs[end])")
    end
    @test errs[end] < tol
    @test errs[end] < errs[1] / 100      # spectral collapse
end
