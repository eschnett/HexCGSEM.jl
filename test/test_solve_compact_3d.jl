# M8 (STEP 2) — 3D Laplace on the COMPACTIFIED shell: the outer boundary is at
# spatial infinity i⁰, where the exact asymptotic value is imposed as Dirichlet
# (no finite-R truncation, no Robin approximation). The element map is singular
# at i⁰, but over-integration samples only interior Gauss nodes and the i⁰ DOFs
# carry the Dirichlet value directly.
#
# u = 1/r is harmonic and → 0 at ∞; on the compactified coordinate it is in fact
# linear in a (since r = R1/(1−a) ⇒ 1/r = (1−a)/R1), so it is captured to high
# accuracy and converges fast.

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, dof_coords, boundary_dofs, solve_dirichlet
using HexMeshes
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

@testset "3D compactified shell, 1/r with exact Dirichlet at i⁰" begin
    R1 = 1.0
    uex(x) = 1 / sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    errs = Float64[]
    for p in (2, 4, 6)
        # inner sphere → tag 2 (Dirichlet u = 1/R1); i⁰ face → tag 1 (Dirichlet u = 0).
        mesh = make_compactified_shell_mesh(Float64, R1, 2; M_r = 2, inner_bc = :dirichlet)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        tq = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(3))
        K = assemble_stiffness(dof, mesh, tq)
        Xg = dof_coords(dof, mesh, refel)
        dvals = Dict{Int, Float64}()
        for d in boundary_dofs(dof, mesh; tags = (Int8(2),))
            dvals[d] = uex(Xg[d])           # inner sphere (finite coords)
        end
        for d in boundary_dofs(dof, mesh; tags = (Int8(1),))
            dvals[d] = 0.0                  # i⁰: exact asymptotic value
        end
        u = solve_dirichlet(K, zeros(dof.ndofs), dvals, dof.ndofs)
        # Error over the finite-coordinate DOFs (the i⁰ nodes sit at r = ∞).
        err = 0.0
        for g in eachindex(u)
            isfinite(Xg[g][1]) || continue
            err = max(err, abs(u[g] - uex(Xg[g])))
        end
        push!(errs, err)
        _progress("3D compactified 1/r p=$p  err=$err")
    end
    # 1/r is linear in the compactification parameter a and the boundary value
    # is imposed exactly at i⁰ ⇒ machine-precision recovery at every p, with NO
    # finite-R truncation floor. This is the step-2 payoff.
    @test maximum(errs) < 1.0e-11
end
