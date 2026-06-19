# Elliptic solve on the 3D two_ball (binary-excision) meshes — a ball minus two
# spherical holes (the 3D binary-black-hole excision domain), exercising all
# three outer-boundary treatments: finite Dirichlet, finite Sommerfeld fall-off
# Robin, and a compactified (R2 = Inf) domain with exact Dirichlet at i⁰.
#
# Manufactured harmonic "binary monopole" u = 1/|x − p₁| + 1/|x − p₂| with the
# 1/r singularities at the two ball centres p₁,₂ = (±d/2, 0, 0) (excised ⇒ u is
# analytic on the domain; decays as → 0 at infinity). Solve Δu = 0.
#
# Exercises the dimension-generic DofHandler on TrilinearHex patches and the 3D
# Robin surface integral on a complex multi-patch curved mesh — no operator
# changes needed beyond the shell/two_hole work.

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, dof_coords, boundary_dofs, assemble_robin, solve_dirichlet
using HexMeshes
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

# Max error over the finite-coordinate DOFs (the i⁰ face of a compactified mesh
# sits at infinity). The monopoles sit at the ball centres (±d/2, 0, 0), so they
# are excised; `R2` only sets the Sommerfeld coefficient.
function _two_ball_linf(mesh, p, d; sommerfeld = false, R2 = 100.0)
    a = d / 2
    uex(x) = 1 / hypot(x[1] - a, x[2], x[3]) + 1 / hypot(x[1] + a, x[2], x[3])
    refel = ReferenceElement(Float64, p)
    dof = DofHandler(mesh, p)
    qr = QuadratureRule(refel, 2p)
    tq = TensorQuadrature(refel, qr, Val(3))
    K = assemble_stiffness(dof, mesh, tq)
    Xg = dof_coords(dof, mesh, refel)
    if sommerfeld
        # Balls Dirichlet (tag 8); outer (tag 7) homogeneous fall-off Robin.
        Krob, brob = assemble_robin(dof, mesh, refel, qr; tags = (Int8(7),), a = x -> 1 / R2, g = x -> 0.0)
        Kt = K + Krob
        b = brob
        dvals = Dict(g => uex(Xg[g]) for g in boundary_dofs(dof, mesh; tags = (Int8(8),)))
    else
        # All boundaries Dirichlet. Finite-coord boundary DOFs (balls + finite
        # outer) get u_exact; i⁰ DOFs (∞ coords, compactified) get the exact
        # asymptotic value 0. Handles finite-Dirichlet and compactified alike.
        Kt = K
        b = zeros(dof.ndofs)
        dvals = Dict(g => (isfinite(Xg[g][1]) ? uex(Xg[g]) : 0.0) for g in boundary_dofs(dof, mesh))
    end
    u = solve_dirichlet(Kt, b, dvals, dof.ndofs)
    return maximum(abs(u[g] - uex(Xg[g])) for g in eachindex(u) if isfinite(Xg[g][1]))
end

@testset "two_ball binary-excision Laplace, separated ($label)" for (label, mesh, somm) in (
    ("finite Dirichlet",
     make_two_ball_mesh(Float64, 1.0, 100.0, 10.0, 2; M_h = 2, M_b = 2, M_i = 2, M_s = 2), false),
    ("finite Sommerfeld",
     make_two_ball_mesh(Float64, 1.0, 100.0, 10.0, 2; M_h = 2, M_b = 2, M_i = 2, M_s = 2,
                        outer_bc = :sommerfeld), true),
    ("compactified",
     make_two_ball_mesh(Float64, 1.0, Inf, 10.0, 2; M_h = 2, M_b = 2, M_i = 2, M_s = 2), false),
)
    errs = [_two_ball_linf(mesh, p, 10.0; sommerfeld = somm) for p in (2, 3, 4)]
    _progress("two_ball separated $label  Linf = $(round.(errs; sigdigits = 3))")
    @test issorted(errs; rev = true)     # converging in p
    @test errs[end] < errs[1] / 5        # clear (spectral) reduction
    @test errs[end] < 1.0e-2
end

@testset "two_ball binary-excision Laplace, touching (finite Dirichlet)" begin
    # Close balls (d = 4 ⇒ hole cubes meet at x = 0): the 42-patch :touching mode.
    mesh = make_two_ball_mesh(Float64, 1.0, 100.0, 4.0, 2; M_h = 2, M_b = 2, M_i = 2, M_s = 2,
                              mode = :touching)
    errs = [_two_ball_linf(mesh, p, 4.0) for p in (2, 3, 4)]
    _progress("two_ball touching  Linf = $(round.(errs; sigdigits = 3))")
    @test issorted(errs; rev = true)
    @test errs[end] < 1.0e-2
end
