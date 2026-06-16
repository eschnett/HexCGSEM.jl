# M2 — over-integrated element stiffness/mass. Key properties:
#   * symmetry of Ke, Me;
#   * constants in the kernel of the stiffness (Ke·1 = 0) — the sharpest
#     structural diagnostic, exact regardless of curvature/quadrature;
#   * mass is SPD and its entries sum to the element volume;
#   * AFFINE elements: over-integration is exact ⇒ Ke is q-independent, and the
#     linear patch test uᵀKe u = |α|²·vol holds;
#   * CURVED elements: low-q stiffness differs from rich-q (aliasing) and
#     converges as q rises — the whole point of over-integration.

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, element_matrices
using HexMeshes
using HexMeshes: element_point_and_jac
using StaticArrays
using LinearAlgebra: norm, eigvals, Symmetric, dot, det
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

# Physical coordinates of the GLL (DOF) nodes of element `e`.
function _gll_coords(mesh, refel, e, ::Val{D}) where {D}
    n = refel.p + 1
    dims = ntuple(_ -> n, Val(D))
    lin = LinearIndices(dims)
    X = Vector{SVector{D, Float64}}(undef, prod(dims))
    for cl in CartesianIndices(dims)
        ξ = SVector{D, Float64}(ntuple(d -> refel.nodes[cl[d]], Val(D)))
        X[lin[cl]], _ = element_point_and_jac(mesh, e, ξ)
    end
    return X
end

# Element volume ∫ |det J| dξ via the (rich) quadrature carried by `tq`.
function _elem_volume(mesh, e, tq)
    v = 0.0
    for qf in 1:tq.nq
        _, J = element_point_and_jac(mesh, e, tq.ξ[qf])
        v += tq.wref[qf] * abs(det(J))
    end
    return v
end

_Ke(mesh, e, refel, q, ::Val{D}) where {D} =
    element_matrices(mesh, e, TensorQuadrature(refel, QuadratureRule(refel, q), Val(D)))[1]

@testset "element ops (D=$D, $label)" for (D, label, meshof, affine) in (
    (1, "line", () -> make_uniform_line(Float64, 4, 0.0, 1.0), true),
    (2, "quad", () -> make_uniform_quad(Float64, 3, 3, 0.0, 1.0), true),
    (2, "annulus", () -> make_annulus_mesh(Float64, 1.0, 2.0, 3), false),
)
    mesh = meshof()
    @testset "p=$p" for p in (3, 5)
        refel = ReferenceElement(Float64, p)
        tq = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(D))
        _progress("D=$D $label p=$p")
        for e in (1, mesh.Ne)
            Ke, Me = element_matrices(mesh, e, tq)
            @test Ke ≈ transpose(Ke)
            @test Me ≈ transpose(Me)
            # Constants in the kernel of the Laplacian stiffness.
            @test norm(Ke * ones(size(Ke, 1))) ≤ 1.0e-9 * norm(Ke)
            # Consistent mass is SPD and its entries sum to the volume.
            @test all(>(0), eigvals(Symmetric(Me)))
            @test sum(Me) ≈ _elem_volume(mesh, e, tq) rtol = 1.0e-10
        end
    end

    if affine
        @testset "affine: q-independence + linear patch test" begin
            p = 4
            refel = ReferenceElement(Float64, p)
            e = 1
            Kep = _Ke(mesh, e, refel, p, Val(D))
            Ke2 = _Ke(mesh, e, refel, 2p, Val(D))
            # Over-integration is exact on affine elements ⇒ no q dependence.
            @test norm(Kep - Ke2) ≤ 1.0e-10 * norm(Ke2)
            # Patch test: u = α·x + c ⇒ uᵀ Ke u = |α|² · vol (exact).
            tq2 = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(D))
            X = _gll_coords(mesh, refel, e, Val(D))
            α = SVector{D, Float64}(ntuple(d -> 0.7d - 0.3, Val(D)))
            u = [dot(α, X[a]) + 1.23 for a in eachindex(X)]
            @test dot(u, Ke2 * u) ≈ sum(abs2, α) * _elem_volume(mesh, e, tq2) rtol = 1.0e-9
        end
    else
        @testset "curvilinear: over-integration de-aliases (converges in q)" begin
            p = 4
            refel = ReferenceElement(Float64, p)
            e = 1
            Kp = _Ke(mesh, e, refel, p, Val(D))
            K3p = _Ke(mesh, e, refel, 3p, Val(D))
            K4p = _Ke(mesh, e, refel, 4p, Val(D))
            @test norm(Kp - K4p) > 1.0e-9 * norm(K4p)     # aliasing present at q = p
            @test norm(K3p - K4p) < norm(Kp - K4p)        # and shrinking with q
        end
    end
end
