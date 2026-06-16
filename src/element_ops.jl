# Per-element CG-SEM operator matrices: the local stiffness `Ke` and mass `Me`,
# built by OVER-INTEGRATION on a Gauss rule to remove the curvilinear
# under-integration ("aliasing") that caps diagonal-norm GLL collocation on
# curved elements.
#
# Weak forms in reference coordinates ξ ∈ [0,1]ᴰ with element map x(ξ),
# J = ∂x/∂ξ:
#
#   Ke[a,b] = ∫ ∇ₓφ_a · ∇ₓφ_b dx
#           = Σ_q w_q Σ_{i,j} (∂_ξᵢ φ_a)(ξ_q) · W_{ij}(ξ_q) · (∂_ξⱼ φ_b)(ξ_q),
#     with W = |det J| · J⁻¹ J⁻ᵀ  (contravariant metric × volume factor),
#   Me[a,b] = ∫ φ_a φ_b dx
#           = Σ_q w_q · |det J|(ξ_q) · φ_a(ξ_q) · φ_b(ξ_q).
#
# The basis is the tensor product of the 1D GLL Lagrange basis; its values and
# reference derivatives at the Gauss points are the tensor products of the
# `QuadratureRule`'s GLL→Gauss matrices `I` (value) and `Id` (derivative). The
# `(q+1)ᴰ` Gauss nodes are strictly interior, so a singular element face (the i⁰
# face of a compactified shell) is never sampled.
#
# Local node ordering is column-major over `(p+1)ᴰ`, identical to `DofHandler`'s
# `LinearIndices`, so `Ke`/`Me` rows align with `local2global[:, e]`.

using HexMeshes: Mesh, element_point_and_jac
using StaticArrays: SVector
using LinearAlgebra: det, inv, transpose, mul!

"""
    TensorQuadrature{D, T}

Element-independent tensor-product evaluation tables for the degree-`p` GLL
basis on the degree-`q` Gauss over-integration rule, in `D` dimensions. Built
once per `(ReferenceElement, QuadratureRule)` pair and reused for every element.

# Fields

* `nl :: Int` — local nodes per element, `(p+1)ᴰ`.
* `nq :: Int` — Gauss points per element, `(q+1)ᴰ`.
* `wref :: Vector{T}` — `(nq,)` reference quadrature weights (tensor product of
  the 1D Gauss weights).
* `V :: Matrix{T}` — `(nq, nl)` basis VALUES at the Gauss points.
* `B :: NTuple{D, Matrix{T}}` — `B[i]` is the `(nq, nl)` matrix of reference
  derivatives `∂_ξᵢ φ` at the Gauss points.
* `ξ :: Vector{SVector{D, T}}` — `(nq,)` Gauss points in `[0,1]ᴰ`, for evaluating
  the element map.
"""
struct TensorQuadrature{D, T}
    nl   :: Int
    nq   :: Int
    wref :: Vector{T}
    V    :: Matrix{T}
    B    :: NTuple{D, Matrix{T}}
    ξ    :: Vector{SVector{D, T}}
end

"""
    TensorQuadrature(refel::ReferenceElement{T}, qr::QuadratureRule{T}, ::Val{D})

Assemble the tensor-product value/derivative tables for `D` dimensions.
"""
function TensorQuadrature(refel::ReferenceElement{T}, qr::QuadratureRule{T}, ::Val{D}) where {T, D}
    nl1 = refel.p + 1
    nq1 = qr.q + 1
    ldims = ntuple(_ -> nl1, Val(D))
    qdims = ntuple(_ -> nq1, Val(D))
    lin_l = LinearIndices(ldims)
    lin_q = LinearIndices(qdims)
    nl = prod(ldims)
    nq = prod(qdims)
    Iv = qr.I
    Id = qr.Id
    gw = qr.weights
    gn = qr.nodes

    wref = Vector{T}(undef, nq)
    V = Matrix{T}(undef, nq, nl)
    B = ntuple(_ -> Matrix{T}(undef, nq, nl), Val(D))
    ξ = Vector{SVector{D, T}}(undef, nq)

    @inbounds for cq in CartesianIndices(qdims)
        qf = lin_q[cq]
        w = one(T)
        for d in 1:D
            w *= gw[cq[d]]
        end
        wref[qf] = w
        ξ[qf] = SVector{D, T}(ntuple(d -> gn[cq[d]], Val(D)))
        for cl in CartesianIndices(ldims)
            lf = lin_l[cl]
            v = one(T)
            for d in 1:D
                v *= Iv[cq[d], cl[d]]
            end
            V[qf, lf] = v
            for i in 1:D
                bi = one(T)
                for d in 1:D
                    bi *= d == i ? Id[cq[d], cl[d]] : Iv[cq[d], cl[d]]
                end
                B[i][qf, lf] = bi
            end
        end
    end

    return TensorQuadrature{D, T}(nl, nq, wref, V, B, ξ)
end

"""
    element_matrices(mesh::Mesh{D}, e, tq::TensorQuadrature{D, T}) → (Ke, Me)

Over-integrated local stiffness `Ke` and mass `Me` (`nl × nl`, dense, symmetric)
for element `e`, using the analytic element Jacobian at each Gauss point.
"""
function element_matrices(mesh::Mesh{D}, e::Integer, tq::TensorQuadrature{D, T}) where {D, T}
    nl = tq.nl
    nq = tq.nq
    # Per-Gauss-point weighted metric components cW[i,j][q] = w_q · |det J| · (J⁻¹J⁻ᵀ)_{ij}
    # and mass weight md[q] = w_q · |det J|. Stiffness then assembles as a sum of
    # D² gemms (no per-point scalar loop), `Ke = Σ_{ij} Bᵢᵀ (cW_{ij} ⊙ Bⱼ)`,
    # which is BLAS-bound for Float64; `Me = Vᵀ (md ⊙ V)`.
    md = Vector{T}(undef, nq)
    cW = Matrix{Vector{T}}(undef, D, D)
    for i in 1:D, j in 1:D
        cW[i, j] = Vector{T}(undef, nq)
    end
    @inbounds for qf in 1:nq
        _, J = element_point_and_jac(mesh, e, tq.ξ[qf])
        dJ = abs(det(J))
        Ji = inv(J)
        W = dJ * (Ji * transpose(Ji))
        w = tq.wref[qf]
        md[qf] = w * dJ
        for j in 1:D, i in 1:D
            cW[i, j][qf] = w * W[i, j]
        end
    end

    Ke = zeros(T, nl, nl)
    Me = zeros(T, nl, nl)
    tmp = Matrix{T}(undef, nq, nl)
    @inbounds for i in 1:D, j in 1:D
        tmp .= cW[i, j] .* tq.B[j]                       # scale rows of Bⱼ by cW_{ij}
        mul!(Ke, transpose(tq.B[i]), tmp, one(T), one(T)) # Ke += Bᵢᵀ · tmp
    end
    tmp .= md .* tq.V
    mul!(Me, transpose(tq.V), tmp, one(T), one(T))        # Me += Vᵀ · (md ⊙ V)

    return Ke, Me
end
