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
* `I1 :: Matrix{T}` — the `(nq1, nl1) = (q+1, p+1)` 1D GLL→Gauss VALUE
  interpolation (`= qr.I`); the per-axis factor of `V`.
* `Id1 :: Matrix{T}` — the 1D GLL→Gauss DERIVATIVE interpolation (`= qr.Id`); the
  per-axis derivative factor of the `B[i]`. Kept so the tensor-product `V`/`B` can
  be applied matrix-free by sum-factorization (1D contractions) instead of as
  dense `(nq, nl)` matrices.
"""
struct TensorQuadrature{D, T}
    nl   :: Int
    nq   :: Int
    wref :: Vector{T}
    V    :: Matrix{T}
    B    :: NTuple{D, Matrix{T}}
    ξ    :: Vector{SVector{D, T}}
    I1   :: Matrix{T}
    Id1  :: Matrix{T}
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

    return TensorQuadrature{D, T}(nl, nq, wref, V, B, ξ, copy(Iv), copy(Id))
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

"""
    stiffness_scratch(tq::TensorQuadrature{D, T}) → (Ke, tmp, cW)

Allocate the reusable work buffers for [`element_stiffness!`](@ref): the output
matrix `Ke` (`nl × nl`), the gemm scratch `tmp` (`nq × nl`), and the metric
buffer `cW` (`D × D` vectors of length `nq`). Allocate once per thread/chunk and
reuse across that chunk's elements to avoid per-element allocation.
"""
function stiffness_scratch(tq::TensorQuadrature{D, T}) where {D, T}
    Ke = Matrix{T}(undef, tq.nl, tq.nl)
    tmp = Matrix{T}(undef, tq.nq, tq.nl)
    cW = Matrix{Vector{T}}(undef, D, D)
    for i in 1:D, j in 1:D
        cW[i, j] = Vector{T}(undef, tq.nq)
    end
    return Ke, tmp, cW
end

"""
    element_metric!(cW, mesh::Mesh{D}, e, tq::TensorQuadrature{D, T}) → cW

Fill the weighted contravariant metric `cW[i,j][q] = w_q · |det J| · (J⁻¹J⁻ᵀ)_{ij}`
at the Gauss points of element `e` (the geometric factor common to the stiffness
weak form). `cW` is a `D×D` matrix of length-`nq` vectors (see
[`stiffness_scratch`](@ref)); it is symmetric in `(i,j)`.
"""
function element_metric!(cW, mesh::Mesh{D}, e::Integer, tq::TensorQuadrature{D, T}) where {D, T}
    @inbounds for qf in 1:tq.nq
        _, J = element_point_and_jac(mesh, e, tq.ξ[qf])
        dJ = abs(det(J))
        Ji = inv(J)
        W = dJ * (Ji * transpose(Ji))
        w = tq.wref[qf]
        for j in 1:D, i in 1:D
            cW[i, j][qf] = w * W[i, j]
        end
    end
    return cW
end

"""
    element_stiffness!(Ke, tmp, cW, mesh::Mesh{D}, e, tq::TensorQuadrature{D, T}) → Ke

Stiffness-only variant of [`element_matrices`](@ref): compute just `Ke` into the
caller-provided buffers (see [`stiffness_scratch`](@ref)), skipping the mass
matrix. Allocation-free, so it can be called in a tight (threaded) loop.
"""
function element_stiffness!(Ke, tmp, cW, mesh::Mesh{D}, e::Integer,
                            tq::TensorQuadrature{D, T}) where {D, T}
    element_metric!(cW, mesh, e, tq)
    fill!(Ke, zero(T))
    @inbounds for i in 1:D, j in 1:D
        tmp .= cW[i, j] .* tq.B[j]
        mul!(Ke, transpose(tq.B[i]), tmp, one(T), one(T))
    end
    return Ke
end

"""
    mass_scratch(tq::TensorQuadrature{D, T}) → (Me, tmp, md)

Allocate the reusable work buffers for [`element_mass!`](@ref): the output matrix
`Me` (`nl × nl`), the gemm scratch `tmp` (`nq × nl`), and the mass-weight buffer
`md` (length `nq`). Allocate once per thread/chunk and reuse.
"""
function mass_scratch(tq::TensorQuadrature{D, T}) where {D, T}
    Me = Matrix{T}(undef, tq.nl, tq.nl)
    tmp = Matrix{T}(undef, tq.nq, tq.nl)
    md = Vector{T}(undef, tq.nq)
    return Me, tmp, md
end

"""
    element_mass!(Me, tmp, md, mesh::Mesh{D}, e, tq::TensorQuadrature{D, T}) → Me

Mass-only variant of [`element_matrices`](@ref): compute just `Me` into the
caller-provided buffers (see [`mass_scratch`](@ref)), skipping the stiffness
matrix. Allocation-free.
"""
function element_mass!(Me, tmp, md, mesh::Mesh{D}, e::Integer,
                       tq::TensorQuadrature{D, T}) where {D, T}
    nq = tq.nq
    @inbounds for qf in 1:nq
        _, J = element_point_and_jac(mesh, e, tq.ξ[qf])
        md[qf] = tq.wref[qf] * abs(det(J))
    end
    fill!(Me, zero(T))
    tmp .= md .* tq.V
    mul!(Me, transpose(tq.V), tmp, one(T), one(T))
    return Me
end

# ------------------------------------------------------------------------------
# Sum-factorization: apply the element stiffness operator `Ke = Σ_{i,j} Bᵢᵀ cW_ij Bⱼ`
# matrix-free, by 1D tensor contractions, instead of forming the dense `nl×nl` Ke.
# Each `Bᵢ` is the tensor product of the 1D operators `Id1` (derivative) on axis `i`
# and `I1` (value interp) on the others, so applying it is a sequence of D 1D
# matrix multiplies (`_contract!`). Per element this costs O(D²·nq·nl1) work and
# O(nq) storage (just the metric `cW`), versus O(nl²) for the dense gemv/storage —
# the difference that makes high-`p` 3D feasible.

# Number of unique components of a symmetric D×D tensor.
_nsym(D::Integer) = (D * (D + 1)) ÷ 2

"""
    _symidx(::Val{D}) → Matrix{Int}

`D×D` table mapping `(i, j)` to a packed symmetric-component index `1:_nsym(D)`
(`s[i,j] == s[j,i]`), used to store/read the symmetric metric `cW`.
"""
function _symidx(::Val{D}) where {D}
    s = Matrix{Int}(undef, D, D)
    c = 0
    @inbounds for j in 1:D, i in 1:j
        c += 1
        s[i, j] = c
        s[j, i] = c
    end
    return s
end

# Apply the separable operator (M[D] ⊗ … ⊗ M[1]) to `src` (column-major, size
# n^D) giving `dst` (size m^D), where every `M[d]` is `m×n`, via one 1D multiply
# per axis. `buf1`/`buf2` are exact-length scratch vectors (see `sumfac_scratch`);
# all reshapes are of full-length `Vector`s, so they stay `Array`s and the
# multiplies hit BLAS.
@inline function _contract!(dst, src, M::NTuple{1}, buf1, buf2, ::Val{1})
    mul!(dst, M[1], src)
    return dst
end

@inline function _contract!(dst, src, M::NTuple{2}, buf1, buf2, ::Val{2})
    m = size(M[1], 1)
    n = size(M[1], 2)
    T1 = reshape(buf1, m, n)
    mul!(T1, M[1], reshape(src, n, n))                 # axis 1
    mul!(reshape(dst, m, m), T1, transpose(M[2]))      # axis 2
    return dst
end

@inline function _contract!(dst, src, M::NTuple{3}, buf1, buf2, ::Val{3})
    m = size(M[1], 1)
    n = size(M[1], 2)
    mul!(reshape(buf1, m, n * n), M[1], reshape(src, n, n * n))   # axis 1
    T1 = reshape(buf1, m, n, n)
    T2 = reshape(buf2, m, m, n)
    @inbounds for k in 1:n                                        # axis 2 (per slice)
        @views mul!(T2[:, :, k], T1[:, :, k], transpose(M[2]))
    end
    mul!(reshape(dst, m * m, m), reshape(buf2, m * m, n), transpose(M[3]))   # axis 3
    return dst
end

"""
    SumFacScratch{D, T}

Per-thread reusable buffers for the sum-factorization element apply
([`sumfac_apply!`](@ref)) and diagonal ([`sumfac_diag_element!`](@ref)). Allocate
once per thread/chunk with [`sumfac_scratch`](@ref) and reuse across elements.
"""
struct SumFacScratch{D, T}
    xin  :: Vector{T}             # (nl) contiguous copy of the element input
    yout :: Vector{T}             # (nl) accumulator for the element output
    acc  :: Vector{T}             # (nl) single-term contraction result
    g    :: NTuple{D, Vector{T}}  # (nq) reference gradients at Gauss points
    f    :: NTuple{D, Vector{T}}  # (nq) metric-weighted gradients
    fb1  :: Vector{T}             # forward-contraction scratch (nl1→nq1)
    fb2  :: Vector{T}
    tb1  :: Vector{T}             # transpose-contraction scratch (nq1→nl1)
    tb2  :: Vector{T}
end

"""
    sumfac_scratch(I1::Matrix{T}, nl, nq, ::Val{D}) → SumFacScratch{D, T}

Allocate the [`SumFacScratch`](@ref) buffers from the 1D operator `I1` (its
`(nq1, nl1) = (q+1, p+1)` shape) and the `D`-dimensional sizes `nl = nl1^D`,
`nq = nq1^D`. The forward/transpose contraction buffers are sized to the exact
intermediate lengths so every reshape stays a BLAS-compatible `Array`.
"""
function sumfac_scratch(I1::Matrix{T}, nl::Integer, nq::Integer, ::Val{D}) where {D, T}
    nl1 = size(I1, 2)
    nq1 = size(I1, 1)
    fb1 = Vector{T}(undef, nq1 * nl1^(D - 1))
    fb2 = Vector{T}(undef, D >= 3 ? nq1^2 * nl1^(D - 2) : 0)
    tb1 = Vector{T}(undef, nl1 * nq1^(D - 1))
    tb2 = Vector{T}(undef, D >= 3 ? nl1^2 * nq1^(D - 2) : 0)
    g = ntuple(_ -> Vector{T}(undef, nq), Val(D))
    f = ntuple(_ -> Vector{T}(undef, nq), Val(D))
    return SumFacScratch{D, T}(Vector{T}(undef, nl), Vector{T}(undef, nl),
                               Vector{T}(undef, nl), g, f, fb1, fb2, tb1, tb2)
end

"""
    sumfac_apply!(ycol, xcol, G, e, I1, Id1, I1t, Id1t, sidx, sc) → ycol

Matrix-free element stiffness apply `ycol = Kₑ · xcol` by sum-factorization, where
`xcol`/`ycol` are the element's columns of the local layout, `G[:, sidx[i,j], e]`
is the stored symmetric metric `cW_ij`, `I1`/`Id1` are the 1D value/derivative
operators and `I1t`/`Id1t` their transposes. Equals the dense `element_stiffness!`'s
`Kₑ · xcol` to round-off.
"""
function sumfac_apply!(ycol, xcol, G, e, I1, Id1, I1t, Id1t, sidx,
                       sc::SumFacScratch{D, T}) where {D, T}
    copyto!(sc.xin, xcol)
    # Step 1: reference gradients g_j = B_j x at the Gauss points.
    @inbounds for j in 1:D
        Mf = ntuple(d -> d == j ? Id1 : I1, Val(D))
        _contract!(sc.g[j], sc.xin, Mf, sc.fb1, sc.fb2, Val(D))
    end
    # Step 2: apply the metric, f_i[q] = Σ_j cW_ij[q] · g_j[q].
    @inbounds for i in 1:D
        fi = sc.f[i]
        fill!(fi, zero(T))
        for j in 1:D
            gj = sc.g[j]
            cw = view(G, :, sidx[i, j], e)
            @simd for q in eachindex(fi)
                fi[q] += cw[q] * gj[q]
            end
        end
    end
    # Step 3: integrate against test gradients, y = Σ_i B_iᵀ f_i.
    fill!(sc.yout, zero(T))
    @inbounds for i in 1:D
        Mt = ntuple(d -> d == i ? Id1t : I1t, Val(D))
        _contract!(sc.acc, sc.f[i], Mt, sc.tb1, sc.tb2, Val(D))
        @simd for a in eachindex(sc.yout)
            sc.yout[a] += sc.acc[a]
        end
    end
    copyto!(ycol, sc.yout)
    return ycol
end

"""
    sumfac_diag_element!(dcol, G, e, Pt, sidx, sc) → dcol

Element diagonal `dcol[a] = Kₑ[a,a]`, matrix-free. Uses
`diag(Bᵢᵀ cW_ij Bⱼ)[a] = Σ_q Bᵢ[q,a] cW_ij[q] Bⱼ[q,a]`, which — because `Bᵢ`,`Bⱼ`
are tensor products — is the contraction of `cW_ij` against the per-axis
elementwise products `P^{ij}_d`. `Pt = (PIIt, PIIdt, PIdIdt)` holds the transposed
1D factors `transpose(I1.⊙I1)`, `transpose(I1.⊙Id1)`, `transpose(Id1.⊙Id1)`.
"""
function sumfac_diag_element!(dcol, G, e, Pt, sidx, sc::SumFacScratch{D, T}) where {D, T}
    PIIt, PIIdt, PIdIdt = Pt
    fill!(sc.yout, zero(T))
    @inbounds for i in 1:D, j in 1:D
        Mp = ntuple(d -> (d == i) & (d == j) ? PIdIdt : ((d == i) | (d == j) ? PIIdt : PIIt), Val(D))
        copyto!(sc.f[1], view(G, :, sidx[i, j], e))           # contiguous source for reshape
        _contract!(sc.acc, sc.f[1], Mp, sc.tb1, sc.tb2, Val(D))
        @simd for a in eachindex(sc.yout)
            sc.yout[a] += sc.acc[a]
        end
    end
    copyto!(dcol, sc.yout)
    return dcol
end
