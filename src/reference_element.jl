# Reference-element data for tensor-product spectral elements. Everything here
# is 1D — the D-dimensional CG-SEM operators (later milestones) are tensor
# products of these 1D objects. Two pieces:
#
#   * `ReferenceElement` — the Gauss–Lobatto–Legendre (GLL) nodal basis of
#     degree `p` (the solution/DOF nodes), on the reference interval `[0, 1]`.
#     GLL nodes are used for the DOFs because they include the element
#     endpoints, which is what makes the continuous (C⁰) inter-element gluing in
#     `dofmap.jl` possible.
#
#   * `QuadratureRule` — a Gauss–Legendre over-integration rule of degree `q`,
#     plus the matrices that interpolate GLL nodal values (and their reference
#     derivatives) to the Gauss points. Over-integration with `q > p` is what
#     removes the curvilinear under-integration ("aliasing") that caps the
#     diagonal-norm GLL-collocation accuracy on curved elements. Gauss nodes are
#     strictly interior to `(0, 1)`, so a singular element face (e.g. the i⁰
#     face of a compactified shell at `ξ = 1`) is never sampled by the
#     integrand — this is what makes the compactified solve well-defined.
#
# All node sets live on `[0, 1]` to match `HexMeshes`' element parameterisation
# (`element_point_and_jac(mesh, e, ξ)` expects `ξ ∈ [0, 1]`). `PolynomialBases`
# builds its bases on `[-1, 1]`; we map nodes via `x = (ξ + 1)/2`, scale the
# quadrature weights by `1/2` (`∫₀¹ = ½ ∫₋₁¹`) and the differentiation matrix by
# `2` (`d/dx_{[0,1]} = 2 · d/dξ_{[-1,1]}`).

using PolynomialBases: LobattoLegendre, GaussLegendre, interpolation_matrix

"""
    default_tol(::Type{T}) → T

Precision-scaled tolerance `sqrt(eps(T))`, matching the convention used across
the HexMeshes / HexSBPSAT test suites. Used as the loose tolerance for
convergence checks; exactness checks use a tighter `eps`-scaled bound directly.
"""
default_tol(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))

"""
    ReferenceElement{T}

Gauss–Lobatto–Legendre nodal basis of polynomial degree `p` (hence `p + 1`
collocation nodes) on the reference interval `[0, 1]`.

# Fields

* `p :: Int` — polynomial degree.
* `nodes :: Vector{T}` — the `p + 1` GLL nodes on `[0, 1]` (endpoints included).
* `weights :: Vector{T}` — GLL quadrature weights on `[0, 1]` (the lumped/diagonal
  mass; exact for polynomials of degree `≤ 2p − 1`).
* `D :: Matrix{T}` — the `(p+1)×(p+1)` differentiation matrix w.r.t. the `[0, 1]`
  coordinate: `(D * u)[i]` is the derivative of the degree-`p` nodal interpolant
  of `u` at node `i`.
* `baryweights :: Vector{T}` — barycentric weights of the GLL nodes (used to
  build interpolation matrices).
"""
struct ReferenceElement{T<:AbstractFloat}
    p           :: Int
    nodes       :: Vector{T}
    weights     :: Vector{T}
    D           :: Matrix{T}
    baryweights :: Vector{T}
end

"""
    ReferenceElement(::Type{T}, p::Int) → ReferenceElement{T}

Build the degree-`p` GLL reference element in precision `T`.
"""
function ReferenceElement(::Type{T}, p::Int) where {T<:AbstractFloat}
    p ≥ 1 || throw(ArgumentError("polynomial degree p must be ≥ 1, got $p"))
    b = LobattoLegendre(p, T)               # on [-1, 1], p+1 nodes
    nodes = (b.nodes .+ one(T)) ./ 2         # → [0, 1]
    weights = b.weights ./ 2                 # ∫₀¹ = ½ ∫₋₁¹
    D = b.D .* 2                             # d/dx_{[0,1]} = 2 · d/dξ_{[-1,1]}
    return ReferenceElement{T}(p, nodes, weights, D, copy(b.baryweights))
end

"""
    QuadratureRule{T}

Gauss–Legendre over-integration rule of degree `q` (hence `q + 1` interior
nodes, exact for polynomials of degree `≤ 2q + 1`) on `[0, 1]`, paired with a
`ReferenceElement` of degree `p` via the GLL→Gauss interpolation matrices.

# Fields

* `q :: Int` — quadrature degree.
* `nodes :: Vector{T}` — the `q + 1` Gauss nodes on `(0, 1)` (strictly interior).
* `weights :: Vector{T}` — Gauss weights on `[0, 1]`.
* `I :: Matrix{T}` — `(q+1)×(p+1)` value interpolation: `(I * u)[g]` is the
  degree-`p` GLL interpolant of `u` evaluated at Gauss node `g` (`= ℓ_j(ξ_g)`).
* `Id :: Matrix{T}` — `(q+1)×(p+1)` derivative interpolation: `(Id * u)[g]` is the
  reference derivative of that interpolant at Gauss node `g` (`= ℓ_j'(ξ_g)`),
  equal to `I * D`.
"""
struct QuadratureRule{T<:AbstractFloat}
    q       :: Int
    nodes   :: Vector{T}
    weights :: Vector{T}
    I       :: Matrix{T}
    Id      :: Matrix{T}
end

"""
    QuadratureRule(refel::ReferenceElement{T}, q::Int = 2 * refel.p) → QuadratureRule{T}

Build the degree-`q` Gauss over-integration rule for `refel`. The default
`q = 2p` is a generous over-integration (exact through degree `4p + 1`), enough
to drive the curvilinear-metric aliasing toward round-off for the analytic
cubed-sphere maps; pass a smaller `q ≥ p` to trade accuracy for cost.
"""
function QuadratureRule(refel::ReferenceElement{T}, q::Int = 2 * refel.p) where {T<:AbstractFloat}
    q ≥ refel.p || throw(ArgumentError("quadrature degree q=$q must be ≥ element degree p=$(refel.p)"))
    g = GaussLegendre(q, T)                  # on [-1, 1], q+1 interior nodes
    nodes = (g.nodes .+ one(T)) ./ 2         # → (0, 1)
    weights = g.weights ./ 2                 # ∫₀¹ = ½ ∫₋₁¹
    # Value interpolation GLL→Gauss. Lagrange interpolation is invariant under
    # the affine [-1,1]↔[0,1] reparameterisation, so building it on the native
    # `[-1, 1]` nodes gives the same matrix as on `[0, 1]`.
    gll = LobattoLegendre(refel.p, T)
    Imat = interpolation_matrix(g.nodes, gll)
    # Derivative interpolation w.r.t. the [0,1] coordinate. `refel.D` is already
    # the [0,1] differentiation matrix, so `ℓ_j'(ξ_g) = (I · D)[g, j]` exactly
    # (the degree-(p−1) derivative is exactly recovered from its GLL values).
    Id = Imat * refel.D
    return QuadratureRule{T}(q, nodes, weights, Imat, Id)
end
