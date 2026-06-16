# Continuous (C⁰) global degree-of-freedom numbering for a tensor-product
# spectral-element mesh, plus the gather/scatter operations that move data
# between the per-element local layout and the global DOF vector.
#
# Dimension-generic in `D ∈ {1, 2, 3}`. The construction is **entity-by-
# dimension**: each local node is classified by `k`, the number of its axes
# that are interior (`2 ≤ iₐ ≤ p`):
#
#   * `k = 0`            — a corner → a mesh VERTEX (dim-0 entity). The global
#                          DOF is just the mesh's deduplicated vertex id
#                          (`mesh.vertex_idx`), so vertices need no work and no
#                          floating-point coordinate matching.
#   * `0 < k < D`        — a shared sub-entity: an EDGE (`k = 1`, `p−1` nodes) or
#                          a FACE (`k = 2`, `D = 3`; `(p−1)²` nodes, D₄-oriented).
#   * `k = D`            — the cell INTERIOR (`(p−1)ᴰ` nodes), never shared.
#
# Sharing is made consistent across elements WITHOUT coordinate comparison by
# keying each shared entity on the GLOBAL VERTEX IDS of its corners and choosing
# a canonical traversal from the smaller id. For an edge with endpoint vertex
# ids `(vₐ, v_b)` the interior node at distance `m` from the local low corner
# maps to canonical slot `m` if `vₐ < v_b`, else `p − m` — so the two elements
# that share the edge agree regardless of which way each traverses it. (The
# `canonical_orientation = false` switch disables the flip; it exists only so the
# tests can confirm that the flip is actually load-bearing.)
#
# The corner→ξ ordering used by `mesh.vertex_idx` is the Gmsh tensor-product
# order, verified against `bilinear_shape`/`trilinear_shape`.

using HexMeshes: Mesh, nv

# Gmsh corner sign patterns: for each corner index `c`, the per-axis endpoint
# (0 = low → local index 1, 1 = high → local index p+1). Verified empirically.
@inline _corner_signs(::Val{1}) = ((0,), (1,))
@inline _corner_signs(::Val{2}) = ((0, 0), (1, 0), (1, 1), (0, 1))
@inline _corner_signs(::Val{3}) =
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))

# Global vertex id of a face corner, given the pinned axis `axp` (sign `sgnp`)
# and the two in-face tangent axes `(t1, t2)` set to signs `(s1, s2)`.
@inline function _face_corner(csigns, vidx, e, axp, sgnp, t1, t2, s1, s2, ::Val{D}) where {D}
    s = ntuple(d -> d == axp ? sgnp : (d == t1 ? s1 : s2), Val(D))
    c = findfirst(==(s), csigns)
    return @inbounds vidx[c, e]
end

@inline _along(axis, o1, o2, la, lb, p) =
    axis == 1 ? (o1 == 0 ? la : p - la) : (o2 == 0 ? lb : p - lb)

# Canonical `(p−1)²`-block slot for a 3D face interior node, derived ONLY from
# the face's four corner global vertex ids (so both elements sharing the face
# agree, regardless of their local D₄ orientation). `la, lb ∈ 1..p−1` are the
# node's distances from the (t1-low, t2-low) corner. Canonical frame: origin =
# minimum-id corner; u-axis toward its smaller-id edge neighbour, v-axis toward
# the larger.
function _face_slot(ids::NTuple{4, Int}, la::Int, lb::Int, p::Int)
    pos = ((0, 0), (1, 0), (0, 1), (1, 1))     # (s1, s2) of corners 1..4 on (t1, t2)
    o = findmin(ids)[2]
    o1, o2 = pos[o]
    n1 = 0
    n2 = 0
    for c in 1:4
        c == o && continue
        if ((pos[c][1] != o1) + (pos[c][2] != o2)) == 1   # edge-adjacent to O
            n1 == 0 ? (n1 = c) : (n2 = c)
        end
    end
    A = ids[n1] < ids[n2] ? n1 : n2
    B = ids[n1] < ids[n2] ? n2 : n1
    axisA = pos[A][1] != o1 ? 1 : 2
    axisB = pos[B][1] != o1 ? 1 : 2
    ca = _along(axisA, o1, o2, la, lb, p)
    cb = _along(axisB, o1, o2, la, lb, p)
    return (ca - 1) * (p - 1) + cb
end

# Sort a 4-tuple (sorting network) — the face dict key.
@inline function _sort4(t::NTuple{4, Int})
    a, b, c, d = t
    a, b = minmax(a, b)
    c, d = minmax(c, d)
    a, c = minmax(a, c)
    b, d = minmax(b, d)
    b, c = minmax(b, c)
    return (a, b, c, d)
end

"""
    DofHandler{D}

Global C⁰ DOF numbering for a degree-`p` tensor-product spectral-element mesh in
`D` dimensions.

# Fields

* `p :: Int` — polynomial degree (so `n = p + 1` nodes per axis).
* `nlocal :: Int` — local nodes per element, `(p+1)ᴰ`.
* `ndofs :: Int` — total number of global DOFs.
* `local2global :: Matrix{Int}` — `(nlocal, Ne)`; maps each element's local node
  (column-major tensor-product index) to its global DOF. This single array IS
  the gather/scatter map.
* `multiplicity :: Vector{Int}` — `(ndofs,)`; how many local nodes map to each
  global DOF (`= diag(QᵀQ)`): 1 for interior nodes, ≥2 for shared entities.
"""
struct DofHandler{D}
    p            :: Int
    nlocal       :: Int
    ndofs        :: Int
    local2global :: Matrix{Int}
    multiplicity :: Vector{Int}
end

ndofs(dof::DofHandler) = dof.ndofs
nlocal(dof::DofHandler) = dof.nlocal

"""
    DofHandler(mesh::Mesh{D}, p::Int; canonical_orientation = true) → DofHandler{D}

Build the global DOF numbering for degree `p` on `mesh`, for `D ∈ {1, 2, 3}`
(1D: vertices + interiors; 2D: + edges; 3D: + faces).
"""
function DofHandler(mesh::Mesh{D}, p::Int; canonical_orientation::Bool = true) where {D}
    p ≥ 1 || throw(ArgumentError("polynomial degree p must be ≥ 1, got $p"))
    n = p + 1
    Ne = mesh.Ne
    dims = ntuple(_ -> n, Val(D))
    lin = LinearIndices(dims)
    nl = prod(dims)
    Nv = nv(mesh)
    csigns = _corner_signs(Val(D))

    local2global = zeros(Int, nl, Ne)
    edge_dofs = Dict{Tuple{Int, Int}, UnitRange{Int}}()
    face_dofs = Dict{NTuple{4, Int}, UnitRange{Int}}()
    ndof = Nv                                  # vertex DOFs occupy 1..Nv

    @inbounds for e in 1:Ne
        for ci in CartesianIndices(dims)
            nd = lin[ci]
            # Count interior axes; remember the last one (used when k == 1).
            k = 0
            dint = 0
            for d in 1:D
                id = ci[d]
                if 1 < id < n
                    k += 1
                    dint = d
                end
            end

            if k == 0
                # Vertex: global DOF is the mesh's canonical vertex id.
                s = ntuple(d -> ci[d] == 1 ? 0 : 1, Val(D))
                c = findfirst(==(s), csigns)
                local2global[nd, e] = mesh.vertex_idx[c, e]
            elseif k == D
                # Cell interior: a fresh, unshared DOF.
                ndof += 1
                local2global[nd, e] = ndof
            elseif k == 1
                # Edge: shared, keyed by its endpoint vertex ids.
                slo = ntuple(a -> a == dint ? 0 : (ci[a] == 1 ? 0 : 1), Val(D))
                shi = ntuple(a -> a == dint ? 1 : (ci[a] == 1 ? 0 : 1), Val(D))
                clo = findfirst(==(slo), csigns)
                chi = findfirst(==(shi), csigns)
                va = mesh.vertex_idx[clo, e]
                vb = mesh.vertex_idx[chi, e]
                key = va < vb ? (va, vb) : (vb, va)
                if !haskey(edge_dofs, key)
                    edge_dofs[key] = (ndof + 1):(ndof + (p - 1))
                    ndof += p - 1
                end
                block = edge_dofs[key]
                m = ci[dint] - 1                       # distance from the local low corner, 1..p−1
                slot = (canonical_orientation && va > vb) ? (p - m) : m
                local2global[nd, e] = block[slot]
            else
                # k = 2 shared face (only reachable for D = 3): one pinned axis,
                # two in-face tangent axes, keyed by the 4 corner vertex ids.
                axp = (ci[1] == 1 || ci[1] == n) ? 1 : ((ci[2] == 1 || ci[2] == n) ? 2 : 3)
                t1 = axp == 1 ? 2 : 1
                t2 = axp == 3 ? 2 : 3
                sgnp = ci[axp] == 1 ? 0 : 1
                ids = (_face_corner(csigns, mesh.vertex_idx, e, axp, sgnp, t1, t2, 0, 0, Val(D)),
                       _face_corner(csigns, mesh.vertex_idx, e, axp, sgnp, t1, t2, 1, 0, Val(D)),
                       _face_corner(csigns, mesh.vertex_idx, e, axp, sgnp, t1, t2, 0, 1, Val(D)),
                       _face_corner(csigns, mesh.vertex_idx, e, axp, sgnp, t1, t2, 1, 1, Val(D)))
                key = _sort4(ids)
                if !haskey(face_dofs, key)
                    face_dofs[key] = (ndof + 1):(ndof + (p - 1)^2)
                    ndof += (p - 1)^2
                end
                block = face_dofs[key]
                la = ci[t1] - 1
                lb = ci[t2] - 1
                slot = canonical_orientation ? _face_slot(ids, la, lb, p) : ((la - 1) * (p - 1) + lb)
                local2global[nd, e] = block[slot]
            end
        end
    end

    mult = zeros(Int, ndof)
    @inbounds for g in local2global
        mult[g] += 1
    end
    @assert all(>(0), local2global) "every local node must be assigned a DOF"

    return DofHandler{D}(p, nl, ndof, local2global, mult)
end

"""
    gather!(ul::AbstractMatrix, ug::AbstractVector, dof::DofHandler) → ul

Scatter the global DOF vector `ug` into the per-element local layout:
`ul[n, e] = ug[local2global[n, e]]`. `ul` has shape `(nlocal, Ne)`.
"""
function gather!(ul::AbstractMatrix, ug::AbstractVector, dof::DofHandler)
    l2g = dof.local2global
    size(ul) == size(l2g) || throw(DimensionMismatch("ul must be (nlocal, Ne)"))
    @inbounds for e in axes(l2g, 2), nd in axes(l2g, 1)
        ul[nd, e] = ug[l2g[nd, e]]
    end
    return ul
end

"""
    scatter_add!(ug::AbstractVector, ul::AbstractMatrix, dof::DofHandler) → ug

Direct stiffness summation: `ug[g] = Σ ul[n, e]` over all local nodes `(n, e)`
with `local2global[n, e] = g`. `ug` is zeroed first. This is the transpose
(`Qᵀ`) of [`gather!`](@ref)'s `Q`.
"""
function scatter_add!(ug::AbstractVector, ul::AbstractMatrix, dof::DofHandler)
    l2g = dof.local2global
    size(ul) == size(l2g) || throw(DimensionMismatch("ul must be (nlocal, Ne)"))
    length(ug) == dof.ndofs || throw(DimensionMismatch("ug must have length ndofs"))
    fill!(ug, zero(eltype(ug)))
    @inbounds for e in axes(l2g, 2), nd in axes(l2g, 1)
        ug[l2g[nd, e]] += ul[nd, e]
    end
    return ug
end
