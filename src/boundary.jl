# Boundary handling: which global DOFs lie on tagged boundary faces (for
# Dirichlet data), and the Robin surface-integral assembly added later in this
# milestone.
#
# Face convention matches HexMeshes: face `fc ∈ 1..2D`, orthogonal reference
# axis `ax = (fc+1)÷2`, low side (`fc` odd) at local index 1 / `ξ_ax = 0`, high
# side (`fc` even) at local index `p+1` / `ξ_ax = 1`.

using HexMeshes: Mesh, element_point_and_jac
using SparseArrays: SparseMatrixCSC, sparse
using StaticArrays: SVector
using LinearAlgebra: norm, cross

"""
    boundary_dofs(dof::DofHandler{D}, mesh::Mesh{D}; tags = nothing) → Vector{Int}

Sorted global DOFs lying on geometric boundary faces (`neighbour == 0`). With
`tags` (a tuple/collection of `Int8` boundary tags) only faces whose
`mesh.conn.bdry` is in `tags` are included; `tags = nothing` takes all boundary
faces.
"""
function boundary_dofs(dof::DofHandler{D}, mesh::Mesh{D}; tags = nothing) where {D}
    n = dof.p + 1
    dims = ntuple(_ -> n, Val(D))
    lin = LinearIndices(dims)
    l2g = dof.local2global
    s = Set{Int}()
    @inbounds for e in 1:mesh.Ne, fc in 1:(2D)
        mesh.conn.neighbour[fc, e] == 0 || continue
        (tags === nothing || mesh.conn.bdry[fc, e] in tags) || continue
        ax = (fc + 1) ÷ 2
        sid = isodd(fc) ? 1 : n
        for cl in CartesianIndices(dims)
            cl[ax] == sid || continue
            push!(s, l2g[lin[cl], e])
        end
    end
    return sort!(collect(s))
end

# Surface measure |ds| at a boundary-face quadrature point, from the element
# Jacobian columns of the in-face tangent axes: a point (D=1, measure 1), an
# edge (D=2, |tangent|), or a quad (D=3, |t₁ × t₂|).
@inline _surface_measure(J, ::Tuple{}) = one(eltype(J))
@inline _surface_measure(J, t::NTuple{1, Int}) = norm(J[:, t[1]])
@inline _surface_measure(J, t::NTuple{2, Int}) = norm(cross(J[:, t[1]], J[:, t[2]]))

# Local flat indices and tangent-axis multi-indices of the nodes on face `fc`.
function _face_node_table(::Val{D}, n::Int, fc::Int) where {D}
    dims = ntuple(_ -> n, Val(D))
    lin = LinearIndices(dims)
    ax = (fc + 1) ÷ 2
    sid = isodd(fc) ? 1 : n
    taxes = ntuple(k -> (k < ax ? k : k + 1), Val(D - 1))   # the D−1 axes ≠ ax
    flats = Int[]
    tans = NTuple{D - 1, Int}[]
    for cl in CartesianIndices(dims)
        cl[ax] == sid || continue
        push!(flats, lin[cl])
        push!(tans, ntuple(k -> cl[taxes[k]], Val(D - 1)))
    end
    return ax, sid, taxes, flats, tans
end

"""
    assemble_robin(dof::DofHandler{D}, mesh::Mesh{D}, refel, qr; tags, a, g)
        → (Krob::SparseMatrixCSC, brob::Vector)

Assemble the Robin boundary contribution for the condition `a·u + ∂ₙu = g` on
faces with `bdry ∈ tags`. After integration by parts the weak form gains
`+∮ a u v` on the left (→ `Krob`) and `+∮ g v` on the right (→ `brob`):

    (K + Krob) u = b + brob .

`a(x)` and `g(x)` are functions of the physical boundary point. The surface
integral is over-integrated on the same Gauss rule `qr` (tangent directions),
with the analytic surface measure from the element Jacobian. A homogeneous
fall-off (`g = 0`) recovers a decaying solution that satisfies `∂ₙu = −a u`.
"""
function assemble_robin(dof::DofHandler{D}, mesh::Mesh{D}, refel::ReferenceElement{T},
                        qr::QuadratureRule{T}; tags, a, g) where {D, T}
    n = refel.p + 1
    Iv = qr.I
    gn = qr.nodes
    gw = qr.weights
    nq1 = qr.q + 1
    l2g = dof.local2global
    tables = ntuple(fc -> _face_node_table(Val(D), n, fc), Val(2D))
    tqdims = ntuple(_ -> nq1, Val(D - 1))

    Ii = Int[]
    Jj = Int[]
    Vv = T[]
    brob = zeros(T, dof.ndofs)

    @inbounds for e in 1:mesh.Ne, fc in 1:(2D)
        mesh.conn.neighbour[fc, e] == 0 || continue
        (tags === nothing || mesh.conn.bdry[fc, e] in tags) || continue
        ax, _, taxes, flats, tans = tables[fc]
        sξ = isodd(fc) ? zero(T) : one(T)
        nfn = length(flats)
        Vf = Vector{T}(undef, nfn)
        for cqt in CartesianIndices(tqdims)
            ξ = SVector{D, T}(ntuple(d -> (d == ax ? sξ : gn[cqt[d < ax ? d : d - 1]]), Val(D)))
            x, J = element_point_and_jac(mesh, e, ξ)
            wqf = one(T)
            for k in 1:(D - 1)
                wqf *= gw[cqt[k]]
            end
            fac = _surface_measure(J, taxes) * wqf
            ac = a(x)
            gc = g(x)
            for i in 1:nfn
                v = one(T)
                for k in 1:(D - 1)
                    v *= Iv[cqt[k], tans[i][k]]
                end
                Vf[i] = v
            end
            for i in 1:nfn
                ga = l2g[flats[i], e]
                brob[ga] += fac * gc * Vf[i]
                fa = fac * ac * Vf[i]
                for j in 1:nfn
                    push!(Ii, ga)
                    push!(Jj, l2g[flats[j], e])
                    push!(Vv, fa * Vf[j])
                end
            end
        end
    end
    return sparse(Ii, Jj, Vv, dof.ndofs, dof.ndofs), brob
end
