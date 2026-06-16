# Physical coordinates of the spectral-element nodes — both in the per-element
# local layout and reduced to the global DOF vector. Used to set boundary data
# and to evaluate manufactured/analytic solutions at the DOFs.

using HexMeshes: Mesh, element_point_and_jac
using StaticArrays: SVector

"""
    node_coords(mesh::Mesh{D}, refel::ReferenceElement{T}) → Matrix{SVector{D,T}}

Physical coordinates of every local GLL node, shape `(nlocal, Ne)`, in the same
column-major local ordering as `DofHandler` and `TensorQuadrature`.
"""
function node_coords(mesh::Mesh{D}, refel::ReferenceElement{T}) where {D, T}
    n = refel.p + 1
    dims = ntuple(_ -> n, Val(D))
    lin = LinearIndices(dims)
    X = Matrix{SVector{D, T}}(undef, prod(dims), mesh.Ne)
    @inbounds for e in 1:mesh.Ne, cl in CartesianIndices(dims)
        ξ = SVector{D, T}(ntuple(d -> refel.nodes[cl[d]], Val(D)))
        X[lin[cl], e], _ = element_point_and_jac(mesh, e, ξ)
    end
    return X
end

"""
    dof_coords(dof::DofHandler{D}, mesh::Mesh{D}, refel) → Vector{SVector{D,T}}

Physical coordinate of each global DOF (`ndofs` of them). For shared DOFs any
representative node is used; they are coincident by construction (M1).
"""
function dof_coords(dof::DofHandler{D}, mesh::Mesh{D}, refel::ReferenceElement{T}) where {D, T}
    X = node_coords(mesh, refel)
    Xg = Vector{SVector{D, T}}(undef, dof.ndofs)
    @inbounds for e in axes(dof.local2global, 2), nd in axes(dof.local2global, 1)
        Xg[dof.local2global[nd, e]] = X[nd, e]
    end
    return Xg
end
