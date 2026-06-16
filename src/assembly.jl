# Global assembly of the CG-SEM operators: scatter the per-element matrices
# (`element_matrices`) into a global `SparseMatrixCSC` through the DOF map, and
# build the Galerkin load vector for a source term.

using HexMeshes: Mesh, element_point_and_jac
using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: det

"""
    assemble_stiffness(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D,T})
        → SparseMatrixCSC{T,Int}

Global stiffness `K[i,j] = ∫ ∇φ_i · ∇φ_j`, assembled by direct stiffness
summation of the over-integrated element matrices (duplicate `(i,j)` triplets
are summed by `sparse`). Symmetric; positive semidefinite (the constant vector
is its null space) before boundary conditions are applied.
"""
function assemble_stiffness(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T}) where {D, T}
    nl = tq.nl
    Ne = mesh.Ne
    cap = nl * nl * Ne
    Ii = Vector{Int}(undef, cap)
    Jj = Vector{Int}(undef, cap)
    Vv = Vector{T}(undef, cap)
    l2g = dof.local2global
    c = 0
    @inbounds for e in 1:Ne
        Ke, _ = element_matrices(mesh, e, tq)
        for b in 1:nl
            gb = l2g[b, e]
            for a in 1:nl
                c += 1
                Ii[c] = l2g[a, e]
                Jj[c] = gb
                Vv[c] = Ke[a, b]
            end
        end
    end
    return sparse(Ii, Jj, Vv, dof.ndofs, dof.ndofs)
end

"""
    assemble_mass(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D,T})
        → SparseMatrixCSC{T,Int}

Global consistent mass `M[i,j] = ∫ φ_i φ_j` (SPD). Used for weighted-L² error
norms and for the source projection of Helmholtz-type problems.
"""
function assemble_mass(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T}) where {D, T}
    nl = tq.nl
    Ne = mesh.Ne
    cap = nl * nl * Ne
    Ii = Vector{Int}(undef, cap)
    Jj = Vector{Int}(undef, cap)
    Vv = Vector{T}(undef, cap)
    l2g = dof.local2global
    c = 0
    @inbounds for e in 1:Ne
        _, Me = element_matrices(mesh, e, tq)
        for b in 1:nl
            gb = l2g[b, e]
            for a in 1:nl
                c += 1
                Ii[c] = l2g[a, e]
                Jj[c] = gb
                Vv[c] = Me[a, b]
            end
        end
    end
    return sparse(Ii, Jj, Vv, dof.ndofs, dof.ndofs)
end

"""
    assemble_load(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D,T}, f)
        → Vector{T}

Galerkin load `b[i] = ∫ f φ_i`, with `f(x::SVector{D})` evaluated at the (rich)
Gauss points and integrated with the element measure `|det J|`.
"""
function assemble_load(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T}, f) where {D, T}
    nl = tq.nl
    rhs = zeros(T, dof.ndofs)
    l2g = dof.local2global
    le = Vector{T}(undef, nl)
    @inbounds for e in 1:mesh.Ne
        fill!(le, zero(T))
        for qf in 1:tq.nq
            x, J = element_point_and_jac(mesh, e, tq.ξ[qf])
            fac = tq.wref[qf] * abs(det(J)) * f(x)
            for a in 1:nl
                le[a] += fac * tq.V[qf, a]
            end
        end
        for a in 1:nl
            rhs[l2g[a, e]] += le[a]
        end
    end
    return rhs
end
