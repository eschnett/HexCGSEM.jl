# Global assembly of the CG-SEM operators: scatter the per-element matrices
# (`element_matrices`) into a global `SparseMatrixCSC` through the DOF map, and
# build the Galerkin load vector for a source term.

using HexMeshes: Mesh, element_point_and_jac
using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: det, BLAS

# Threaded direct-stiffness-summation core shared by `assemble_stiffness` and
# `assemble_mass`. Each element `e` owns the disjoint contiguous COO block at
# base offset `(e-1)·nl²`, so the element loop is embarrassingly parallel: we
# split `1:Ne` into one contiguous chunk per thread, each chunk filling its own
# triplets with thread-local scratch. The per-element kernels (`element_stiffness!`
# / `element_mass!`) are BLAS-bound, so when threading we pin BLAS to a single
# thread to avoid Julia-threads × BLAS-threads oversubscription; a serial call
# (one chunk) leaves BLAS untouched so it can still use multiple threads.
# Write order is independent of the chunking, so the result is identical (and
# `sparse` sums duplicates deterministically) regardless of thread count.
function _assemble_galerkin(fill_chunk!::F, dof::DofHandler{D}, mesh::Mesh{D},
                            tq::TensorQuadrature{D, T}) where {D, T, F}
    Ne = mesh.Ne
    cap = tq.nl * tq.nl * Ne
    Ii = Vector{Int}(undef, cap)
    Jj = Vector{Int}(undef, cap)
    Vv = Vector{T}(undef, cap)
    nchunks = max(1, min(Ne, Threads.nthreads()))
    bnds = round.(Int, range(0, Ne; length = nchunks + 1))
    blas0 = BLAS.get_num_threads()
    nchunks > 1 && BLAS.set_num_threads(1)
    try
        @sync for t in 1:nchunks
            lo = bnds[t] + 1
            hi = bnds[t + 1]
            lo > hi && continue
            Threads.@spawn fill_chunk!(Ii, Jj, Vv, lo, hi)
        end
    finally
        nchunks > 1 && BLAS.set_num_threads(blas0)
    end
    return sparse(Ii, Jj, Vv, dof.ndofs, dof.ndofs)
end

"""
    assemble_stiffness(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D,T})
        → SparseMatrixCSC{T,Int}

Global stiffness `K[i,j] = ∫ ∇φ_i · ∇φ_j`, assembled by direct stiffness
summation of the over-integrated element matrices (duplicate `(i,j)` triplets
are summed by `sparse`). Symmetric; positive semidefinite (the constant vector
is its null space) before boundary conditions are applied.

The element loop is multithreaded over `Threads.nthreads()`; start Julia with
`-t N` (or `JULIA_NUM_THREADS=N`) to use it.
"""
function assemble_stiffness(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T}) where {D, T}
    nl = tq.nl
    nl2 = nl * nl
    l2g = dof.local2global
    return _assemble_galerkin(dof, mesh, tq) do Ii, Jj, Vv, lo, hi
        Ke, tmp, cW = stiffness_scratch(tq)
        @inbounds for e in lo:hi
            element_stiffness!(Ke, tmp, cW, mesh, e, tq)
            c = (e - 1) * nl2
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
    end
end

"""
    assemble_mass(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D,T})
        → SparseMatrixCSC{T,Int}

Global consistent mass `M[i,j] = ∫ φ_i φ_j` (SPD). Used for weighted-L² error
norms and for the source projection of Helmholtz-type problems.

The element loop is multithreaded over `Threads.nthreads()`; start Julia with
`-t N` (or `JULIA_NUM_THREADS=N`) to use it.
"""
function assemble_mass(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T}) where {D, T}
    nl = tq.nl
    nl2 = nl * nl
    l2g = dof.local2global
    return _assemble_galerkin(dof, mesh, tq) do Ii, Jj, Vv, lo, hi
        Me, tmp, md = mass_scratch(tq)
        @inbounds for e in lo:hi
            element_mass!(Me, tmp, md, mesh, e, tq)
            c = (e - 1) * nl2
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
    end
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
