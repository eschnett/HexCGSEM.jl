# Direct solve of the assembled SPD elliptic system with Dirichlet constraints,
# by reduction to the free DOFs and a sparse Cholesky factorization (CHOLMOD).
# Conditioning-insensitive, so it reaches ~1e-12 even on the ill-conditioned
# compactified operator (where iterative solvers struggle).

using SparseArrays: SparseMatrixCSC
using LinearAlgebra: cholesky, Symmetric

"""
    solve_dirichlet(K, b, dirichlet::Dict{Int,T}, ndofs) → u::Vector{T}

Solve `K u = b` subject to `u[d] = dirichlet[d]` for the constrained DOFs, via
the reduced system on the free DOFs:

    K_FF u_F = b_F − K_FD u_D ,    u = u_D ⊕ u_F .

`K` must be symmetric positive (semi)definite; after removing the Dirichlet rows
`K_FF` is SPD and is factored with `cholesky`.
"""
function solve_dirichlet(K::SparseMatrixCSC{T}, b::AbstractVector{T},
                         dirichlet::Dict{Int, T}, ndofs::Integer) where {T}
    uD = zeros(T, ndofs)
    isdir = falses(ndofs)
    for (d, v) in dirichlet
        uD[d] = v
        isdir[d] = true
    end
    free = findall(!, isdir)
    rhs = b - K * uD
    Kff = K[free, free]
    uf = cholesky(Symmetric(Kff)) \ rhs[free]
    u = copy(uD)
    u[free] = uf
    return u
end
