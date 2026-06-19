# Solving the SPD elliptic system with Dirichlet constraints, by reduction to the
# free DOFs. Two paths:
#
#   * `solve_dirichlet`     — assemble the global `K` and factor it with sparse
#                             Cholesky (CHOLMOD). Conditioning-insensitive, so it
#                             reaches ~1e-12 even on the ill-conditioned
#                             compactified operator; the robust fallback.
#   * `solve_dirichlet_cg`  — never assemble `K`: apply it matrix-free through the
#                             DOF map (`y = Qᵀ·blockdiag(Kₑ)·Q·x`) and solve with
#                             Jacobi-preconditioned CG. Avoids both the serial
#                             COO→CSC build and the Cholesky fill-in, so it scales
#                             to far larger / higher-order meshes.

using SparseArrays: SparseMatrixCSC
using LinearAlgebra: LinearAlgebra, cholesky, Symmetric, Diagonal, diag, mul!, BLAS
using HexMeshes: Mesh
using Krylov: cg

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

"""
    StiffnessOperator{D, T}

Matrix-free representation of the global stiffness operator restricted to a set
of free DOFs, for use as the `A` in a Krylov solve. Applying it never forms the
global `K`: it gathers the free-DOF vector into the per-element layout, applies
the precomputed dense element matrices `Kₑ` (`Qᵀ·blockdiag(Kₑ)·Q`), and scatters
back, adding the (small, boundary-only) Robin matrix `Krob` when present.

# Fields

* `dof :: DofHandler{D}` — the gather/scatter map.
* `Ke :: Vector{Matrix{T}}` — `Ne` dense `nl×nl` element stiffness matrices,
  precomputed once via [`element_stiffness!`](@ref).
* `Krob :: Union{Nothing, SparseMatrixCSC{T, Int}}` — optional Robin contribution.
* `free :: Vector{Int}` — global indices of the free (non-Dirichlet) DOFs; the
  operator acts on vectors of length `length(free)`.
* `u_full, y_full :: Vector{T}` — `(ndofs,)` full-space scratch.
* `ul, yl :: Matrix{T}` — `(nl, Ne)` per-element scratch.

Construct with [`StiffnessOperator(dof, mesh, tq; free, Krob)`](@ref).
"""
struct StiffnessOperator{D, T}
    dof    :: DofHandler{D}
    Ke     :: Vector{Matrix{T}}
    Krob   :: Union{Nothing, SparseMatrixCSC{T, Int}}
    free   :: Vector{Int}
    u_full :: Vector{T}
    y_full :: Vector{T}
    ul     :: Matrix{T}
    yl     :: Matrix{T}
end

"""
    StiffnessOperator(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T};
                      free::Vector{Int}, Krob = nothing) → StiffnessOperator{D, T}

Precompute the per-element stiffness matrices and the scratch buffers for a
matrix-free operator on the free DOFs `free`. The element loop is multithreaded
over `Threads.nthreads()` with BLAS pinned to one thread (the per-element kernels
are BLAS-bound), mirroring `assemble_stiffness`.
"""
function StiffnessOperator(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T};
                           free::Vector{Int},
                           Krob::Union{Nothing, SparseMatrixCSC{T, Int}} = nothing) where {D, T}
    Ne = mesh.Ne
    nl = tq.nl
    Ke = [Matrix{T}(undef, nl, nl) for _ in 1:Ne]
    nchunks = max(1, min(Ne, Threads.nthreads()))
    bnds = round.(Int, range(0, Ne; length = nchunks + 1))
    blas0 = BLAS.get_num_threads()
    nchunks > 1 && BLAS.set_num_threads(1)
    try
        @sync for t in 1:nchunks
            lo = bnds[t] + 1
            hi = bnds[t + 1]
            lo > hi && continue
            Threads.@spawn begin
                Kscr, tmp, cW = stiffness_scratch(tq)
                @inbounds for e in lo:hi
                    element_stiffness!(Kscr, tmp, cW, mesh, e, tq)
                    Ke[e] .= Kscr
                end
            end
        end
    finally
        nchunks > 1 && BLAS.set_num_threads(blas0)
    end
    u_full = Vector{T}(undef, dof.ndofs)
    y_full = Vector{T}(undef, dof.ndofs)
    ul = Matrix{T}(undef, nl, Ne)
    yl = Matrix{T}(undef, nl, Ne)
    return StiffnessOperator{D, T}(dof, Ke, Krob, free, u_full, y_full, ul, yl)
end

Base.size(op::StiffnessOperator) = (length(op.free), length(op.free))
Base.size(op::StiffnessOperator, d::Integer) = d <= 2 ? length(op.free) : 1
Base.eltype(::StiffnessOperator{D, T}) where {D, T} = T

# Full-space apply `y_full = K·u_full (+ Krob·u_full)`, used both inside `mul!`
# (on the embedded free vector) and for the Dirichlet lift `K·uD` in the RHS, so
# the operator and the RHS see exactly the same discrete operator. The element
# matvecs write disjoint columns of `yl` (embarrassingly parallel, BLAS pinned);
# the cheap O(nl·Ne) `scatter_add!` is left serial (it `+=`s into shared DOFs).
function _apply_full!(op::StiffnessOperator{D, T}, y_full::AbstractVector{T},
                      u_full::AbstractVector{T}) where {D, T}
    dof = op.dof
    Ke = op.Ke
    ul = op.ul
    yl = op.yl
    gather!(ul, u_full, dof)
    Ne = size(ul, 2)
    nchunks = max(1, min(Ne, Threads.nthreads()))
    bnds = round.(Int, range(0, Ne; length = nchunks + 1))
    blas0 = BLAS.get_num_threads()
    nchunks > 1 && BLAS.set_num_threads(1)
    try
        @sync for t in 1:nchunks
            lo = bnds[t] + 1
            hi = bnds[t + 1]
            lo > hi && continue
            Threads.@spawn @inbounds for e in lo:hi
                mul!(view(yl, :, e), Ke[e], view(ul, :, e))
            end
        end
    finally
        nchunks > 1 && BLAS.set_num_threads(blas0)
    end
    scatter_add!(y_full, yl, dof)
    op.Krob === nothing || mul!(y_full, op.Krob, u_full, one(T), one(T))
    return y_full
end

function LinearAlgebra.mul!(y_free::AbstractVector, op::StiffnessOperator{D, T},
                            x_free::AbstractVector) where {D, T}
    u_full = op.u_full
    y_full = op.y_full
    fill!(u_full, zero(T))
    @inbounds for (i, g) in enumerate(op.free)
        u_full[g] = x_free[i]
    end
    _apply_full!(op, y_full, u_full)
    @inbounds for (i, g) in enumerate(op.free)
        y_free[i] = y_full[g]
    end
    return y_free
end

"""
    stiffness_diagonal(op::StiffnessOperator{D, T}) → Vector{T}

The diagonal of the full global operator, `diag(Qᵀ·blockdiag(Kₑ)·Q) (+ diag(Krob))`,
computed matrix-free by scattering the element diagonals. (`dof.multiplicity` is
only the unweighted assembly count, not this metric-weighted diagonal.) Restrict
to `op.free` for the Jacobi preconditioner.
"""
function stiffness_diagonal(op::StiffnessOperator{D, T}) where {D, T}
    dof = op.dof
    nl = dof.nlocal
    Ne = length(op.Ke)
    dl = Matrix{T}(undef, nl, Ne)
    @inbounds for e in 1:Ne
        Kee = op.Ke[e]
        for a in 1:nl
            dl[a, e] = Kee[a, a]
        end
    end
    d = Vector{T}(undef, dof.ndofs)
    scatter_add!(d, dl, dof)
    op.Krob === nothing || (d .+= diag(op.Krob))
    return d
end

# How often `solve_dirichlet_cg` prints a CG progress line when `verbose = true`.
const _CG_PROGRESS_STRIDE = 25

"""
    solve_dirichlet_cg(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T},
                       b::AbstractVector{T}, dirichlet::Dict{Int, T};
                       Krob = nothing, brob = nothing,
                       atol = 0.0, rtol = 1e-10, itmax = 0, verbose = true)
        → (u::Vector{T}, stats)

Matrix-free analog of [`solve_dirichlet`](@ref): solve `(K + Krob) u = b + brob`
subject to `u[d] = dirichlet[d]`, by Jacobi-preconditioned CG on the free-DOF
reduced system `K_FF u_F = (b + brob − (K+Krob)·u_D)_F`, never assembling `K`.

`K` is applied through a [`StiffnessOperator`](@ref); the Jacobi preconditioner is
`Diagonal(1 ./ diag(K)_F)` (the reciprocal, since Krylov applies `M` by `mul!`).
Returns the full solution `u = u_D ⊕ u_F` and the Krylov `stats` (read
`stats.niter`, `stats.solved`, `stats.status`). Note CG's `rtol` is measured in the
M-weighted norm; recompute `‖b − A u‖` directly for a true backward error.

With `verbose = true` (the default) a CG progress line — iteration count and the
current (M-weighted) residual norm — is printed to `stderr` every
`$(_CG_PROGRESS_STRIDE)` iterations, plus a final summary.

Errors if the system is singular (no Dirichlet DOFs and no Robin term: `K` then
has the constant null space) — pin a DOF or add a Robin term.
"""
function solve_dirichlet_cg(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T},
                            b::AbstractVector{T}, dirichlet::Dict{Int, T};
                            Krob::Union{Nothing, SparseMatrixCSC{T, Int}} = nothing,
                            brob::Union{Nothing, AbstractVector{T}} = nothing,
                            atol::Real = 0.0, rtol::Real = 1.0e-10, itmax::Integer = 0,
                            verbose::Bool = true) where {D, T}
    ndofs = dof.ndofs
    (isempty(dirichlet) && Krob === nothing) &&
        throw(ArgumentError("singular system (constant null space): provide Dirichlet DOFs or a Robin term"))
    uD = zeros(T, ndofs)
    isdir = falses(ndofs)
    for (d, v) in dirichlet
        uD[d] = v
        isdir[d] = true
    end
    free = findall(!, isdir)
    op = StiffnessOperator(dof, mesh, tq; free = free, Krob = Krob)

    # Free-DOF RHS: (b + brob − (K+Krob)·uD)[free].
    KuD = Vector{T}(undef, ndofs)
    _apply_full!(op, KuD, uD)
    rhs = collect(b)
    brob === nothing || (rhs .+= brob)
    rhs .-= KuD
    b_free = rhs[free]

    diagK = stiffness_diagonal(op)
    Minv = Diagonal(one(T) ./ @view diagK[free])

    # Progress: `stats.niter` is only set when CG returns, so count iterations in
    # the callback closure; `history = true` makes the per-iteration residual norm
    # available as `stats.residuals[end]`. The callback always returns `false`
    # (never requests an early exit), it only prints.
    if verbose
        printstyled(stderr, "    cg: solving $(length(free)) free DOFs (rtol = $(T(rtol)))\n"; color = :light_black)
        flush(stderr)
    end
    progress = function (workspace)
        rs = workspace.stats.residuals
        k = length(rs)
        if k == 1 || k % _CG_PROGRESS_STRIDE == 0
            r = rs[end]
            printstyled(stderr, "    cg: iter $(lpad(k, 5))   ‖r‖ = $(round(r; sigdigits = 3))\n";
                        color = :light_black)
            flush(stderr)
        end
        return false
    end
    cb = verbose ? progress : (workspace -> false)
    x_free, stats = cg(op, b_free; M = Minv, atol = T(atol), rtol = T(rtol), itmax = Int(itmax),
                       history = verbose, callback = cb)
    if verbose
        printstyled(stderr, "    cg: $(stats.status) in $(stats.niter) iters\n"; color = :light_black)
        flush(stderr)
    end

    u = copy(uD)
    @inbounds for (i, g) in enumerate(free)
        u[g] = x_free[i]
    end
    return u, stats
end
