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

# Run `f(lo, hi)` over contiguous chunks of `1:n`, one Julia task per chunk, with
# BLAS pinned to one thread for the duration (the per-element kernels are
# BLAS-bound, so Julia-threads × BLAS-threads would oversubscribe). Mirrors the
# threading harness in `_assemble_galerkin`.
function _threaded_chunks(f::F, n::Integer) where {F}
    nchunks = max(1, min(n, Threads.nthreads()))
    bnds = round.(Int, range(0, n; length = nchunks + 1))
    blas0 = BLAS.get_num_threads()
    nchunks > 1 && BLAS.set_num_threads(1)
    try
        @sync for t in 1:nchunks
            lo = bnds[t] + 1
            hi = bnds[t + 1]
            lo > hi && continue
            Threads.@spawn f(lo, hi)
        end
    finally
        nchunks > 1 && BLAS.set_num_threads(blas0)
    end
end

# Per-element operator payloads (the two backends), dispatched by `_apply_element!`
# and `_element_diag!`. `:dense` stores the explicit `nl×nl` element matrices;
# `:sumfac` stores only the per-element metric and applies the operator by
# sum-factorization (see element_ops.jl), trading O(nl²) storage for O(nq).

struct DenseElementOp{T}
    Ke :: Vector{Matrix{T}}
end

# Holds only the per-element metric `G` and the small 1D operators — NOT the dense
# tensor-product `B`/`V` tables of the TensorQuadrature (which sum-factorization
# never uses), so the stored footprint is `O(nq·Ne)`, not `O(nl²·Ne)`.
struct SumFacElementOp{D, T}
    G    :: Array{T, 3}                  # (nq, nsym, Ne) symmetric metric cW
    I1   :: Matrix{T}                    # 1D value interp (nq1×nl1)
    Id1  :: Matrix{T}                    # 1D derivative interp
    I1t  :: Matrix{T}                    # transpose(I1)
    Id1t :: Matrix{T}                    # transpose(Id1)
    Pt   :: NTuple{3, Matrix{T}}         # transposed diagonal factors (PIIt, PIIdt, PIdIdt)
    sidx :: Matrix{Int}                  # (i,j) → packed symmetric component
    nl   :: Int
    nq   :: Int
end

function _build_dense(mesh::Mesh{D}, tq::TensorQuadrature{D, T}) where {D, T}
    Ne = mesh.Ne
    nl = tq.nl
    Ke = [Matrix{T}(undef, nl, nl) for _ in 1:Ne]
    _threaded_chunks(Ne) do lo, hi
        Kscr, tmp, cW = stiffness_scratch(tq)
        @inbounds for e in lo:hi
            element_stiffness!(Kscr, tmp, cW, mesh, e, tq)
            Ke[e] .= Kscr
        end
    end
    return DenseElementOp{T}(Ke)
end

function _build_sumfac(mesh::Mesh{D}, tq::TensorQuadrature{D, T}) where {D, T}
    Ne = mesh.Ne
    nq = tq.nq
    sidx = _symidx(Val(D))
    G = Array{T, 3}(undef, nq, _nsym(D), Ne)
    _threaded_chunks(Ne) do lo, hi
        cW = Matrix{Vector{T}}(undef, D, D)
        for i in 1:D, j in 1:D
            cW[i, j] = Vector{T}(undef, nq)
        end
        @inbounds for e in lo:hi
            element_metric!(cW, mesh, e, tq)
            for j in 1:D, i in 1:j
                copyto!(view(G, :, sidx[i, j], e), cW[i, j])
            end
        end
    end
    Pt = (Matrix(transpose(tq.I1 .* tq.I1)),
          Matrix(transpose(tq.I1 .* tq.Id1)),
          Matrix(transpose(tq.Id1 .* tq.Id1)))
    return SumFacElementOp{D, T}(G, copy(tq.I1), copy(tq.Id1), Matrix(transpose(tq.I1)),
                                 Matrix(transpose(tq.Id1)), Pt, sidx, tq.nl, tq.nq)
end

"""
    StiffnessOperator{D, T, E}

Matrix-free representation of the global stiffness operator restricted to a set
of free DOFs, for use as the `A` in a Krylov solve. Applying it never forms the
global `K`: it gathers the free-DOF vector into the per-element layout, applies
the per-element operator `E` (`Qᵀ·blockdiag(Kₑ)·Q`), and scatters back, adding the
(small, boundary-only) Robin matrix `Krob` when present.

The per-element backend `E` is either `DenseElementOp` (explicit `nl×nl` matrices)
or `SumFacElementOp` (sum-factorization; only the metric is stored). The rest of
the operator — `mul!`, gather/scatter, free-DOF embedding — is identical.

# Fields

* `dof :: DofHandler{D}` — the gather/scatter map.
* `eop :: E` — the per-element backend.
* `Krob :: Union{Nothing, SparseMatrixCSC{T, Int}}` — optional Robin contribution.
* `free :: Vector{Int}` — global indices of the free (non-Dirichlet) DOFs; the
  operator acts on vectors of length `length(free)`.
* `u_full, y_full :: Vector{T}` — `(ndofs,)` full-space scratch.
* `ul, yl :: Matrix{T}` — `(nl, Ne)` per-element scratch.

Construct with [`StiffnessOperator(dof, mesh, tq; free, Krob, backend)`](@ref).
"""
struct StiffnessOperator{D, T, E}
    dof    :: DofHandler{D}
    eop    :: E
    Krob   :: Union{Nothing, SparseMatrixCSC{T, Int}}
    free   :: Vector{Int}
    u_full :: Vector{T}
    y_full :: Vector{T}
    ul     :: Matrix{T}
    yl     :: Matrix{T}
end

"""
    StiffnessOperator(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T};
                      free::Vector{Int}, Krob = nothing, backend = :dense)
        → StiffnessOperator

Precompute the per-element backend and scratch buffers for a matrix-free operator
on the free DOFs `free`. `backend = :dense` stores the explicit element matrices
(faster gemv at low `p`); `backend = :sumfac` stores only the metric and applies
the operator by sum-factorization (`O(nq)` instead of `O(nl²)` storage — the
choice for memory-bound high-order 3D). The element loop is multithreaded over
`Threads.nthreads()` with BLAS pinned to one thread, mirroring `assemble_stiffness`.
"""
function StiffnessOperator(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T};
                           free::Vector{Int},
                           Krob::Union{Nothing, SparseMatrixCSC{T, Int}} = nothing,
                           backend::Symbol = :dense) where {D, T}
    eop = if backend === :sumfac
        _build_sumfac(mesh, tq)
    elseif backend === :dense
        _build_dense(mesh, tq)
    else
        throw(ArgumentError("backend must be :dense or :sumfac, got :$backend"))
    end
    nl = tq.nl
    Ne = mesh.Ne
    u_full = Vector{T}(undef, dof.ndofs)
    y_full = Vector{T}(undef, dof.ndofs)
    ul = Matrix{T}(undef, nl, Ne)
    yl = Matrix{T}(undef, nl, Ne)
    return StiffnessOperator{D, T, typeof(eop)}(dof, eop, Krob, free, u_full, y_full, ul, yl)
end

# Per-element apply `ycol = Kₑ · xcol`, dispatched on the backend. `sc` is the
# per-chunk scratch (`nothing` for dense, a `SumFacScratch` for sumfac).
@inline _apply_element!(eop::DenseElementOp, ycol, xcol, e, sc) = mul!(ycol, eop.Ke[e], xcol)
@inline function _apply_element!(eop::SumFacElementOp{D, T}, ycol, xcol, e, sc) where {D, T}
    return sumfac_apply!(ycol, xcol, eop.G, e, eop.I1, eop.Id1, eop.I1t, eop.Id1t, eop.sidx, sc)
end

_chunk_scratch(::DenseElementOp) = nothing
_chunk_scratch(eop::SumFacElementOp{D}) where {D} = sumfac_scratch(eop.I1, eop.nl, eop.nq, Val(D))

Base.size(op::StiffnessOperator) = (length(op.free), length(op.free))
Base.size(op::StiffnessOperator, d::Integer) = d <= 2 ? length(op.free) : 1
Base.eltype(::StiffnessOperator{D, T, E}) where {D, T, E} = T

# Full-space apply `y_full = K·u_full (+ Krob·u_full)`, used both inside `mul!`
# (on the embedded free vector) and for the Dirichlet lift `K·uD` in the RHS, so
# the operator and the RHS see exactly the same discrete operator. The element
# matvecs write disjoint columns of `yl` (embarrassingly parallel, BLAS pinned);
# the cheap O(nl·Ne) `scatter_add!` is left serial (it `+=`s into shared DOFs).
function _apply_full!(op::StiffnessOperator{D, T}, y_full::AbstractVector{T},
                      u_full::AbstractVector{T}) where {D, T}
    dof = op.dof
    eop = op.eop
    ul = op.ul
    yl = op.yl
    gather!(ul, u_full, dof)
    _threaded_chunks(size(ul, 2)) do lo, hi
        sc = _chunk_scratch(eop)
        @inbounds for e in lo:hi
            _apply_element!(eop, view(yl, :, e), view(ul, :, e), e, sc)
        end
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
# Fill the per-element diagonals `dl[a, e] = Kₑ[a, a]`, dispatched on the backend.
function _element_diag!(dl, eop::DenseElementOp)
    nl = size(dl, 1)
    @inbounds for e in axes(dl, 2)
        Kee = eop.Ke[e]
        for a in 1:nl
            dl[a, e] = Kee[a, a]
        end
    end
    return dl
end
function _element_diag!(dl, eop::SumFacElementOp{D, T}) where {D, T}
    _threaded_chunks(size(dl, 2)) do lo, hi
        sc = sumfac_scratch(eop.I1, eop.nl, eop.nq, Val(D))
        @inbounds for e in lo:hi
            sumfac_diag_element!(view(dl, :, e), eop.G, e, eop.Pt, eop.sidx, sc)
        end
    end
    return dl
end

function stiffness_diagonal(op::StiffnessOperator{D, T}) where {D, T}
    dof = op.dof
    dl = Matrix{T}(undef, dof.nlocal, size(op.ul, 2))
    _element_diag!(dl, op.eop)
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
`$(_CG_PROGRESS_STRIDE)` iterations, plus a final summary. `backend` selects the
[`StiffnessOperator`](@ref) element backend (`:dense` or `:sumfac`); the result is
identical to round-off.

Errors if the system is singular (no Dirichlet DOFs and no Robin term: `K` then
has the constant null space) — pin a DOF or add a Robin term.
"""
function solve_dirichlet_cg(dof::DofHandler{D}, mesh::Mesh{D}, tq::TensorQuadrature{D, T},
                            b::AbstractVector{T}, dirichlet::Dict{Int, T};
                            Krob::Union{Nothing, SparseMatrixCSC{T, Int}} = nothing,
                            brob::Union{Nothing, AbstractVector{T}} = nothing,
                            atol::Real = 0.0, rtol::Real = 1.0e-10, itmax::Integer = 0,
                            verbose::Bool = true, backend::Symbol = :dense) where {D, T}
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
    op = StiffnessOperator(dof, mesh, tq; free = free, Krob = Krob, backend = backend)

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
