# M0 — reference element + over-integration. Exactness properties (machine
# precision), run across Float32/Float64/BigFloat to keep the precision-generic
# path honest.

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, default_tol
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

# ∫₀¹ xᵏ dx = 1/(k+1).
_moment(::Type{T}, k::Int) where {T} = inv(T(k + 1))

@testset "reference element (T=$T)" for T in (Float32, Float64, BigFloat)
    # Exactness round-off scales with eps(T); a few thousand eps covers the
    # operator/quadrature condition growth up to the degrees tested.
    tol = 5000 * eps(T)

    @testset "default_tol" begin
        @test default_tol(T) == sqrt(eps(T))
    end

    @testset "GLL degree p=$p" for p in (2, 4, 6)
        refel = ReferenceElement(T, p)
        _progress("ReferenceElement T=$T p=$p")

        @test length(refel.nodes) == p + 1
        @test length(refel.weights) == p + 1
        @test size(refel.D) == (p + 1, p + 1)
        # Endpoints are GLL nodes (needed for C⁰ gluing); nodes lie in [0,1].
        @test abs(refel.nodes[1] - zero(T)) ≤ tol
        @test abs(refel.nodes[end] - one(T)) ≤ tol
        @test all(n -> -tol ≤ n ≤ one(T) + tol, refel.nodes)
        # Weights are positive and sum to the length of [0,1].
        @test all(>(zero(T)), refel.weights)
        @test abs(sum(refel.weights) - one(T)) ≤ tol

        # Differentiation exact for polynomials of degree ≤ p.
        for k in 0:p
            d = refel.D * (refel.nodes .^ k)
            exact = k == 0 ? zeros(T, p + 1) : T(k) .* refel.nodes .^ (k - 1)
            @test maximum(abs.(d .- exact)) ≤ tol
        end

        # GLL quadrature exact for polynomials of degree ≤ 2p−1.
        for k in 0:(2p - 1)
            @test abs(sum(refel.weights .* refel.nodes .^ k) - _moment(T, k)) ≤ tol
        end
        # ... and NOT exact at degree 2p (the mass-lumping under-integration —
        # confirms GLL collocation, not a consistent mass). The lumping error is
        # a precision-independent discretization quantity (~8e-3 at p=2 down to
        # ~1e-7 at p=6); only assert it where it sits safely above round-off. In
        # Float32 the p=6 lumping error ≈ eps(Float32), so this distinction is
        # not resolvable there — skip it.
        if T !== Float32
            @test abs(sum(refel.weights .* refel.nodes .^ (2p)) - _moment(T, 2p)) > 1e-9
        end
    end

    @testset "over-integration p=$p, q=$q" for (p, q) in ((4, 4), (4, 8), (6, 12))
        refel = ReferenceElement(T, p)
        qr = QuadratureRule(refel, q)
        _progress("QuadratureRule T=$T p=$p q=$q")

        @test length(qr.nodes) == q + 1
        @test size(qr.I) == (q + 1, p + 1)
        @test size(qr.Id) == (q + 1, p + 1)
        # Gauss nodes are strictly interior — never sample a singular face.
        @test all(n -> zero(T) < n < one(T), qr.nodes)

        # Gauss quadrature exact for polynomials of degree ≤ 2q+1.
        for k in 0:(2q + 1)
            @test abs(sum(qr.weights .* qr.nodes .^ k) - _moment(T, k)) ≤ tol
        end

        # GLL→Gauss value interpolation exact for degree ≤ p.
        for k in 0:p
            @test maximum(abs.(qr.I * (refel.nodes .^ k) .- qr.nodes .^ k)) ≤ tol
        end
        # GLL→Gauss derivative interpolation exact for degree ≤ p.
        for k in 0:p
            di = qr.Id * (refel.nodes .^ k)
            exact = k == 0 ? zeros(T, q + 1) : T(k) .* qr.nodes .^ (k - 1)
            @test maximum(abs.(di .- exact)) ≤ tol
        end
    end
end
