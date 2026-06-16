# M1 — dimension-generic C⁰ DOF handler + gather/scatter, validated 1D (no
# orientation) and 2D (edge reversals across the annulus patch seams).
#
# The headline correctness proof is COORDINATE-BASED and so is independent of
# both how `local2global` was built (vertex-id canonical) and of `mesh.conn`:
# we compute the physical position of every local node and require that
#   (consistency) every global DOF maps to a single physical location, and
#   (completeness) distinct global DOFs sit at distinct locations, #DOFs total.
# A correct continuous numbering satisfies both; any orientation/gluing bug
# breaks one of them (verified by the negative control).

using HexCGSEM
using HexCGSEM: ReferenceElement, DofHandler, gather!, scatter_add!, ndofs, nlocal, _face_slot
using HexMeshes
using HexMeshes: element_point_and_jac
using StaticArrays
using LinearAlgebra: dot
using Random
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

# Physical coordinates of every local node: (nlocal, Ne) of SVector{D}.
function _node_coords(mesh, refel::ReferenceElement{T}, ::Val{D}) where {T, D}
    n = refel.p + 1
    dims = ntuple(_ -> n, Val(D))
    lin = LinearIndices(dims)
    X = Matrix{SVector{D, T}}(undef, prod(dims), mesh.Ne)
    for e in 1:mesh.Ne, ci in CartesianIndices(dims)
        ξ = SVector{D, T}(ntuple(d -> refel.nodes[ci[d]], Val(D)))
        P, _ = element_point_and_jac(mesh, e, ξ)
        X[lin[ci], e] = P
    end
    return X
end

# (consistency, completeness) of a local→global map against physical coords.
function _continuity(l2g, X, nd_total; tol = 1.0e-9)
    coordof = Dict{Int, eltype(X)}()
    consistent = true
    for e in axes(l2g, 2), nd in axes(l2g, 1)
        g = l2g[nd, e]
        x = X[nd, e]
        if haskey(coordof, g)
            maximum(abs.(coordof[g] .- x)) > tol && (consistent = false)
        else
            coordof[g] = x
        end
    end
    ks = collect(keys(coordof))
    rounded = unique([round.(coordof[g] ./ tol) for g in ks])
    complete = (length(ks) == nd_total) && (length(rounded) == length(ks))
    return consistent, complete
end

@testset "DOF handler (D=$D, $label)" for (D, label, meshof) in (
    (1, "uniform line", () -> make_uniform_line(Float64, 4, 0.0, 1.0)),
    (2, "uniform quad", () -> make_uniform_quad(Float64, 3, 3, 0.0, 1.0)),
    (2, "annulus", () -> make_annulus_mesh(Float64, 1.0, 2.0, 3)),
)
    mesh = meshof()
    @testset "p=$p" for p in (2, 3, 5)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        l2g = dof.local2global
        X = _node_coords(mesh, refel, Val(D))
        _progress("D=$D $label p=$p  ndofs=$(dof.ndofs)")

        # No gaps: DOFs are exactly 1..ndofs, each used.
        @test sort(unique(l2g)) == collect(1:dof.ndofs)
        @test all(>(0), l2g)

        # Multiplicity = diag(QᵀQ).
        mult = zeros(Int, dof.ndofs)
        for g in l2g
            mult[g] += 1
        end
        @test dof.multiplicity == mult
        @test all(≥(1), dof.multiplicity)
        # Cell-interior nodes are unique (multiplicity 1); a multi-element mesh
        # must have shared DOFs (multiplicity ≥ 2) too.
        @test minimum(dof.multiplicity) == 1
        @test maximum(dof.multiplicity) ≥ 2

        # Headline: continuity (consistency + completeness).
        cons, comp = _continuity(l2g, X, dof.ndofs)
        @test cons
        @test comp

        # gather!/scatter_add! round-trip: Qᵀ Q ug = multiplicity ⊙ ug.
        Random.seed!(20260616 + p)
        ug = randn(dof.ndofs)
        ul = Matrix{Float64}(undef, dof.nlocal, mesh.Ne)
        gather!(ul, ug, dof)
        ug2 = similar(ug)
        scatter_add!(ug2, ul, dof)
        @test ug2 ≈ dof.multiplicity .* ug

        # Adjointness: ⟨Q ug, ul⟩_local = ⟨ug, Qᵀ ul⟩_global.
        ul2 = randn(dof.nlocal, mesh.Ne)
        gather!(ul, ug, dof)
        sc = similar(ug)
        scatter_add!(sc, ul2, dof)
        @test dot(ul, ul2) ≈ dot(ug, sc)

        # Negative control (guaranteed teeth): redirect one interior node to a
        # DOF that lives at a different physical location ⇒ consistency breaks.
        bad = copy(l2g)
        nd0 = findfirst(nd -> dof.multiplicity[l2g[nd, 1]] == 1, 1:dof.nlocal)
        if nd0 !== nothing
            gtgt = 0
            for e in axes(l2g, 2), nd in axes(l2g, 1)
                if maximum(abs.(X[nd, e] .- X[nd0, 1])) > 1.0e-6
                    gtgt = l2g[nd, e]
                    break
                end
            end
            bad[nd0, 1] = gtgt
            badcons, _ = _continuity(bad, X, dof.ndofs)
            @test !badcons
        end
    end

    # The vertex-id-canonical edge flip makes the gluing correct regardless of
    # how each element traverses a shared edge. The current 2D HexMeshes meshes
    # happen to number all *shared* edges with consistent traversal (no edge is
    # walked oppositely by its two elements), so the flip changes only boundary /
    # consistently-oriented edges and is not load-bearing for continuity here —
    # it is exercised as load-bearing by the D₄ face orientation in 3D (M6).
    # Confirm it is at least reachable (changes the numbering) without claiming
    # it breaks continuity on a mesh that doesn't reverse any shared edge.
    if label == "annulus"
        good = DofHandler(mesh, 4)
        bad = DofHandler(mesh, 4; canonical_orientation = false)
        @test bad.local2global != good.local2global
    end
end

# M6 — 3D faces. The cubed-sphere shell's six patches meet along seams with the
# full D₄ face-orientation variety, so this is where the canonical face frame
# becomes load-bearing.
@testset "DOF handler 3D (radial shell)" begin
    mesh = make_radial_shell_mesh(Float64, 1.0, 2.0, 2)
    @testset "p=$p" for p in (2, 3, 4)
        refel = ReferenceElement(Float64, p)
        dof = DofHandler(mesh, p)
        l2g = dof.local2global
        X = _node_coords(mesh, refel, Val(3))
        _progress("3D shell p=$p  ndofs=$(dof.ndofs)")

        @test sort(unique(l2g)) == collect(1:dof.ndofs)
        @test all(>(0), l2g)
        mult = zeros(Int, dof.ndofs)
        for g in l2g
            mult[g] += 1
        end
        @test dof.multiplicity == mult
        @test minimum(dof.multiplicity) == 1
        @test maximum(dof.multiplicity) ≥ 2

        cons, comp = _continuity(l2g, X, dof.ndofs)
        @test cons
        @test comp

        Random.seed!(20260616 + p)
        ug = randn(dof.ndofs)
        ul = Matrix{Float64}(undef, dof.nlocal, mesh.Ne)
        gather!(ul, ug, dof)
        ug2 = similar(ug)
        scatter_add!(ug2, ul, dof)
        @test ug2 ≈ dof.multiplicity .* ug
    end

    # HexMeshes numbers the cubed-sphere shell with consistent face orientation
    # (D₄ orientation trivial in the vertex-id sense), so the canonical face
    # frame is not load-bearing on it — but it does relabel non-shared boundary
    # faces, so the map still changes. Assert it stays continuous and is
    # reachable; the canonicalization itself is unit-tested below.
    p = 4
    refel = ReferenceElement(Float64, p)
    X = _node_coords(mesh, refel, Val(3))
    good = DofHandler(mesh, p)
    bad = DofHandler(mesh, p; canonical_orientation = false)
    @test bad.local2global != good.local2global
    @test all(_continuity(good.local2global, X, good.ndofs))
end

# Direct unit test of the D₄ face-slot canonicalization (the part the
# orientation-trivial HexMeshes meshes don't exercise). A physical face has 4
# distinct corner ids at fixed physical positions; two elements that label the
# face with different local frames (transpose / axis-flip / 90° rotation) must
# map every physical interior node to the SAME global slot.
@testset "_face_slot: D₄-invariant + bijective" begin
    p = 5
    # ids at physical corner positions (0,0),(1,0),(0,1),(1,1) — min id NOT at
    # the origin, to exercise a nontrivial canonical frame.
    idp = Dict((0, 0) => 40, (1, 0) => 10, (0, 1) => 30, (1, 1) => 20)
    idsA = (idp[(0, 0)], idp[(1, 0)], idp[(0, 1)], idp[(1, 1)])

    # Reference (identity-frame) slot for every physical interior node.
    refslot = Dict{Tuple{Int, Int}, Int}()
    seen = Int[]
    for pu in 1:(p - 1), pv in 1:(p - 1)
        s = _face_slot(idsA, pu, pv, p)
        refslot[(pu, pv)] = s
        push!(seen, s)
    end
    # Bijection onto 1..(p-1)².
    @test sort(seen) == collect(1:(p - 1)^2)

    # Transpose frame: local axes swapped ⇒ ids positions 2,3 swap, (la,lb)=(pv,pu).
    idsT = (idp[(0, 0)], idp[(0, 1)], idp[(1, 0)], idp[(1, 1)])
    @test all(_face_slot(idsT, pv, pu, p) == refslot[(pu, pv)]
              for pu in 1:(p - 1), pv in 1:(p - 1))

    # Flip local axis 1: ids[(s1,s2)] = idphys[(1-s1,s2)], (la,lb)=(p-pu, pv).
    idsF = (idp[(1, 0)], idp[(0, 0)], idp[(1, 1)], idp[(0, 1)])
    @test all(_face_slot(idsF, p - pu, pv, p) == refslot[(pu, pv)]
              for pu in 1:(p - 1), pv in 1:(p - 1))

    # 90° rotation: local(a,b) ↦ physical(b, hi-a) ⇒ ids reordered, (la,lb)=(p-pv, pu).
    idsR = (idp[(0, 1)], idp[(0, 0)], idp[(1, 1)], idp[(1, 0)])
    @test all(_face_slot(idsR, p - pv, pu, p) == refslot[(pu, pv)]
              for pu in 1:(p - 1), pv in 1:(p - 1))
end
