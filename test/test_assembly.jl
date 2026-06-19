# Global assembly: the multithreaded `assemble_stiffness` / `assemble_mass` must
# reproduce, to round-off, an independent single-threaded reference built from
# the unchanged `element_matrices` (which computes Ke and Me together). This
# guards the stiffness-/mass-only kernels, the per-element offset bookkeeping,
# and thread-count independence — run the suite with `-t N` to exercise the
# parallel path.

using HexCGSEM
using HexCGSEM: ReferenceElement, QuadratureRule, TensorQuadrature, DofHandler,
                assemble_stiffness, assemble_mass, element_matrices,
                element_stiffness!, element_mass!, stiffness_scratch, mass_scratch
using HexMeshes
using SparseArrays: sparse
using LinearAlgebra: norm
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr)))

# Single-threaded reference assembly using the original both-matrices kernel.
function _ref_assemble(dof, mesh, tq::TensorQuadrature{D, T}, which::Symbol) where {D, T}
    nl = tq.nl
    cap = nl * nl * mesh.Ne
    Ii = Vector{Int}(undef, cap)
    Jj = Vector{Int}(undef, cap)
    Vv = Vector{T}(undef, cap)
    l2g = dof.local2global
    c = 0
    for e in 1:mesh.Ne
        Ke, Me = element_matrices(mesh, e, tq)
        A = which === :K ? Ke : Me
        for b in 1:nl
            gb = l2g[b, e]
            for a in 1:nl
                c += 1
                Ii[c] = l2g[a, e]
                Jj[c] = gb
                Vv[c] = A[a, b]
            end
        end
    end
    return sparse(Ii, Jj, Vv, dof.ndofs, dof.ndofs)
end

@testset "assembly threading vs serial reference ($(Threads.nthreads()) threads)" begin
    @testset "$label (D=$D)" for (D, label, meshof) in (
        (2, "annulus", () -> make_annulus_mesh(Float64, 1.0, 2.0, 4)),
        (3, "two_ball separated", () -> make_two_ball_mesh(Float64, 1.0, 100.0, 10.0, 2;
                                                           M_h = 2, M_b = 2, M_i = 2, M_s = 2,
                                                           outer_bc = :dirichlet, mode = :separated)),
    )
        mesh = meshof()
        @testset "p=$p" for p in (3, 4)
            refel = ReferenceElement(Float64, p)
            dof = DofHandler(mesh, p)
            tq = TensorQuadrature(refel, QuadratureRule(refel, 2p), Val(D))
            _progress("D=$D $label p=$p  (Ne=$(mesh.Ne), ndofs=$(dof.ndofs))")

            # Per-element kernels match the reference element_matrices exactly.
            Kref_e, Mref_e = element_matrices(mesh, 1, tq)
            Ks, ts, cWs = stiffness_scratch(tq)
            Ms, tm, md = mass_scratch(tq)
            element_stiffness!(Ks, ts, cWs, mesh, 1, tq)
            element_mass!(Ms, tm, md, mesh, 1, tq)
            # Round-off only: the gemm rounding can differ from element_matrices
            # by a ulp when BLAS uses a different number of threads.
            @test Ks ≈ Kref_e rtol = 1.0e-10
            @test Ms ≈ Mref_e rtol = 1.0e-10

            # Global assembly reproduces the serial reference.
            K = assemble_stiffness(dof, mesh, tq)
            M = assemble_mass(dof, mesh, tq)
            Kref = _ref_assemble(dof, mesh, tq, :K)
            Mref = _ref_assemble(dof, mesh, tq, :M)
            @test K ≈ Kref rtol = 1.0e-10
            @test M ≈ Mref rtol = 1.0e-10
        end
    end
end
