# HexCGSEM.jl

Continuous-Galerkin spectral-element (CG-SEM) operators on the conforming
cubed-sphere meshes of [HexMeshes.jl](https://github.com/eschnett/HexMeshes.jl),
for solving linear elliptic (Laplace) problems to high accuracy.

CG-SEM is chosen over the SBP-SAT/DG operators of HexSBPSAT for *elliptic*
solves because it is better-conditioned (symmetric positive-definite, no penalty
parameter), it admits consistent (over-)integration that removes the curvilinear
under-integration aliasing floor, and its variational form makes radial
compactification to spatial infinity clean (the Dirichlet energy stays finite at
i⁰ while the strong-form operator degenerates).

The operators are dimension-generic in `D ∈ {1, 2, 3}`: every type and function
is a tensor product of the 1D reference element, with an entity-by-dimension
global DOF numbering (vertices, edges, faces, interiors) built on the mesh's
globally-deduplicated vertex ids.

Status: under construction (see the milestone plan). Currently implemented:

* **Reference element** — `ReferenceElement` (Gauss–Lobatto–Legendre nodal
  basis) and `QuadratureRule` (Gauss over-integration) on the reference interval
  `[0, 1]`, plus the GLL→Gauss interpolation matrices used to de-alias
  curvilinear element operators.
