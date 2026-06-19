"""
    HexCGSEM

Continuous-Galerkin spectral-element (CG-SEM) operators on the conforming
cubed-sphere meshes of `HexMeshes`, for linear elliptic (Laplace) problems at
high accuracy. Dimension-generic in `D ∈ {1, 2, 3}`: the multi-dimensional
operators are tensor products of a 1D reference element, with an
entity-by-dimension global DOF numbering built on the mesh's deduplicated vertex
ids.

# Layers (built incrementally per the milestone plan)

* `reference_element.jl` — `ReferenceElement` (Gauss–Lobatto–Legendre nodal
                           basis) and `QuadratureRule` (Gauss over-integration)
                           on the reference interval `[0, 1]`, plus the GLL→Gauss
                           value/derivative interpolation matrices used to
                           de-alias curvilinear element operators.
* `dofmap.jl`            — `DofHandler{D}`, the global C⁰ DOF numbering
                           (entity-by-dimension, keyed on the mesh's deduplicated
                           vertex ids), and the `gather!` / `scatter_add!`
                           operations between the local and global layouts.
* `element_ops.jl`       — `TensorQuadrature` (tensor-product GLL→Gauss tables)
                           and `element_matrices` (over-integrated local
                           stiffness `Ke` and mass `Me`).
* `geometry.jl`          — physical node / DOF coordinates (`node_coords`,
                           `dof_coords`).
* `assembly.jl`          — global `assemble_stiffness` / `assemble_mass` /
                           `assemble_load` via the DOF map.
* `boundary.jl`          — `boundary_dofs` (tagged boundary DOFs) and the Robin
                           surface-integral assembly.
* `solve.jl`             — `solve_dirichlet`, the reduced-system sparse Cholesky
                           solve, and `solve_dirichlet_cg`, the matrix-free
                           Jacobi-preconditioned CG solve (`StiffnessOperator`)
                           that never assembles the global `K`.
"""
module HexCGSEM

include("reference_element.jl")
include("dofmap.jl")
include("element_ops.jl")
include("geometry.jl")
include("assembly.jl")
include("boundary.jl")
include("solve.jl")

export ReferenceElement, QuadratureRule, default_tol
export DofHandler, gather!, scatter_add!, ndofs, nlocal
export TensorQuadrature, element_matrices
export node_coords, dof_coords
export assemble_stiffness, assemble_mass, assemble_load
export boundary_dofs, assemble_robin, solve_dirichlet
export StiffnessOperator, stiffness_diagonal, solve_dirichlet_cg

end # module HexCGSEM
