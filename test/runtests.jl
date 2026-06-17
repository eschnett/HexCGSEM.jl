using Test
using HexCGSEM

function _section(label)
    printstyled(stderr, "── ", label, " ──\n"; color = :cyan, bold = true)
    return flush(stderr)
end

_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan); flush(stderr))

@testset verbose = true "HexCGSEM" begin
    _section("test_reference_element.jl")
    include("test_reference_element.jl")
    _section("test_dofmap.jl")
    include("test_dofmap.jl")
    _section("test_element_ops.jl")
    include("test_element_ops.jl")
    _section("test_solve.jl")
    include("test_solve.jl")
    _section("test_solve_robin.jl")
    include("test_solve_robin.jl")
    _section("test_solve_shell_3d.jl")
    include("test_solve_shell_3d.jl")
    _section("test_solve_compact_3d.jl")
    include("test_solve_compact_3d.jl")
    _section("test_solve_two_hole.jl")
    include("test_solve_two_hole.jl")
end
