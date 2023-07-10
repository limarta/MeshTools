module MeshTools
using PlyIO
using LinearAlgebra
using Arpack
using SparseArrays
using Rotations
using DataStructures

vdot(x,y; dims=1) = sum(x .* y, dims=dims)
multicross(x,y) = reduce(hcat, cross.(eachcol(x), eachcol(y)))

include("mesh.jl")
include("geom.jl")
include("operators/operators.jl")
include("descriptors/descriptors.jl")
include("geodesics/geodesic.jl")
include("transformations/arap.jl")
include("read.jl")

export Mesh, cot_laplacian

end # module MeshTools
