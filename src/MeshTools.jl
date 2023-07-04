module MeshTools
using PlyIO
using LinearAlgebra
using Arpack
using SparseArrays

vdot(x,y; dims=1) = sum(x .* y, dims=dims)
multicross(x,y) = reduce(hcat, cross.(eachcol(x), eachcol(y)))

include("mesh.jl")
include("geom.jl")
include("vectorfield/vectorfield.jl")
include("laplacian/laplacian.jl")
include("descriptors/descriptors.jl")
include("operators.jl")
include("geodesics/geodesic.jl")
include("read.jl")

export Mesh, cot_laplacian

end # module MeshTools
