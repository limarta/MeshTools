using MeshTools
using LinearAlgebra
include("viz.jl")
# V,F = MeshTools.readply("examples/icosahedron.ply")
V,F = MeshTools.readoff("examples/spherers.off")
mesh = MeshTools.Mesh(V,F)
L = cot_laplacian(mesh)
# heat_method(mesh, L, 1)
# dist = heat_method(mesh, L, 100, dt=0.00010)
# meshviz(mesh, shift_coordinates=true, color=dist)
D = div(mesh, L)