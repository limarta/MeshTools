using MeshTools
using LinearAlgebra
using Rotations
include("viz.jl")
# V,F = MeshTools.readply("examples/icosahedron.ply")
V,F = MeshTools.readoff("examples/cat0.off")
mesh = MeshTools.Mesh(V,F)
L = cot_laplacian(mesh)
source = 1
dist = heat_method(mesh, L, source, dt=0.50)
# ax3, msh3 = mesh(fig[1, 3], X, T, color=dist[:], shading=false, colormap=:heat)
# meshviz(mesh, shift_coordinates=false, color=dist, colormap=:prism)
# viz_point!(mesh, source, shift_coordinates=false, color=:pink, markersize=100)

# constraints = Dict{Int, Vector{Float32}}(1=>[-0.2387692, 1.31038, 0.13],
#                 2=>[-.2758292, 1.258256, 0.12],
#                 3=>[-.267489, 1.34744, 0.159128],
#                 4=>[-.312866, 1.22271, 0.146236])
rot = rand(RotMatrix3{Float64})
constraints = Dict{Int, Vector{Float64}}(k=>rot*V[:,k] for k=1:10:mesh.nv)
# constraints[32] = V[:,32].+3
# constraints = Dict{Int, Vector{Float64}}(k=>V[:,k] for k=1:1)

new_mesh = arap_minimization(mesh, constraints, L);
# println("Norm ", norm(mesh.V - new_mesh.V))
# ax3, msh3 = mesh(fig[1, 3], X, T, color=dist[:], shading=false, colormap=:heat)
meshviz(mesh, shift_coordinates=false, color=dist, colormap=:prism)
meshviz(new_mesh, shift_coordinates=false, color=dist, colormap=:heat)
# viz_point!(mesh, source, shift_coordinates=false, color=:pink, markersize=100)