using MeshTools
include("viz.jl")
# V,F = MeshTools.readply("examples/icosahedron.ply")
V,F = MeshTools.readoff("examples/cat0.off")
mesh = MeshTools.Mesh(V,F)
dist = fast_marching(mesh, [1])
dist = [dist[k] for k=1:mesh.nv]
meshviz(mesh, shift_coordinates=false, color=dist, colormap=:prism)
