using MeshTools
using LinearAlgebra
using ProfileView
# V,F = MeshTools.readply("examples/icosahedron.ply")
V,F = MeshTools.readoff("examples/gourd.off")
mesh = MeshTools.Mesh(V,F)
L = cot_laplacian(mesh)
# heat_method(mesh, L, 1)
Div = MeshTools.div(mesh);
grad = MeshTools.face_grad(mesh)
# println(norm(D * grad + L)/norm(L))