using MeshTools
using Arpack
V,F = MeshTools.readoff("examples/cat0.off")
mesh = MeshTools.Mesh(V,F)
L = unweighted_laplacian(mesh)
k = 10
eigs(L, nev=k, maxiter = 200, sigma=1e-6)