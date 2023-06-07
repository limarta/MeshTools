using Arpack
using LinearAlgebra
using SparseArrays
using LinearAlgebra
using MeshTools
V,F = MeshTools.readoff("examples/cat0.off")
mesh = MeshTools.Mesh(V,F)
L = cot_laplacian(mesh)
A = MeshTools.vertex_area(V,F)
Asp = spdiagm(A)
λ, ϕ =  MeshTools.laplacian_basis(L, Asp, k=20, mode=:arpack)
dot(ϕ[:,1], Asp* ϕ[:,1])
dot(ϕ[:,1], Asp * ϕ[:,2])
norm(L * ϕ[:,1] - λ[1] * Asp * ϕ[:,1]) / (norm(ϕ[:,1]))

using MeshTools
t = 1
init = zeros(mesh.nv)
init[1] = 10.0
heat = MeshTools.heat_spectral(λ, ϕ, Asp, init, 10000)'
heat = MeshTools.heat_implicit(L, A, init, dt=2,steps=1000)'
hks = MeshTools.heatKernelSignature(λ, ϕ, Asp, t)'