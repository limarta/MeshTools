using MeshTools
using ArnoldiMethod
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using LinearMaps

# TODO: Check umfpack
V,F = MeshTools.readoff("examples/michael1.off")
mesh = MeshTools.Mesh(V,F)
L = cot_laplacian(mesh)
A = MeshTools.vertex_area(V,F)
λ, ϕ = eigs(L, nev=128, sigma=0)


# Define a matrix whose eigenvalues you want
# Factorizes A and builds a linear map that applies inv(A) to a vector.
function construct_linear_map(A)
    F = lu(A)
    LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x) , size(A,1), ismutating=true)
end

# Target the largest eigenvalues of the inverted problem
function eig(L)
    decomp, = partialschur(construct_linear_map(L), nev=128, tol=1e-10, restarts=100, which=LM())
    λs_inv, X = partialeigen(decomp)
end
λs_inv, X =  eig(L)

# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.
λs = 1 ./ λs_inv
println(λs)
println()
println(λ)
 
# Show that Ax = xλ
# @show norm(A * X - X * Diagonal(λs)) # 7.38473677258669e-6