using Arpack
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using LinearAlgebra
using LinearMaps
using MeshTools
V,F = MeshTools.readoff("examples/cat0.off")
mesh = MeshTools.Mesh(V,F)
L = cot_laplacian(mesh)
A = MeshTools.vertex_area(V,F)
Asp = spdiagm(A)

@time λ, ϕ =  MeshTools.laplacian_basis(L, Asp, k=20, mode=:arpack)

struct ShiftAndInvert{TA,TB,TT}
    A_lu::TA
    B::TB
    temp::TT
end

function (M::ShiftAndInvert)(y,x)
    mul!(M.temp, M.B, x)
    ldiv!(y, M.A_lu, M.temp)
end
function construct_linear_map(A,B)
    a = ShiftAndInvert(lu(A),B,Vector{eltype(A)}(undef, size(A,1)))
    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)
end

M = construct_linear_map(L, Asp);
@time IterativeSolvers.lobpcg(L, Asp, true, 20)
# # Target the largest eigenvalues of the inverted problem
# function eig(L)
#     decomp, = partialschur(construct_linear_map(L), nev=128, tol=1e-10, restarts=100, which=LM())
#     λs_inv, X = partialeigen(decomp)
# end
# λs_inv, X =  eig(L)