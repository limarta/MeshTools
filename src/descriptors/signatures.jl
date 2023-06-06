function heatKernelSignature(λ, ϕ, A, t)
    hks = 0
end
using Arpack
using LinearAlgebra
using SparseArrays
using MeshTools
V,F = MeshTools.readoff("examples/michael1.off")
mesh = MeshTools.Mesh(V,F)
L = cot_laplacian(mesh)
A = MeshTools.vertex_area(V,F)
Asp = spdiagm(A)
λ, ϕ =  MeshTools.laplacian_basis(L, Asp, k=128, mode=:arpack)
dot(ϕ[:,1], Asp* ϕ[:,1])
dot(ϕ[:,1], Asp * ϕ[:,2])
norm(L * ϕ[:,1] - λ[1] * Asp * ϕ[:,1]) / (norm(ϕ[:,1]))

# TODO: Area-agnostic hks

# function heat_spectral(λ::Vector, ϕ::Matrix, init, t; A=nothing)
#     """
#     init - |V| or |V|×|C|
#     note that init may be a single vector or length(t) vectors. If it is a single vector, then heat is diffused for each time t. If it is
#     multiple vectors, then vector init[i] is diffused for time t[i]
#     c = ϕ'*(A .* init) .* exp.(-λ * t')
    
#     If A is defined, then inner products are treated as A-inner products
#     """
#     if A === nothing
#         c = ϕ'*(init) .* exp.(-λ * t')
#         heat = abs.(ϕ * c)
#     else
#         # Area aware
#     end
#     heat
# end

# function hks = heatKernelSignature( laplaceBasis, eigenvalues, Ae, numTimes )
#     %HEATKERNELSIGNATURE Summary of this function goes here
#     %   Detailed explanation goes here
#     numEigenfunctions = size(eigenvalues,1);
    
#     D = laplaceBasis' * (Ae * laplaceBasis.^2);
    
#     absoluteEigenvalues = abs(eigenvalues);
#     emin = absoluteEigenvalues(2);
#     emax = absoluteEigenvalues(end);
    
#     t = linspace(emin,emax,numTimes);
    
#     T = exp(-abs(eigenvalues*t));
    
#     hks = D*T;
#     hks = laplaceBasis*hks;
    
#     end
    

function waveKernelSignature()
end