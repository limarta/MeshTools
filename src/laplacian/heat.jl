"""
Heat diffusion is governed by Δu = u_t
There are two common ways of solving heat diffusion on the surface of a mesh.
Spectral mode uses the eigenfunctions of Δ to compute the heat directly by the formula
    h(t) = ∑<f,ϕ>*ϕ*exp(-λ*t)
Implicit mode on the other hand uses backward Euler steps by solving
    (I+dt Δ) h_{t+1} = h_t

Depending on the use case, one mode may be better than the other. Implicit mode tends to be
easier to solve for a small number of timesteps. However, since each time step requires
solving a system of equations, this can quickly slow down for scenarios where diffusion
is done many times.

Spectral mode requires solving for eigenvectors of the Laplacian which can be expensive. Luckily,
we can approximate diffusion fairly accurately using only first K eigenvectors. K tends to be fairly
small even for sizable meshes (roughly 50 to 500), and there are many efficient eigensolvers for this.
Another benefit of spectral mode is that once the eigenvectors have been computed, these can be cached
for later computation.
"""

function heat_implicit(L, A, signal; dt=0.001, steps=1)
    M = spdiagm(A)
    D = cholesky(M+dt*L)
    heat = signal
    for t=1:steps
        heat = D \ (heat .* A)
    end
    return heat
end


function heat_spectral(λ::Vector, ϕ::Matrix, init, t; A=nothing)
    """
    init - |V| or |V|×|C|
    note that init may be a single vector or length(t) vectors. If it is a single vector, then heat is diffused for each time t. If it is
    multiple vectors, then vector init[i] is diffused for time t[i]
    c = ϕ'*(A .* init) .* exp.(-λ * t')
    
    If A is defined, then inner products are treated as A-inner products
    """
    if A === nothing
        c = ϕ'*(init) .* exp.(-λ * t')
        heat = abs.(ϕ * c)
    else
        # Area aware
    end
    heat
end

# function heat(L, A, init, t; k=200)
#     # λ, ϕ = eigs(L ./ A, nev=k, sigma=1e-8)
#     λ, ϕ = eigs(L , nev=k, sigma=1e-8)
#     heat(λ, ϕ, A, init, t)
# end


# Convenience Functions
# heat_implicit(mesh::Mesh, signal; dt=0.001, steps=100) = heat_implicit(mesh.cot_laplacian, mesh.vertex_area, signal, dt=dt, steps=steps) 
# heat(mesh::Mesh, init, t, k=200) = heat(mesh.cot_laplacian, mesh.vertex_area, init, t, k=k)

# function decompose_feature_by_spectrum(mesh::Mesh, λ, ϕ, f)
#     c = vec(ϕ'*(mesh.vertex_area .* f))
#     real.(ϕ .* c')
# end

# function hks(λ, ϕ, A, t)
#     n = size(ϕ)[1]
#     init = I(n)
#     h = heat_diffusion(λ, ϕ, A, init, t)
#     diag(h)
# end

# hks(λ, ϕ, A, n::Int) = hcat([hks(λ, ϕ, A, 1.5f0^t) for t=-n:0]...) 
# function get_spectrum(mesh::Mesh; k=200)
#     # L = mesh.cot_laplacian ./ mesh.vertex_area
#     L = mesh.cot_laplacian
#     λ, ϕ = eigs(L, nev=k, sigma=1e-8)
#     λ, ϕ =  convert.(Float32, real.(λ)), convert.(Float32, real.(ϕ))
# end

export heat_implicit, heat_spectral