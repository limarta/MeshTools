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

function heat_implicit(L, A, init; dt=0.001, steps=1)
    M = spdiagm(A)
    D = cholesky(M+dt*L)
    heat = init
    for t=1:steps
        heat = D \ (M*heat)
    end
    return heat
end

function heat_implicit(mesh::Mesh, L, init; dt=0.001, steps=1)
    heat_implicit(L, vertex_area(mesh), init, dt=dt, steps=steps)
    # D = cholesky(M+dt*L)
    # heat = init
    # for t=1:steps
        # heat = D \ (M*heat)
    # end
    # return heat
end

function heat_spectral(λ::Vector, ϕ::Matrix, init, t)
    """
    init - |V| or |V|×|C|
    note that init may be a single vector or length(t) vectors. If it is a single vector, then heat is diffused for each time t. If it is
    multiple vectors, then vector init[i] is diffused for time t[i]
    c = ϕ'*(A .* init) .* exp.(-λ * t')
    
    If A is defined, then inner products are treated as A-inner products
    """
    c = ϕ'*(init) .* exp.(-λ * t')
    heat = abs.(ϕ * c)
end

function heat_spectral(λ, ϕ, A, init, t)
    c = ϕ' * (A*init) .* exp.(-λ * t')
    ϕ * c
end
