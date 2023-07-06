function heat_method(mesh::Mesh, L, i; dt = 1.0)
    # https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/
    nv = mesh.nv
    init = zeros(nv)
    init[i] = 1.0
    heat = heat_implicit(mesh, L, init, dt=dt)
    # println("heat")
    # display(heat)
    ∇ = face_grad(mesh)
    grad_H = reshape(∇ * heat, 3, :)
    grad_H = grad_H ./ sqrt.(sum(grad_H .^ 2, dims=1))
    Δ = div(mesh)
    X = Δ * vec(grad_H)
    dist = L \ X
    dist .-= minimum(dist)
    dist
end
export heat_method