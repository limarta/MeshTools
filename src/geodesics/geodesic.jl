function heat_method(mesh::Mesh, L, i)
    # https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/
    nv = mesh.nv
    init = zeros(nv)
    init[i] = 1.0
    heat = heat_implicit(mesh, L, init)
    ∇ = face_grad(mesh)
    grad_H = reshape(∇ * heat, 3, :)'
    ftov = FtoV(mesh.V, mesh.F, face_area(mesh.V, mesh.F))
    grad_H =  (ftov * grad_H)'
    grad_H = grad_H ./ sqrt.(sum(grad_H .^ 2, dims=1))
    Δ = div(mesh)
    println(size(Δ))
    println(size(vec(grad_H)))
    X = Δ * vec(grad_H)
    println(size(X))
    println(size(L))
    L \ X
end
export heat_method