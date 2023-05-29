function get_operators(mesh::Mesh; k=200)
    λ, ϕ = get_spectrum(mesh, k=k)
    grad_x, grad_y = mesh.∇_x, mesh.∇_y
    grad_x = convert.(Float32, grad_x)
    grad_y = convert.(Float32, grad_y)
    mesh.cot_laplacian, convert.(Float32, mesh.vertex_area), λ, ϕ, grad_x, grad_y
end