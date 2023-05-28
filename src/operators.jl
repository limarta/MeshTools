function cot_laplacian(V,F)
    nv = size(V)[2]
    nf = size(F)[2]
    #For all 3 shifts of the roles of triangle vertices
    #to compute different cotangent weights
    cots = zeros(nf, 3)
    for perm in [(1,2,3), (2,3,1), (3,1,2)]
        i, j, k = perm
        u = V[:,F[i,:]] - V[:, F[k,:]]
        v = V[:, F[j,:]] - V[:, F[k,:]]
        cotAlpha = vec(abs.(vdot(u,v; dims=1))) ./ norm.(eachcol(multicross(u,v)))
        cots[:,i] = cotAlpha

    end
    I = F[1,:]; J = F[2,:]; K = F[3,:];

    L = sparse([I;J;K], [J;K;I], [cots[:,1];cots[:,2];cots[:,3]], nv, nv)
    L = L + L'
    rowsums = vec(sum(L,dims=2))
    L = spdiagm(0 => rowsums) - L
    return 0.5 * L
end

function FtoV(V,F, face_area)
    A = face_area
    nf = size(F)[2]
    nv = size(V)[2]
    G = sparse(vec(F), vec(repeat(1:nf, 1, 3)'), vec(repeat(A, 1, 3)'), nv, nf)
    G = (G' ./ A)'
end

function get_operators(mesh::Mesh; k=200)
    λ, ϕ = get_spectrum(mesh, k=k)
    grad_x, grad_y = mesh.∇_x, mesh.∇_y
    grad_x = convert.(Float32, grad_x)
    grad_y = convert.(Float32, grad_y)
    mesh.cot_laplacian, convert.(Float32, mesh.vertex_area), λ, ϕ, grad_x, grad_y
end

ops = (:cot_laplacian,)
# TODO: Figure out how to automate with macros :)
for op in ops
    eval(:($op(mesh::Mesh) = $op(mesh.V, mesh.F)))
end
# cot_laplacian(mesh::Mesh) = cot_laplacian(mesh.V, mesh.F)
# FtoV(mesh::Mesh) = FtoV(mesh.V, mesh.F, face_area(mesh))
# vertex_grad(mesh::Mesh) = vertex_grad(mesh.V, mesh.F, mesh.vertex_normals)
# tangent_basis(mesh::Mesh) =  tangent_basis(mesh.V, mesh.F, mesh.vertex_normals)