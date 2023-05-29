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

# TODO: Figure out how to automate with macros :)
for op in (:cot_laplacian,)
    eval(:($op(mesh::Mesh) = $op(mesh.V, mesh.F)))
end

export cot_laplacian
include("heat.jl")