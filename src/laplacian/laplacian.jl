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


function laplacian_basis(L, A; k=20, mode::Symbol = :arnoldi)
    if mode == :arnoldi
# # Define a matrix whose eigenvalues you want
# # Factorizes A and builds a linear map that applies inv(A) to a vector.
# function construct_linear_map(A)
#     F = lu(A)
#     LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x) , size(A,1), ismutating=true)
# end

# # Target the largest eigenvalues of the inverted problem
# function eig(L)
#     decomp, = partialschur(construct_linear_map(L), nev=128, tol=1e-10, restarts=100, which=LM())
#     λs_inv, X = partialeigen(decomp)
# end
# λs_inv, X =  eig(L)
        error()
    elseif mode == :arpack
        λ, ϕ  = eigs(L, A, nev=k, sigma=1e-6)
        λ = real.(λ)
        λ[1] = 0
        ϕ = real.(ϕ)
        # Normalize wrt to A
        Z = sqrt.(vec(sum(ϕ .* (A * ϕ), dims=1)))
        ϕ ./= Z'
        λ, ϕ 
    else
        error()
    end
end
# function laplacian_basis(L)
#     eigs()
# end

# TODO: Figure out how to automate with macros :)
for op in (:cot_laplacian,)
    eval(:($op(mesh::Mesh) = $op(mesh.V, mesh.F)))
end

export cot_laplacian
include("heat.jl")