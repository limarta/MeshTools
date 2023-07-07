# https://igl.ethz.ch/projects/ARAP/arap_web.pdf
function arap_energy(V1, V2, arap_state)
    R = arap_state.R
    neighbors = arap_state.neighbors
    weights = arap_state.weights
    # Compute the ARAP energy
    E = 0.0
    for v=1:size(V1,2)
        vu = V1[:,neighbors[v]]
        vu_new = V2[:, neighbors[v]]
        E_u = vu_new - R[v] * vu
        E_u = vec(sum(E_u .* E_u; dims=1))
        w_ij = weights[v,neighbors[v]]
        E_u = sum(E_u .* w_ij)
        E +=  E_u
    end
    E
end
arap_energy(mesh1::Mesh, mesh2::Mesh, arap_state) = arap_energy(mesh1.V, mesh2.V, arap_state)

vmul(A,B) = eachcol(A) .* transpose.(eachcol(B))

function optimal_rotation(V1, V2, arap_state)
    nv = size(V1,2)
    R = Vector{RotMatrix3{Float64}}(undef, nv)
    neighbors = arap_state.neighbors
    weights = arap_state.weights
    for v=1:nv
        vu = V1[:, neighbors[v]]
        vu_new = V2[:, neighbors[v]]
        cov = vmul(vu,vu_new)
        cov = sum(cov .* weights[v, neighbors[v]])
        factors = svd(cov)
        R_i = (factors.U * factors.Vt)'
        if det(R_i) < 0
            U_copy = copy(factors.U)
            flip_ind = argmin(factors.S)
            U_copy[:,flip_ind] .*= -1
            R_i = (U_copy * factors.Vt)'
        end
        R[v] = RotMatrix3(R_i)
        # println(det(R_i), " " )
    end
    R
end

function constraint_simplification(V, L, constraints)
    if length(constraints) == 0
        return L, zeros(size(L,1), 3), 1:size(L,2)
    end
    v = collect(keys(constraints))
    w = L[:,v] 
    constrained_points = [constraints[p] for p in v]
    for i=eachindex(v)
        V[:,v[i]] .= constrained_points[i]
    end
    c = sum(eachcol(w) .* transpose.(constrained_points))
    indices = [i for i=1:size(L,2) if i∉v]
    L_new = L[:, indices]
    V, L_new, c, indices
end

function arap_minimization(mesh, constraints, L)
    V = mesh.V
    nv = mesh.nv
    V_new = copy(V)
    V_new, L_new, c, indices = constraint_simplification(V_new, L, constraints);
    nn = MeshTools.nearest_neighbors(mesh)
    state = ARAPState(nn, L, ones(RotMatrix3{Float64}, nv))
    for i=1:20
        println("Energy: ", arap_energy(V, V_new, state))
        R_new = optimal_rotation(V, V_new, state)
        state = ARAPState(nn, L, R_new)
        V_new =  optimal_position(V, V_new, state, L, L_new, c, indices)
    end
    MeshTools.Mesh(V_new, copy(mesh.F))
end
# function arap_minimization(mesh::Mesh, constraints, L)
#     V_new = arap_minimization(mesh.V, constraints, L)
#     MeshTools.Mesh(V_new, copy(mesh.F))
# end

function optimal_position(V1,V2, state, L, L_new, c, indices)
    # println("optimal position")
    # println(size(L_new))
    # println(size(c))
    neighbors = state.neighbors
    R = state.R
    b = Matrix{Float64}(undef, size(V1,2),3)
    for i in axes(V1,2)
        temp = zeros(3)
        for j in neighbors[i]
            temp += L[i,j] * (R[i] + R[j]) * (V1[:,i] - V1[:,j])
        end
        b[i,:] = temp
    end

    b = b/2 .- c
    p_new = L_new \ b
    # println("p_new ", findall(isnan.(p_new)))
    V2[:,indices] = p_new'
    V2 
end

function nearest_neighbors(mesh::Mesh)
    neighbors = Dict{Int, Set{Int}}(k=>Set{Int}() for k=1:mesh.nv)
    for f=1:mesh.nf
        for (i,v) in enumerate(mesh.F[:,f])
            for (j,u) in enumerate(mesh.F[:,f])
                if i != j
                    push!(neighbors[v], u)
                end
            end
        end
    end
    d = Dict{Int, Vector{Int}}()
    for (key, value_set) in neighbors
        d[key] = collect(value_set)
    end
    d 
end

struct ARAPState{U<:AbstractVecOrMat{RotMatrix3{Float64}}, L<:AbstractMatrix}
    neighbors::Dict{Int, Vector{Int}}
    weights::L
    R::U
end

struct ARAPConstraint
end

export arap_minimization, arap_energy, nearest_neighbors, optimal_rotation, ARAPState