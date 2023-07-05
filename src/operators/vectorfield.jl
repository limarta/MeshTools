function face_grad(mesh::Mesh)
    V = mesh.V
    F = mesh.F
    ∇ = spzeros(3 * mesh.nf, mesh.nv)
    A = face_area(mesh)
    N = mesh.face_normals

    u = repeat(F[1,:], inner=3)
    v = repeat(F[2,:], inner=3)
    w = repeat(F[3,:], inner=3)
    uv = V[:,F[2,:]] - V[:,F[1,:]]
    vw = V[:,F[3,:]] - V[:,F[2,:]]
    wu = V[:,F[1,:]] - V[:,F[3,:]]
    J = 1:3*mesh.nf
    G2 = cross.(eachcol(N), eachcol(wu)) ./ A
    G1 = cross.(eachcol(N), eachcol(vw)) ./ A
    G3 = cross.(eachcol(N), eachcol(uv)) ./ A
    G1 = collect(Iterators.flatten(G1))
    G2 = collect(Iterators.flatten(G2))
    G3 = collect(Iterators.flatten(G3))
    g = sparse([J;J;J], [u;v;w], [G1; G2; G3], 3*mesh.nf, mesh.nv)
    g ./= 2

    # for f=1:mesh.nf
    #     u, v, w = F[:,f]
    #     vw = V[:,w] - V[:,v]
    #     wu = V[:,u] - V[:,w]
    #     uv = V[:,v] - V[:,u]
    #     J = 3f .+ (-2:0)
    #     ∇[J, u], ∇[J, v], ∇[J, w] = map(e->cross(N[:,f], e)/2A[f], [vw, wu, uv])
    # end
    # println(norm(g - ∇))
    return g
end

function vertex_grad(V,F,N)
    nv = size(V)[2]
    frames = tangent_basis(V,F,N)
    ∇ = zeros(2,nv,nv)
    one_ring_neighbors = Vector{Set}()
    for v in 1:nv
        push!(one_ring_neighbors, Set{Int}())
    end
    
    for f in eachcol(F)
        for v in f
            union!(one_ring_neighbors[v], f)
        end
    end

    count = sum(length.(one_ring_neighbors))
    I = Vector{Int}(undef,count)
    J = Vector{Int}(undef,count)
    V_x = Vector{Float64}(undef, count)
    V_y = Vector{Float64}(undef, count)
    j = 1
    for i=1:nv # Naive
        # P(∇f) = Df -> Solve via least squares -> ∇ = (P'P)^{-1}P'D
        neighbors = one_ring_neighbors[i]
        delete!(neighbors, i)
        neighbors = collect(neighbors)
        edges = V[:,neighbors]
        center = V[:,i]
        edges = edges .- center
        proj_edges = embed_in_plane(frames[:,i,:], edges)'
        D_ = zeros(length(neighbors), length(neighbors)+1)
        D_[:,1] .= -1
        ind_ = CartesianIndex.(1:length(neighbors), 2:length(neighbors)+1)
        D_[ind_] .= 1
        grads = proj_edges \ D_
        # println("grads size: ", size(grads), " neighbors ", length(neighbors))
        len = length(neighbors)+1
        I[j:j+len-1] = fill(i, len)
        J[j] = i
        J[j+1:j+len-1] =  neighbors
        V_x[j:j+len-1] = grads[1,:]
        V_y[j:j+len-1] = grads[2,:]
        j = j+len
    end
    I = convert.(Int, I)
    J = convert.(Int, J)
    V_x = convert.(Float32, V_x)
    V_y = convert.(Float32, V_y)
    ∇_x = sparse(I, J, V_x)
    ∇_y = sparse(I,J, V_y)
    ∇_x, ∇_y
end
function embed_in_plane(frame, edges)
    # Project the edges onto the tangent plane defined by frame
    e_1 = frame[:,1]
    e_2 = frame[:,2]
    c_1 = sum(e_1 .* edges; dims=1)
    c_2 = sum(e_2 .* edges; dims=1)
    embedding = [c_1; c_2]
end

function project_to_plane(normal::AbstractMatrix, u::AbstractVector)
    # Project u onto the tangent plane defined by vertex normals
    u .- vdot(normal, u;dims=1) ./ sum(normal.^2; dims=1) .* normal
end

function tangent_basis(V,F,N)
    # frame 3×|V|×2
    # N - vertex normals
    nv = size(V)[2]
    e_1 = [1.0, 0, 0]
    e_2 = [0, 1.0, 0]
    t_1 = project_to_plane(N, e_1)
    t_1 = t_1 ./ norm(t_1; dims=1)
    t_2 = project_to_plane(N, e_2)
    t_2 = t_2 ./ norm(t_2; dims=1)
    frame = zeros(3,nv,2)
    frame[:,:,1] = t_1
    frame[:,:,2] = t_2
    frame
end

function world_coordinates(mesh::Mesh, gradients)
    frame = tangent_basis(mesh)
    x_1 = gradients[1,:]' .* frame[:,:,1]
    x_2 = gradients[2,:]' .* frame[:,:,2] 
    x_1 + x_2
end

function div(mesh::Mesh)
    # ∇⋅ is |V|×3|F|
    V = mesh.V
    F = mesh.F
    # ∇ = spzeros(mesh.nv, 3 * mesh.nf)
    uv = V[:,F[2,:]] - V[:,F[1,:]]
    vw = V[:,F[3,:]] - V[:,F[2,:]]
    wu = V[:,F[1,:]] - V[:,F[3,:]]
    cotan = -[sum(uv.*wu; dims=1); sum(vw.*uv; dims=1); sum(wu.*vw; dims=1)]
    s = [norm.(cross.(eachcol(uv), eachcol(wu)));;
        norm.(cross.(eachcol(vw), eachcol(uv)));;
        norm.(cross.(eachcol(wu), eachcol(vw)))]'
    cotan ./= s

    u = repeat(F[1,:], inner=3)
    v = repeat(F[2,:], inner=3)
    w = repeat(F[3,:], inner=3)
    J = 1:3*mesh.nf
    A = vec(cotan[2,:]' .* uv - cotan[3,:]' .* wu)
    B = vec(-cotan[1,:]' .* uv + cotan[3,:]' .* wu)
    C = vec(cotan[1,:]' .* wu - cotan[2,:]' .* vw)
    ∇ = sparse([u;v;w], [J;J;J], [A;B;C], mesh.nv, 3*mesh.nf)
    ∇ ./= 2

end

export face_grad, vertex_grad