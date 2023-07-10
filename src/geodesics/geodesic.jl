function heat_method(mesh::Mesh, L, i; dt = 1.0)
    # https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/
    nv = mesh.nv
    init = zeros(nv)
    init[i] = 1.0
    heat = heat_implicit(mesh, L, init, dt=dt)
    ∇ = face_grad(mesh)
    grad_H = reshape(∇ * heat, 3, :)
    grad_H = grad_H ./ sqrt.(sum(grad_H .^ 2, dims=1))
    Δ = div(mesh)
    X = Δ * vec(grad_H)
    dist = L \ X
    dist .-= minimum(dist)
    dist
end
"""
Fast Marching
Similar to Dijkstra's algorithm
-Fast exact and approximate geodesics on meshes https://dl.acm.org/doi/10.1145/1073204.1073228
-NUMERICAL GEOMETRY OF NON-RIGID SHAPES
-https://code.google.com/archive/p/geodesic/
"""

function fast_marching(mesh, source)
    nv = mesh.nv
    V = mesh.V
    F = mesh.F
    dists = Dict{Int, Float64}([i=>i ∈ source ? 0 : Inf for i=1:mesh.nv])
    visited = Set{Int}()
    frontier = PriorityQueue([i=>0.0 for i in source])
    while length(visited) != nv
        u, d = peek(frontier)
        dequeue!(frontier)
        push!(visited, u)

        adj = findall([u in f for f in eachcol(F)])
        for f in adj
            T = F[:,f]
            for v in T
                v ∈ visited && continue
                push!(frontier, v=>dists[v])
                x,y = setdiff(T,v)
                pos =  V[:,setdiff(T, v)] .- V[:,v]
                d = [dists[x], dists[y]]
                if any(isinf.(d))
                    d3 = minimum(d + norm.(eachcol(pos)))
                else
                    d3 = eikonal_update(pos, d)
                    # Check for consistency
                end
                dists[v] = min(dists[v], d3)
            end
        end
    end
    dists
end

function eikonal_update(V, d)
    Q = inv(V' * V)
    disc = sum(Q * d)^2 - (sum(Q) * (d' * Q * d -1))
    if disc < 0
        return Inf
    end
    p = (sum(Q * d) + sqrt(disc)) / sum(Q)
end

export fast_marching, eikonal_update, heat_method