function face_area_normals(V,F)
    T = V[:,F]
    u = T[:,2,:] - T[:,1,:]
    v = T[:,3,:] - T[:,1,:]
    A = 0.5 * multicross(u,v)
end

face_area(V,F) = norm.(eachcol(face_area_normals(V,F)))
face_centers(V,F) = dropdims(sum(V[:,F], dims=2) ./ 3; dims=2)
function face_normals(V,F)
    A = face_area_normals(V,F)
    normalize!.(eachcol(A))
    A
end

function vertex_area(V,F)
    B = zeros(size(V)[2])
    for f in eachcol(F)
        T = V[:,f]
        x = T[:,1] - T[:,2]
        y = T[:,3] - T[:,2]
        A = 0.5*(sum(cross(x,y).^2)^0.5)
        B[f] .+= A
    end
    B ./= 3
    return B
end

function vertex_normals(V,F)
    A = face_area_normals(V, F)
    n = zero(V)
    for (i, f) in enumerate(eachcol(F))
        for v in f
            n[:, v] += A[:, i]
        end
    end
    normalize!.(eachcol(n))
    return n
end

function FtoV(V,F, face_area)
    A = face_area
    nf = size(F)[2]
    nv = size(V)[2]
    G = sparse(vec(F), vec(repeat(1:nf, 1, 3)'), vec(repeat(A, 1, 3)'), nv, nf)
    G = (G' ./ A)'
end

# function normalize_mesh(V,F)
#     Z = maximum(vec(norm(V,dims=1)))
#     V ./= Z
#     Mesh(V, F) # ???
# end
# function normalize_area(mesh::Mesh)
#     total_area = sum(mesh.face_area)
#     return Mesh(mesh.V/sqrt(total_area), mesh.F, mesh.normals)
# end


######################
# Convenience Methods
######################

# TODO: Figure out how to automate with macros :)
for op in (:face_area_normals, :face_area, :face_centers, :face_normals, :vertex_area, :vertex_normals)
    eval(:($op(mesh::Mesh) = $op(mesh.V, mesh.F)))
end
# macro convenience(ops...)
#     blocks = map(ops) do op
#         quote
#             $op(mesh::Mesh) = $op(mesh.V, mesh.F)
#         end
#     end
#     ex = Expr(:block , blocks...)
#     println(ex)
#     ex
# end

# @convenience(face_area_normals, face_area)