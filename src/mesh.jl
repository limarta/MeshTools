# Triangular meshes
struct Mesh{M<:AbstractMatrix,N<:AbstractMatrix{Int}, T, U<:AbstractVector}
    V::M  # 3×|V|
    F::N # 3×|F|
    face_normals::T
    vertex_normals::T
    face_area::U
    vertex_area::U
    nv::Int # vertex count
    nf::Int # face count
end

function Mesh(V,F)
    N_face = face_normals(V,F)
    N_vert = vertex_normals(V,F)
    A_face = face_area(V,F)
    A_vert = vertex_area(V,F)
    Mesh(V, F, N_face, N_vert, A_face, A_vert, size(V)[2], size(F)[2])
end

function Base.copy(mesh::Mesh)
    Mesh(copy(mesh.V), copy(mesh.F), 
        copy(mesh.face_normals), copy(mesh.vertex_normals), copy(mesh.face_area), copy(mesh.vertex_area),
        mesh.nv, mesh.nf) 
end
