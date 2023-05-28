# Triangular meshes
struct Mesh
    V::Matrix  # |V|×3
    F::Matrix{Int} # |F|×3
    face_normals::Matrix
    vertex_normals::Matrix
    face_area::Vector
    vertex_area::Vector
    nv::Int # vertex count
    nf::Int # face count
end

function Mesh(V,F)
    A_face = face_area(V,F)
    A_vert = vertex_area(V,F)
    N_face = face_normals(V,F)
    N_vert = vertex_normals(V,F)
    Mesh(V, F, N_face, N_vert, A_face, A_vert, size(V)[1], size(V)[1])
end

function Base.copy(mesh::Mesh)
    Mesh(copy(mesh.V), copy(mesh.F), 
        copy(mesh.face_normals), copy(mesh.vertex_normals), copy(mesh.face_area), copy(mesh.vertex_area),
        mesh.nv, mesh.nf) 
end

# mutable struct MeshOperators
# # ∇_x
# # ∇_y
# # FtoV
#     cot_laplacian
# end

