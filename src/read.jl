function readoff(filename::String)
    X, T = open(filename) do f
        s = readline(f)
        if s[1:3] != "OFF"
            error("The file is not a valid OFF one.")
        end
        s = readline(f)
        nv, nt = parse.(Int, split(s));
        X = zeros(Float64, 3, nv);
        T = zeros(Int, 3, nt);
        for i=1:nv
            s = readline(f)
            X[:,i] = parse.(Float64, split(s))
        end
        for i=1:nt
            s = readline(f)
            T[:,i] = parse.(Int64, split(s))[2:end] .+ 1
        end
        X, T
    end
end

function readply(fname)
    ply = load_ply(fname)
    x = ply["vertex"]["x"]
    y = ply["vertex"]["y"]
    z = ply["vertex"]["z"]
    V = hcat(x,y,z)'
    F = stack(ply["face"]["vertex_indices"])'
    # points = Meshes.Point3.(x, y, z)
    V, F
end
