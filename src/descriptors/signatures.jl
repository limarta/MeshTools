function heatKernelSignature(λ, ϕ, A, t)
    c = (A*ϕ)' .* exp.(-λ.* t)
    hks = vec(sum(ϕ' .* c, dims=1))
    
    # Which is correct?
    # D = ϕ' * (A * ϕ.^2);
    # T = exp.(-(λ.*t));
    # hks = D*T;
    # hks = ϕ*hks;
end

# SLOW!
function heatKernelSignature(L, A, t)
    M = spdiagm(A)
    D = Matrix(M+t*L)
    heat = D \ M
    return diag(heat,0)
end

function waveKernelSignature()
    # TODO
end