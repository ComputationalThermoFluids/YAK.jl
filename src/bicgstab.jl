"""
    bicgstab!
    The biconjugate gradient stabilized method 'bicgstab!' iteratively solves a linear systems with nonsymmetric  matrix or operator (A) and searches for the optimal 'x' in a Krylov subspace of maximal size 'maxiter' or stops when 'norm(A*x - b) < tol'.
    If no preconditioner is supplied, indentity matrix is used.
    The default 'verbose' level is zero, which means no printed output.
"""
# Dolean, Jolivet & Nataf 2015 - page 103 - Algorithm 3.4
function bicgstab!(
    x,
    A,
    b;
    Pr = I,
    tol::Real = √eps(eltype(x)),
    maxiter::Int = length(b)*length(b))

    r = deepcopy(b)
    rs = deepcopy(b)
    mul!(r, A, x, -1, true) # r = b - A * x

    z = similar(r)
    ldiv!(z, Pr, r) # z = Pr * r

    p = deepcopy(r)
    v = deepcopy(r)
    s = similar(r)
    y = similar(r)
    t = similar(r)
    tn = similar(r)

    ρ, α, ω = 1., 1., 1.
    history, iter = [norm(z)], 0
    while iter < maxiter && last(history) ≥ tol
        
        δ = dot(rs, r)
        β = (δ * α) / (ρ * ω)
        d = deepcopy(p)
        axpy!(-ω, v, d) # d = d - ω * v
        p = deepcopy(r)
        axpy!(β, d, p) # p = p + β * d

        ldiv!(y, Pr, p) # y = M^{-1} * p
        mul!(v, A, y) # v = A * y
        α = δ / dot(rs, v)
        s = deepcopy(r) # s = r 
        mul!(s, v, α, -1, true) # r = r - α * v

        ldiv!(z, Pr, s) # z = M^{-1} * s 
        mul!(t, A, z) # t = A * z
        ldiv!(tn, Pr, t) # tn = M^{-1} * t 
        ω = dot(tn, z) / dot(tn, tn)

        axpy!(α, y, x) # x = x + α * y 
        axpy!(ω, z, x) # x = x + ω * z

        r = deepcopy(s) # r = s 
        mul!(r, t, ω, -1, true) # r = r - ω * t
        push!(history, norm(r))
        ρ = δ
        iter += 1
    end
    return x
end # bicgstab