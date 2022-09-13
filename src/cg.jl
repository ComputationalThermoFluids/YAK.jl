"""
    cg!
    The conjugate gradient method 'CG' iteratively solves a linear system with a positive definite (and hence symmetric or Hermitian) coefficient matrix or operator (A) and searches for the optimal 'x' in a Krylov subspace of maximal size 'maxiter' or stops when 'norm(A*x - b) < tol'.
    If no preconditioner is supplied, indentity matrix is used.
    The default 'verbose' level is zero, which means no printed output.

"""
# Dolean, Jolivet & Nataf 2015 - page 97 - Algorithm 3.2
# equivalent to Saad 2003 Page 259 - Algorithm 9.1 
function cg!(
    x,
    A,
    b;
    Pr = I,
    tol::Real = √eps(eltype(x)),
    maxiter::Int = length(b)*length(b))
    
    r = deepcopy(b)
    mul!(r, A, x, -1., true) # r = b - A * x 
    
    z = similar(r)
    ldiv!(z, Pr, r) # overwrite z with Pr \ r

    p = deepcopy(z)
    q = similar(p)

    history, iter = [norm(z)], 0
    while iter ≤ maxiter && last(history) ≥ tol
        ρ = dot(r, z)
        mul!(q, A, p) # q = A * p
        α = ρ / dot(p, q)
        axpy!(+α, p, x) # x = x + α * p
        axpy!(-α, q, r) # r = r - α * q
        ldiv!(z, Pr, r)
        β = dot(r, z) / ρ
        axpby!(true, z, β, p) # p = z + β * p
        push!(history, norm(z))
        iter += 1
    end
    return x
end
