struct CGWorkspace{T}
    buffer::T
end

const CGWs = CGWorkspace
const CGWsVal = Val(4)

function CGWorkspace(x, args...)
    T = mapreduce(eltype, promote_type, args, init=eltype(x))
    buffer = ntuple(CGWsVal) do _
        similar(x, T)
    end
    CGWorkspace(buffer)
end

default_tol(args...) = √eps(mapreduce(eltype, promote_type, args))
default_maxiter(x, _...) = length(x)
default_history(args...) = mapreduce(eltype, promote_type, args)[]

"""
    cg!

The conjugate gradient method 'CG' iteratively solves a linear system
with a positive definite (and hence symmetric or Hermitian) coefficient
matrix or operator (A) and searches for the optimal 'x' in a Krylov
subspace of maximal size 'maxiter' or stops when 'norm(A*x - b) < tol'.
If no preconditioner is supplied, indentity matrix is used.
The default 'verbose' level is zero, which means no printed output.

# Reference

Dolean, Jolivet & Nataf 2015 - page 97 - Algorithm 3.2
equivalent to Saad 2003 Page 259 - Algorithm 9.1

"""
function cg!(
    x, A, b,
    ws=CGWs(x, A, b),
    Pr=I,
    history=default_history(x, A, b),
    tol=default_tol(x, A, b),
    maxiter=default_maxiter(x, A, b))

    r, z, p, q = ws.buffer

    copy!(r, b)                # r = b
    mul!(r, A, x, -true, true) # r = b - A * x
    ldiv!(z, Pr, r)            # z = Pr \ r
    copy!(p, z)                # p = z

    empty!(history)
    push!(history, norm(z))

    while length(history) ≤ maxiter &&
          last(history) ≥ tol

        ρ = dot(r, z)          # ρ= (r, z)
        mul!(q, A, p)          # q = A * p
        α = ρ / dot(p, q)      # α = ρ / (p, q)
        axpy!(+α, p, x)        # x = x + α * p
        axpy!(-α, q, r)        # r = r - α * q
        ldiv!(z, Pr, r)        # z = Pr \ r
        β = dot(r, z) / ρ      # β = (r, z) / ρ
        axpby!(true, z, β, p)  # p = z + β * p

        push!(history, norm(z))
    end

    return x
end
