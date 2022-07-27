# Preconditioned Conjugate Gradient (PCG) for symmetric AS -> not RAS
function cg2!(x, A, b;
             Pr=I,
             maxiter=length(b) ÷ 4,
             abstol=√eps(mapreduce(eltype, promote_type, (x, A, b))))

    r = deepcopy(b)
    mul!(r, A, x, -1, true)

    z = similar(r)
    ldiv!(z, Pr, r)

    p = deepcopy(z)
    q = similar(p)

    iter = 0
    res = [norm(z)]

    while iter < maxiter && last(res) ≥ abstol
        rho = dot(r, z)
        mul!(q, A, p)
        alpha = rho / dot(p, q)

        axpy!(alpha, p, x)
        axpy!(-alpha, q, r)

        ldiv!(z, Pr, r)

        beta = dot(r, z) / rho
        axpby!(true, z, beta, p)

        push!(res, norm(z))
        iter += 1
    end

    iter, res
end
