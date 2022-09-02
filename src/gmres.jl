"""
    gmres!
    The generalized minimal residual method 'gmres' iteratively solves a linear systems with nonsymmetric matrix or operator (A) and approximates the solution 'x' Krylov subspace of maximal size 'maxiter' with minimal residual or stops when 'norm(A*x - b) < tol'.
    If no preconditioner is supplied, indentity matrix is used.
    The default 'verbose' level is zero, which means no printed output.
"""
function gmres!(
    x,
    A,
    b;
    Pr = I,
    tol::Real = √eps(eltype(x)),
    maxiter::Int = length(b)*length(b),
    restart::Int = length(b),
    )
    m = restart
    history, iter = [norm(b)], 0
    while iter < maxiter && last(history) ≥ tol

        q = Vector{Any}(undef, m + 1)
        J = Vector{Any}(undef, m)
        H = zeros(m + 1, m)
        s = zeros(m + 1)

        rs = deepcopy(b)
        z = similar(rs)

        mul!(rs, A, x, -1, true)
        ldiv!(z, Pr, rs)
        s = [norm(rs)]
        q = [rs / s[1]]

        for i = 1:m
            ldiv!(z, Pr, q[i])

            # Arnoldi iteration
            for k = 1:i
                H[k, i] = dot(z, q[k])
                #axpy!(-H[k, i], q[k], z)
                z -= H[k, i] * q[k]
            end
            H[i+1, i] = norm(z)
            q[i+1] = z / H[i+1, i]

            # Apply previous Givens rotations to solve least squares
            for k = 1:i-1
                #mul!(H[1:i+1, i],J[k],H[1:i+1, i])
                H[1:i+1, i] = J[k] * H[1:i+1, i]
            end
            J[i], = givens(H[i, i], H[i+1, i], i, i + 1)
            # Update s and H
            #mul!(H[1:i+1, i], J[i],H[1:i+1, i])
            H[1:i+1, i] = J[i] * H[1:i+1, i]
            #mul!(s,J[i],s)
            s = J[i] * s

            ##### Solve the projected problem Hy = β * e1 in the least-squares sense
            ##rhs = solve_least_squares!(g.arnoldi, g.β, g.k)

            ## And improve the solution x ← x + Pr \ (V * y)
            ##update_solution!(g.x, view(rhs, 1 : g.k - 1), g.arnoldi, g.Pr, g.k, g.Ax)

            push(history, abs(s[i+1])) # Norm of residual
            # Check residual, compute x, and stop if possible
            if history[end] < tol
                x = gmres_update(x, s, q, i, H)
                @goto exit
            end
        end
        iter += 1
        x = gmres_update(x, s, q, m, H) # Update x before the restart
    end
    @label exit
    return x
end # gmres

function gmres_update(x, s, q, i, H)
    y = H[1:i, 1:i] \ s[1:i]
    for k in eachindex(y)
        x += q[k] * y[k]
    end
    return x
end