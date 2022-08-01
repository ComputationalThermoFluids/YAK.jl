
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


function cg3(
    x,
    A,
    b;
    Pr = I, # avoid Identity()
    abstol::Real = zero(mapreduce(eltype, promote_type, (x, A, b))),
    reltol::Real = √eps(mapreduce(eltype, promote_type, (x, A, b))),
    maxiter::Int = length(b) ÷ 4,
    log::Bool = false,
    initially_zero::Bool = false,
    verbose::Bool = false,
)
    r = deepcopy(b)
    !initially_zero && mul!(r, A, x, -1, true) # r = b - A * x

    z = similar(r)
    ldiv!(z, Pr, r) # overwrite z with solve Pr z = r -> z = Pr \ r

    p = deepcopy(z)
    q = similar(p)

    ρ = norm(z)
    history, iter = [ρ], 0
    tol = max(reltol * history[1], abstol)
    while iter < maxiter && last(history) ≥ tol
        mul!(q, A, p) # q = A * p
        ρ = dot(r, z)
        α = ρ / dot(p, q)
        axpy!(+α, p, x) # x += α * p
        axpy!(-α, q, r) # r -= α * q
        β = dot(r, z) / ρ
        axpby!(true, z, β, p) # p = z + β * p
        push!(history, norm(z))
        verbose && println("iter $(iter) residual: $(history[iter])")
        iter += 1
    end
    verbose && println("iter $(length(history)) residual: $(history[end])")
    ifelse(log, (x, history), (x,))
end # cg

function gmres_update(x, s, q, i, H)
    y = H[1:i, 1:i] \ s[1:i]
    for k in eachindex(y)
        x += q[k] * y[k]
    end
    return x
end

function gmres3!(
    x,
    A,
    b;
    Pr = I, # avoid Identity()
    abstol::Real = zero(mapreduce(eltype, promote_type, (x, A, b))),
    reltol::Real = √eps(mapreduce(eltype, promote_type, (x, A, b))),
    restart::Int = min(20, size(A, 2)),
    maxiter::Int = length(b) ÷ 4,
    log::Bool = false,
    initially_zero::Bool = false,
    verbose::Bool = false,
)

    m = restart
    history = [norm(b)]
    tol = max(reltol * history[1], abstol)
    for j = 1:maxiter

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
            if history[end] < abstol
                x = gmres_update(x, s, q, i, H)
                @goto exit
            end
        end
        x = gmres_update(x, s, q, m, H) # Update x before the restart
    end
    @label exit
    verbose && println("iter $(length(history)) residual: $(history[end])")
    ifelse(log, (x, history), (x,))
end # gmres


function bicgstab3!(
    x,
    A,
    b;
    Pr = I,
    abstol::Real = zero(mapreduce(eltype, promote_type, (x, A, b))),
    reltol::Real = √eps(mapreduce(eltype, promote_type, (x, A, b))),
    maxiter::Int = length(b) ÷ 4,
    log::Bool = false,
    initially_zero::Bool = false,
    verbose::Bool = false,
)

    r = deepcopy(b)
    rs = deepcopy(b)
    !initially_zero && mul!(r, A, x, -1, true) # r = b - A * x

    z = similar(r)
    ldiv!(z, Pr, r) # z = Pr * r

    p = similar(r) 
    v = similar(r) 
    s = similar(r) 
    y = similar(r) 
    t = similar(r) 
    tn = similar(r) 
    ρ, α, ω = 1.0, 1.0, 1.0
    history = [norm(z)]
    tol = max(reltol * history[1], abstol)
    iter = 0
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
        #ω = dot(t, z) / dot(t, t)
        axpy!(α, y, x) # x = x + α * y 
        axpy!(ω, z, x) # x = x + ω * z
        r = deepcopy(s) # r = s 
        mul!(r, t, ω, -1, true) # r = r - ω * t
        push!(history, norm(r))
        ρ = δ
        iter += 1
        verbose && println("iter $(iter) residual: $(history[iter])")
    end
    verbose && println("iter $(length(history)) residual: $(history[end])")
    log ? (x, history) : x
end # bicgstab



function cgs3!(
    x,
    A,
    b;
    Pr = I,
    abstol::Real = zero(mapreduce(eltype, promote_type, (x, A, b))),
    reltol::Real = √eps(mapreduce(eltype, promote_type, (x, A, b))),
    maxiter::Int = length(b) ÷ 4,
    log::Bool = false,
    initially_zero::Bool = false,
    verbose::Bool = false,
)

    r = deepcopy(b)
    rs = deepcopy(b)
    !initially_zero && mul!(r, A, x, -1, true) # r = b - A * x

    z = similar(r)
    ldiv!(z, Pr, r) # z = Pr * r
    u = deepcopy(r)
    p = deepcopy(r)
    q = deepcopy(r)
    y = similar(r)
    v = similar(r)

    Az = similar(r)
    ρ, α, ω = 1.0, 1.0, 1.0
    history = [norm(z)]
    tol = max(reltol * history[1], abstol)
    iter = 0
    while iter < maxiter && last(history) ≥ tol
        δ = dot(rs, r)
        β = δ / ρ
        u = deepcopy(r)
        axpy!(β, q, u) # u = r + β * q
        p = deepcopy(u)
        axpy!(β, q, p) # p = p + β * q
        axpy!(β*β, p, p) # p = p + β * β * p
        ldiv!(y, Pr, p) # y = M^{-1} * p
        mul!(v, A, y) # v = A * y
        α = δ / dot(rs, v)
        q = deepcopy(u) # q = u 
        axpy!(-α, v, q) # q = q - α * v
        ldiv!(z, Pr, u+q) # z = M^{-1} * (u + q)
        axpy!(+α, z, x) # x = x + α * y 
        mul!(Az, A, z) # Az = A * z
        axpy!(-α, Az, r) # x = x + ω * z
        push!(history, norm(r))
        ρ = δ
        iter += 1
        verbose && println("iter $(iter) residual: $(history[iter])")
    end
    verbose && println("iter $(length(history)) residual: $(history[end])")
    ifelse(log, (x, history), (x,))
end # CGS
