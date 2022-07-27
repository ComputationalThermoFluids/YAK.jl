
function cgs!(
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
    log ? (x, history) : x
end # CGS