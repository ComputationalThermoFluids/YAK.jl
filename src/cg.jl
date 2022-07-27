
function cg!(
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
    log ? (x, history) : x
end # cg