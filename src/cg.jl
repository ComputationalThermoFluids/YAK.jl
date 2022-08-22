# Preconditioned Conjugate Gradient (PCG) for symmetric AS
function cg!(
    x,
    A,
    b;
    Pr = I, # avoid Identity()
    abstol::Real = zero(mapreduce(eltype, promote_type, (x, A, b))),
    reltol::Real = √eps(mapreduce(eltype, promote_type, (x, A, b))),
    maxiter::Int = length(b) ÷ 4,
    initially_zero::Bool = false,
    verbose::Bool = false)

    r = deepcopy(b)
    !initially_zero && mul!(r, A, x, -1, true) # r = b - A * x

    z = similar(r)
    ldiv!(z, Pr, r) # overwrite z with solve: Pr * z = r -> z = Pr \ r

    p = deepcopy(z)
    q = similar(p)

    ρ = norm(z)
    history, iter = [ρ], 0
    tol = max(reltol * history[1], abstol)
    while iter < maxiter && last(history) ≥ tol
        mul!(q, A, p) # q = A * p
        ρ = dot(r, z)
        α = ρ / dot(p, q)
        axpy!( α, p, x) # x += α * p
        axpy!(-α, q, r) # r -= α * q
        ldiv!(z, Pr, r)
        β = dot(r, z) / ρ
        axpby!(true, z, β, p) # p = z + β * p
        push!(history, norm(z))
        iter += 1
        verbose && println("iter $(iter) residual: $(history[iter])")
    end
    verbose && println("iter $(length(history)) residual: $(history[end])")
    x
end

function cg(
    A,b;
    Pr = I,
    x0 = nothing,
    abstol::Real = zero(mapreduce(eltype, promote_type, (A, b))),
    reltol::Real = √eps(mapreduce(eltype, promote_type, (A, b))),
    maxiter::Int = length(b) ÷ 4,
    log::Bool = false,
    verbose::Bool = false)

    r = deepcopy(b)
    z = similar(r)
    ldiv!(z, Pr, r) # overwrite z with solve Pr z = r -> z = Pr \ r

    if x0 === nothing
        x = similar(r)
        fill!(x,0.)
    else
        x = deepcopy(x0)
        mul!(r, A, x, -1, true) # r = b - A * x
    end 

    p = deepcopy(z)
    q = similar(p)

    ρ = norm(z)
    history, iter = [ρ], 0
    tol = max(reltol * history[1], abstol)
    while iter < maxiter && last(history) ≥ tol
        mul!(q, A, p) # q = A * p
        ρ = dot(r, z)
        α = ρ / dot(p, q)
        axpy!( α, p, x) # x += α * p
        axpy!(-α, q, r) # r -= α * q
        ldiv!(z, Pr, r)
        β = dot(r, z) / ρ
        axpby!(true, z, β, p) # p = z + β * p
        push!(history, norm(z))
        iter += 1
        verbose && println("iter $(iter) residual: $(history[iter])")
    end
    verbose && println("iter $(length(history)) residual: $(history[end])")
    log ? (x, history) : x #,nothing
end
