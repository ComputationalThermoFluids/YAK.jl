function bicgstab!(
    x,
    A,
    b;
    Pr = I,
    abstol::Real = zero(mapreduce(eltype, promote_type, (x, A, b))),
    reltol::Real = √eps(mapreduce(eltype, promote_type, (x, A, b))),
    maxiter::Int = length(b) ÷ 4,
    initially_zero::Bool = false,
    verbose::Bool = false,
)

    r = deepcopy(b)
    rs = deepcopy(b)
    !initially_zero && mul!(r, A, x, -1, true) # r = b - A * x

    z = similar(r)
    ldiv!(z, Pr, r) # z = Pr * r

    p = deepcopy(r)
    v = deepcopy(r)
    s = similar(r)
    y = similar(r)
    t = similar(r)
    tn = similar(r)

    ρ, α, ω = 1.0, 1.0, 1.0
    history, iter = [norm(z)], 0
    tol = max(reltol * history[1], abstol)
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
        #CartesianDDM.axpy_axpy!(α, y, ω, z, x) # x = x + α * y + ω * z

        r = deepcopy(s) # r = s 
        mul!(r, t, ω, -1, true) # r = r - ω * t
        push!(history, norm(r))
        ρ = δ
        iter += 1
        verbose && println("iter $(iter) residual: $(history[iter])")
    end
    verbose && println("iter $(length(history)) residual: $(history[end])")
    x
end # bicgstab

function bicgstab(
    A,b;
    Pr = I,
    x0 = nothing,
    abstol::Real = zero(mapreduce(eltype, promote_type, (A, b))),
    reltol::Real = √eps(mapreduce(eltype, promote_type, (A, b))),
    maxiter::Int = length(b) ÷ 4,
    log::Bool = false,
    verbose::Bool = false,
)

    r = deepcopy(b)
    rs = deepcopy(b)

    if x0 === nothing
        x = similar(r)
        fill!(x,0.)
    else
        x = deepcopy(x0)
        mul!(r, A, x, -1, true) # r = b - A * x
    end 

    z = similar(r)
    ldiv!(z, Pr, r) # z = Pr * r

    p = deepcopy(r)
    v = deepcopy(r)
    s = similar(r)
    y = similar(r)
    t = similar(r)
    tn = similar(r)

    ρ, α, ω = 1.0, 1.0, 1.0
    history, iter = [norm(z)], 0
    tol = max(reltol * history[1], abstol)
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
        #CartesianDDM.axpy_axpy!(α, y, ω, z, x) # x = x + α * y + ω * z

        r = deepcopy(s) # r = s 
        mul!(r, t, ω, -1, true) # r = r - ω * t
        push!(history, norm(r))
        ρ = δ
        iter += 1
        verbose && println("iter $(iter) residual: $(history[iter])")
    end
    verbose && println("iter $(length(history)) residual: $(history[end])")
    log ? (x, history) : x #,nothing
end # bicgstab

