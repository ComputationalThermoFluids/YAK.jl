
"""
    cg!(x, A, b; kwargs...) -> x, [history]
Solves the problem ``Ax = b`` with conjugate gradient.
# Arguments
- `x`: Initial guess, will be updated in-place;
- `A`: linear operator;
- `b`: right-hand side.
## Keywords
- `initially_zero::Bool`: If `true` assumes that `iszero(x)` so that one
  matrix-vector product can be saved when computing the initial
  residual vector;
- `abstol::Real = zero(real(eltype(b)))`,
  `reltol::Real = sqrt(eps(real(eltype(b))))`: absolute and relative
  tolerance for the stopping condition
  `|r_k| ≤ max(reltol * |r_0|, abstol)`, where `r_k = A * x_k - b`
- `restart::Int = min(20, size(A, 2))`: restarts GMRES after specified number of iterations;
- `maxiter::Int = size(A, 2)`: maximum number of inner iterations of GMRES;
- `Pl`: left preconditioner;
- `Pr`: right preconditioner;
- `log::Bool`: keep track of the residual norm in each iteration;
- `verbose::Bool`: print convergence information during the iterations.
# Return values
**if `log` is `false`**
- `x`: approximate solution.
**if `log` is `true`**
- `x`: approximate solution;
- `history`: convergence history.
"""
function cg!(
    x,
    A,
    b;
    Pr = Identity(),
    abstol::Real = zero(real(eltype(b))),
    #reltol::Real = sqrt(eps(real(eltype(b)))),
    maxiter::Int = size(A, 2),
    log::Bool = false,
    initially_zero::Bool = false,
    verbose::Bool = false)

    r = b - A * x
    z = similar(r)
    ldiv!(z,Pr,r)
    p = z
    history, history[1], ρ = zeros(1), norm(z), 1.0
    for i = 1:maxiter # algorithm 3.2 pg 88 DDM Book
        ρ = dot(r, z)
        q = A * p
        α = ρ / dot(p, q)
        x += α * p
        r -= α * q
        ldiv!(z,Pr,r)
        p = z + (dot(r, z) / ρ) * p
        append!(history, norm(z))
        history[i+1] < abstol && @goto exit
    end
    @label exit
    verbose && println("i=$(length(res)), absres= $(res[end])")
    log ? (x, history) : x
end # cg

function gmres_update(x, s, q, i, H)
    y = H[1:i, 1:i] \ s[1:i]
    for k in eachindex(y)
        x += q[k] * y[k]
    end
    return x
end

# function update_solution!(x, y, arnoldi::ArnoldiDecomp{T}, Pr::Identity, k::Int, Ax) where {T}
#     # Update x ← x + V * y
#     mul!(x, view(arnoldi.V, :, 1 : k - 1), y, one(T), one(T))
# end

# function update_solution!(x, y, arnoldi::ArnoldiDecomp{T}, Pr, k::Int, Ax) where {T}
#     # Computing x ← x + Pr \ (V * y) and use Ax as a work space
#     mul!(Ax, view(arnoldi.V, :, 1 : k - 1), y)
#     ldiv!(Pr, Ax)
#     x .+= Ax
# end

function gmres!(
    x,
    A,
    b;
    #Pl = Identity(),
    Pr = Identity(),
    abstol::Real = zero(real(eltype(b))),
    #reltol::Real = sqrt(eps(real(eltype(b)))),
    restart::Int = min(20, size(A, 2)),
    maxiter::Int = size(A, 2),
    log::Bool = false,
    initially_zero::Bool = false,
    verbose::Bool = false)

    m = restart
    history, history[1] = zeros(1), norm(b)
    for j = 1:maxiter

        q = Vector{Any}(undef, m + 1)
        J = Vector{Any}(undef, m)
        H = zeros(m + 1, m)
        s = zeros(m + 1)

        rs = b - A * x
        ldiv!(z,Pr,rs)
        s[1] = norm(r)
        q[1] = r / s[1]

        for i = 1:m
            z = Pr(A * q[i])

            # Arnoldi iteration
            for k = 1:i
                H[k, i] = dot(z, q[k])
                z -= H[k, i] * q[k]
            end
            H[i+1, i] = norm(z)
            q[i+1] = z / H[i+1, i]

            # Apply previous Givens rotations to solve least squares
            for k = 1:i-1
                H[1:i+1, i] = J[k] * H[1:i+1, i]
            end
            J[i], = givens(H[i, i], H[i+1, i], i, i + 1)
            # Update s and H
            H[1:i+1, i] = J[i] * H[1:i+1, i]
            s = J[i] * s

            ##### Solve the projected problem Hy = β * e1 in the least-squares sense
            ##rhs = solve_least_squares!(g.arnoldi, g.β, g.k)

            ## And improve the solution x ← x + Pr \ (V * y)
            ##update_solution!(g.x, view(rhs, 1 : g.k - 1), g.arnoldi, g.Pr, g.k, g.Ax)

            append!(history, abs(s[i+1])) # Norm of residual
            # Check residual, compute x, and stop if possible
            if history[end] < abstol
                x = gmres_update(x, s, q, i, H)
                @goto exit
            end
        end
        x = gmres_update(x, s, q, m, H) # Update x before the restart
    end
    @label exit
    verbose && println("i=$(length(history)), absres= $(history[end])")
    log ? (x, history) : x
end # gmres


function bicgstab!(
    x,
    A,
    b;
    #Pl = Identity(),
    Pr = Identity(),
    abstol::Real = zero(real(eltype(b))),
    #reltol::Real = sqrt(eps(real(eltype(b)))),
    maxiter::Int = size(A, 2),
    log::Bool = false,
    initially_zero::Bool = false,
    verbose::Bool = false)

    r = b - A * x
    z = Pr * r
    rb, p, v = r, r * 0.0, r * 0.0
    ρ, α, ω = 1.0, 1.0, 1.0
    history, history[1] = zeros(1), norm(z)
    for i = 1:maxiter
        append!(history, norm(z))
        history[i+1] < abstol && @goto exit
        δ = dot(rb, r)
        β = (δ * α) / (ρ * ω)
        p = (p - ω * v) * β + r
        ldiv!(yn,Pr,p)
        v = A * yn
        α = δ / dot(rb, v)
        rs = α * v - r
        ldiv!(zn,Pr,rs)
        t = A * zn
        ldiv!(tn,Pr,t)
        ω = dot(tn, zn) / dot(tn, tn)
        x += (α * yn + ω * zn) # xi = xim1 + α*y + ω*z
        r = rs - ω * t
        ldiv!(z,Pr,r)
        ρ = δ
    end
    @label exit
    verbose && println("i=$(length(history)), absres= $(history[end])")
    log ? (x, history) : x
end # bicgstab

"""
    gmres(A, b; kwargs...) -> x, [history]
Same as [`gmres!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
gmres(A, b; kwargs...) = gmres!(zerox(A, b), A, b; initially_zero = true, kwargs...)

"""
    cg(A, b; kwargs...) -> x, [history]
Same as [`cg!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
cg(A, b; kwargs...) = cg!(zerox(A, b), A, b; initially_zero = true, kwargs...)

"""
    bicgstab(A, b; kwargs...) -> x, [history]
Same as [`bicgstab!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
bicgstab(A, b; kwargs...) = bicgstab!(zerox(A, b), A, b; initially_zero = true, kwargs...)