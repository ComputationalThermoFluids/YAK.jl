
function linsolve(A, b, x₀, alg::CG)

    if x₀ === nothing
        x = similar(b);fill!(b, 0.0)
    else
        x = x₀
    end

    r = deepcopy(b)
    mul!(r, A, x, -1, true) # r = b - A * x

    normr = norm(r)
    #S = typeof(normr)

    # Algorithm parameters
    maxiter = alg.maxiter
    # tol::S = alg.tol
    tol = alg.tol
    numops = 1 # operator A has been applied once to determine r
    numiter = 0

    # Check for early return
    normr < tol && return x #(x, ConvergenceInfo(1, r, normr, numiter, numops))

    # First iteration
    ρ = normr^2
    p = mul!(similar(r), r, 1)
    q = similar(b)
    q = mul!(q,A,p) #apply(A, p, α₀, α₁)
    α = ρ / dot(p, q)
    x = axpy!(+α, p, x)
    r = axpy!(-α, q, r)
    normr = norm(r)
    ρold = ρ
    ρ = normr^2
    β = ρ / ρold
    numops += 1
    numiter += 1
    if alg.verbose > 1
        msg = "CG linsolve in iter $numiter: "
        msg *= "normres = "
        msg *= @sprintf("%.12e", normr)
        @info msg
    end

    # Check for early return
    normr < tol && return x # (x, ConvergenceInfo(1, r, normr, numiter, numops))
    
    z = similar(b)
    while numiter < maxiter
        axpby!(1, r, β, p)
        q = mul!(q,A,p) # apply(A, p, α₀, α₁)
        α = ρ / dot(p, q)
        x = axpy!(+α, p, x)
        r = axpy!(-α, q, r)
        normr = norm(r)
        if normr < tol # recompute to account for buildup of floating point errors
            r = mul!(r, b, 1)
            mul!(z,A,x) # apply(A, x, α₀,α₁),
            r = axpy!(-1, z, r) # axpy!(-1, apply(A, x, α₀,α₁), r)
            normr = norm(r)
            ρ = normr^2
            β = zero(β) # restart CG
        else
            ρold = ρ
            ρ = normr^2
            β = ρ / ρold
        end
        if normr < tol
            if alg.verbose > 0
                @info """CG linsolve converged at iteration $numiter:
                 *  norm of residual = $normr
                 *  number of operations = $numops"""
            end
            return x # (x, ConvergenceInfo(1, r, normr, numiter, numops))
        end
        numops += 1
        numiter += 1
        if alg.verbose > 1
            msg = "CG linsolve in iter $numiter: "
            msg *= "normres = "
            msg *= @sprintf("%.12e", normr)
            @info msg
        end
    end
    if alg.verbose > 0
        @warn """CG linsolve finished without converging after $numiter iterations:
         *  norm of residual = $normr
         *  number of operations = $numops"""
    end
    return x # (x, ConvergenceInfo(0, r, normr, numiter, numops))
end



function linsolve(A, b, x₀, alg::BiCGStab)
    x = mul!(similar(b), x₀, 1)
    y₀ = similar(b)
    r = similar(b)
    mul!(y₀,A,x₀)
    r = axpy!(-1., y₀, r)
    normr = norm(r)

    # Algorithm parameters
    maxiter = alg.maxiter
    tol = alg.tol
    numops = 1 # operator has been applied once to determine r
    numiter = 0

    # Check for early return
    if normr < tol
        if alg.verbose > 0
            @info """BiCGStab linsolve converged without any iterations:
             *  norm of residual = $normr
             *  number of operations = 1"""
        end
        return x# (x, ConvergenceInfo(1, r, normr, numiter, numops))
    end

    # First iteration
    numiter += 1
    r_shadow = mul!(similar(r), r, 1) # shadow residual
    ρ = dot(r_shadow, r)

    if ρ ≈ 0.0 # Method fails if ρ is zero.
        @warn """BiCGStab linsolve errored after $numiter iterations:
        *   norm of residual = $normr
        *   number of operations = $numops"""
        return x#(x, ConvergenceInfo(0, r, normr, numiter, numops))
    end

    ## BiCG part of the algorithm.
    p = mul!(similar(r), r, 1)
    v = similar(b)
    mul!(v,A,p)
    numops += 1

    σ = dot(r_shadow, v)
    α = ρ / σ

    s = mul!(similar(r), r, 1)
    s = axpy!(-α, v, s) # half step residual

    xhalf = mul!(similar(x), x, 1)
    xhalf = axpy!(+α, p, xhalf) # half step iteration

    normr = norm(s)
    z = similar(b)
    # Check for early return at half step.
    if normr < tol
        # Replace approximate residual with the actual residual.
        s = mul!(similar(b), b, 1)
        mul!(z,A,xhalf)
        s = axpy!(-1, z, s)
        numops += 1

        normr_act = norm(s)
        if normr_act < tol
            if alg.verbose > 0
                @info """BiCGStab linsolve converged at iteration $(numiter-1/2):
                 *  norm of residual = $normr_act
                 *  number of operations = $numops"""
            end
            return xhalf #(xhalf, ConvergenceInfo(1, s, normr_act, numiter, numops))
        end
    end

    ## GMRES part of the algorithm.
    t = similar(b) #
    mul!(t,A,s)
    numops += 1

    ω = dot(t, s) / dot(t, t)

    x = mul!(x, xhalf, 1)
    x = axpy!(+ω, s, x) # full step iteration

    r = mul!(r, s, 1)
    r = axpy!(-ω, t, r) # full step residual

    # Check for early return at full step.
    normr = norm(r)
    if normr < tol
        # Replace approximate residual with the actual residual.
        r = mul!(r, b, 1)
        mul!(z,A,x)
        r = axpy!(-1, z, r)
        numops += 1

        normr_act = norm(r)
        if normr_act < tol
            if alg.verbose > 0
                @info """BiCGStab linsolve converged at iteration $(numiter):
                *  norm of residual = $normr_act
                *  number of operations = $numops"""
            end
            return x # (x, ConvergenceInfo(1, r, normr_act, numiter, numops))
        end
    end

    while numiter < maxiter
        if alg.verbose > 0
            msg = "BiCGStab linsolve in iter $numiter: "
            msg *= "normres = "
            msg *= @sprintf("%12e", normr)
            @info msg
        end

        numiter += 1
        ρold = ρ
        ρ = dot(r_shadow, r)
        β = (ρ / ρold) * (α / ω)

        p = axpy!(-ω, v, p)
        p = axpby!(1, r, β, p)

        mul!(v,A,p)
        numops += 1

        σ = dot(r_shadow, v)
        α = ρ / σ

        s = mul!(s, r, 1)
        s = axpy!(-α, v, s) # half step residual

        xhalf = mul!(xhalf, x, 1)
        xhalf = axpy!(+α, p, xhalf) # half step iteration

        normr = norm(s)

        if alg.verbose > 0
            msg = "BiCGStab linsolve in iter $(numiter-1/2): "
            msg *= "normres = "
            msg *= @sprintf("%12e", normr)
            @info msg
        end

        # Check for return at half step.
        if normr < tol
            # Compute non-approximate residual.
            s = mul!(similar(b), b, 1)
            mul!(z,A,xhalf)
            s = axpy!(-1, z, s)
            numops += 1

            normr_act = norm(s)
            if normr_act < tol
                if alg.verbose > 0
                    @info """BiCGStab linsolve converged at iteration $(numiter-1/2):
                    *  norm of residual = $normr_act
                    *  number of operations = $numops"""
                end
                return xhalf #(xhalf, ConvergenceInfo(1, s, normr_act, numiter, numops))
            end
        end

        ## GMRES part of the algorithm.
        mul!(t,A,s)
        numops += 1

        ω = dot(t, s) / dot(t, t)

        x = mul!(x, xhalf, 1)
        x = axpy!(+ω, s, x) # full step iteration

        r = mul!(r, s, 1)
        r = axpy!(-ω, t, r) # full step residual

        # Check for return at full step.
        normr = norm(r)
        if normr < tol
            # Replace approximate residual with the actual residual.
            r = mul!(r, b, 1)
            mul!(z,A,x)
            r = axpy!(-1, z, r)
            numops += 1

            normr_act = norm(r)
            if normr_act < tol
                if alg.verbose > 0
                    @info """BiCGStab linsolve converged at iteration $(numiter):
                    *  norm of residual = $normr_act
                    *  number of operations = $numops"""
                end
                return x #(x, ConvergenceInfo(1, r, normr_act, numiter, numops))
            end
        end
    end

    if alg.verbose > 0
        @warn """BiCGStab linsolve finished without converging after $numiter iterations:
        *   norm of residual = $normr
        *   number of operations = $numops"""
    end
    return x# (x, ConvergenceInfo(0, r, normr, numiter, numops))
end




function cg(
    A,b;
    Pr = I,
    x0 = nothing,
    abstol::Real = zero(mapreduce(eltype, promote_type, (A, b))),
    reltol::Real = √eps(mapreduce(eltype, promote_type, (A, b))),
    maxiter::Int = length(b)*length(b),
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
        mul!(r, A, x, -1., true) # r = b - A * x
    end 

    cg!(x, A, b;abstol=1e-7,initially_zero = false,verbose = true)
#

#    verbose && println("iter $(length(history)) residual: $(history[end])")
#    log ? (x, history) : (x,)
end

function cg!(
    x,
    A,
    b;
    Pr = I, # avoid Identity()
    abstol::Real = zero(eltype(x)),
    reltol::Real = √eps(eltype(x)),
    maxiter::Int = length(b)*length(b),
    initially_zero::Bool = true,
    verbose::Bool = false)

    if initially_zero # skip an operation if x is zero
        r = deepcopy(b)
    else
        r = similar(b)
        mul!(r, A, x, -1., true) # r = b - A * x 
    end
    
    z = similar(r)
    ldiv!(z, Pr, r) # overwrite z with solve: Pr * z = r -> z = Pr \ r

    p = deepcopy(z)
    q = similar(p)

    history, iter = [norm(z)], 0
    tol = abstol # max(reltol * history[1], abstol)
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
        verbose && println("iter $(iter) residual: $(history[iter])")
        iter == maxiter && println("maxiter $(maxiter) reached!")
    end
    println(history)
    verbose && println("iter $(length(history)) residual: $(history[end])")
    x
end





,Printf

# A general structure to pass on convergence information
"""
    struct ConvergenceInfo{S,T}
        converged::Int
        residual::T
        normres::S
        numiter::Int
        numops::Int
    end
Used to return information about the solution found by the iterative method.
  - `converged`: the number of solutions that have converged according to an appropriate
    error measure and requested tolerance for the problem. Its value can be zero or one for
    [`linsolve`](@ref), [`exponentiate`](@ref) and  [`expintegrator`](@ref), or any integer
    `>= 0` for [`eigsolve`](@ref), [`schursolve`](@ref) or [`svdsolve`](@ref).
  - `residual:` the (list of) residual(s) for the problem, or `nothing` for problems without
    the concept of a residual (i.e. `exponentiate`, `expintegrator`). This is a single
    vector (of the same type as the type of vectors used in the problem) for `linsolve`, or
    a `Vector` of such vectors for `eigsolve`, `schursolve` or `svdsolve`.
  - `normres`: the norm of the residual(s) (in the previous field) or the value of any other
    error measure that is appropriate for the problem. This is a `Real` for `linsolve` and
    `exponentiate`, and a `Vector{<:Real}` for `eigsolve`, `schursolve` and `svdsolve`. The
    number of values in `normres` that are smaller than a predefined tolerance corresponds
    to the number `converged` of solutions that have converged.
  - `numiter`: the number of iterations (sometimes called restarts) used by the algorithm.
  - `numops`: the number of times the linear map or operator (A) was applied
"""
struct ConvergenceInfo{S,T}
    converged::Int # how many vectors have converged, 0 or 1 for linear systems, exponentiate, any integer for eigenvalue problems
    residual::T
    normres::S
    numiter::Int
    numops::Int
end
function Base.show(io::IO, info::ConvergenceInfo)
    print(io, "ConvergenceInfo: ")
    info.converged == 0 && print(io, "no converged values ")
    info.converged == 1 && print(io, "one converged value ")
    info.converged > 1 && print(io, "$(info.converged) converged values ")
    println(
        io,
        "after ",
        info.numiter,
        " iterations and ",
        info.numops,
        " applications of the linear map;"
    )
    return println(io, "norms of residuals are given by $((info.normres...,)).")
end

abstract type LinearSolver end

export CG
struct CG{S<:Real} <: LinearSolver
    maxiter::Int
    tol::S
    verbose::Int
end
CG(;
    maxiter::Integer = 1000,
    tol::Real = 1e-12,
    verbose::Int = 0
) = CG(maxiter, tol, verbose)

export BiCGStab
struct BiCGStab{S<:Real} <: LinearSolver
    maxiter::Int
    tol::S
    verbose::Int
end
BiCGStab(;
    maxiter::Integer = 1000,
    tol::Real = 1e-12,
    verbose::Int = 0
) = BiCGStab(maxiter, tol, verbose)
export linsolve
include("linsolve.jl")




export cgs!
include("cgs.jl")

export gmres!
include("gmres.jl")

using LinearAlgebra,IterativeSolvers,SparseArrays; 
n = 100; T = Float64; abstol = √eps(float(T)); b = rand(T, n); #A = rand(T, n, n); A = A'I *A; # SPD
A = spdiagm(-1=>-ones(n-1), 0=>2ones(n), 1=>-ones(n-1))
x = rand(T, n); Pr = I; abstol = zero(mapreduce(eltype, promote_type, (x, A, b))); reltol = √eps(mapreduce(eltype, promote_type, (x, A, b))); maxiter = length(b)*length(b)*length(b); x_initially_zero = false; verbose = false;


l = YAK.CG(maxiter=n, tol=abstol, verbose = 3)
@time x =  YAK.linsolve(A, b, x0, l)
@time norm(A * x - b)

l = YAK.BiCGStab(maxiter=n, tol=abstol, verbose = 3)
@time x =  YAK.linsolve(A, b, x0, l)
@time norm(A * x - b)