struct Workspace{T,N,A<:AbstractVector{T}}
    data::NTuple{N,A}
end

parent(this::Workspace) = this.data
eltype(this::Workspace{T}) where {T} = T
length(this::Workspace{T,N}) where {T,N} = N

function Workspace(::Val{N}, x::AbstractVector, args...) where {N}
    T = mapreduce(eltype, promote_type, args, init=eltype(x))
    data = ntuple(Val(N)) do _
        similar(x, T)
    end
    Workspace(data)
end

bicgstabws(args...) = Workspace(Val(8), args...)

"""
This implementation assumes an initial guess is provided in
the first input. If not, simply set it to zero before hand.

"""
function bicgstab!(x, A, b, ws::Workspace{T}, c=b;
                   Pl=I,
                   Pr=I,
                   itmax=2length(x),
                   atol=√eps(T),
                   rtol=√eps(T),
                   history=Vector{T}(undef, itmax+1)) where {T}

    u, r, p, v, s, q, y, t = parent(ws)

    r₀ = d = q
    z = y

    mul!(r₀, A, x)                 # r₀ = A x
    axpby!(one(T), b, -one(T), r₀) # r₀ = b - r₀

    fill!(u, zero(eltype(u)))      # x₀
    fill!(s, zero(eltype(s)))      # s₀
    fill!(v, zero(eltype(v)))      # v₀

    ldiv!(r, Pl, r₀)               # r₀
    copy!(p, r)                    # p₁

    α = one(T)                     # α₀
    ω = one(T)                     # ω₀
    ρ = one(T)                     # ρ₀

    res = norm(r)

    it = 1
#    empty!(history)
#    push!(history, norm(r))

    # zero-residual
    iszero(res) &&
        return true, false, false, it-1

    tol = atol + rtol * res

    next = dot(c, r)               # ρ₁ = ⟨r̅₀,r₀⟩

    # breakdown (b⋅c = 0)
    iszero(next) &&
        return true, false, true, it-1

    # stopping criteria
    solved = ≤(res, tol)
    tired = >(it, itmax)
    broken = false

    resize!(history, itmax+1)
    history[it] = norm(r)

    while !|(solved, tired, broken)
        ρ = next

        ldiv!(y, Pr, p)            # yₖ = N⁻¹pₖ
        mul!(q, A, y)              # qₖ = Ayₖ
        ldiv!(v, Pl, q)            # vₖ = M⁻¹qₖ
        α = ρ / dot(c, v)          # αₖ = ⟨r̅₀,rₖ₋₁⟩ / ⟨r̅₀,vₖ⟩
        copy!(s, r)                # sₖ = rₖ₋₁
        axpy!(-α, v, s)            # sₖ = sₖ - αₖvₖ
        axpy!(α, y, u)             # xₐᵤₓ = xₖ₋₁ + αₖyₖ
        ldiv!(z, Pr, s)            # zₖ = N⁻¹sₖ
        mul!(d, A, z)              # dₖ = Azₖ
        ldiv!(t, Pl, d)            # tₖ = M⁻¹dₖ
        ω = dot(t, s) / dot(t, t)  # ⟨tₖ,sₖ⟩ / ⟨tₖ,tₖ⟩
        axpy!(ω, z, u)             # xₖ = xₐᵤₓ + ωₖzₖ
        copy!(r, s)                # rₖ = sₖ
        axpy!(-ω, t, r)            # rₖ = rₖ - ωₖtₖ
        next = dot(c, r)           # ρₖ₊₁ = ⟨r̅₀,rₖ⟩
        β = (next / ρ) * (α / ω)   # βₖ₊₁ = (ρₖ₊₁ / ρₖ) * (αₖ / ωₖ)
        axpy!(-ω, v, p)            # pₐᵤₓ = pₖ - ωₖvₖ
        axpby!(one(T), r, β, p)    # pₖ₊₁ = rₖ₊₁ + βₖ₊₁pₐᵤₓ

        # residual norm
        res = norm(r)
        it += 1

        # stopping criteria
        solved = ≤(res + one(T), one(T)) ||
                 ≤(res, tol)
        tired = it > itmax
        broken = iszero(α) || isnan(α)

        history[it] = res
    end

    # update x
    axpy!(one(T), u, x)

    return solved, tired, broken, it-1
end
