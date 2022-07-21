
function PCG(x, b, M::Function; maxiter, abstol)
    # Preconditioned Conjugate Gradient (PCG) for symmetric ASM -> not RAS
    tic = time()
    un = x * 0.0
    r = b - A(un)
    z = M(r)
    p = z
    res, res[1], ρ = zeros(1), norm(z), 1.0
    for i = 1:maxiter # algorithm 3.2 pg 88 DDM Book
        ρ = dot(r, z)
        q = A(p)
        α = ρ / dot(p, q)
        un += α * p
        r -= α * q
        z = M(r)
        p = z + (dot(r, z) / ρ) * p
        append!(res, norm(z))
        res[i+1] < abstol && @goto exit
    end
    @label exit
    println("PCG i=$(length(res)), absres= $(res[end]) in ", time() - tic, " seconds")
    return un, res
end # PCG

function gmres_update(x, s, q, i, H)
    y = H[1:i, 1:i] \ s[1:i]
    for k in eachindex(y)
        x += q[k] * y[k]
    end
    return x
end

function GMRES(x, b, M::Function; maxiter, abstol, m)
    tic = time()
    res, res[1] = zeros(1), norm(b)
    for j = 1:maxiter

        q = Vector{Any}(undef, m + 1)
        J = Vector{Any}(undef, m)
        H = zeros(m + 1, m)
        s = zeros(m + 1)

        rs = b - A(x)
        r = M(rs)
        s[1] = norm(r)
        q[1] = r / s[1]

        for i = 1:m
            z = M(A(q[i]))

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

            append!(res, abs(s[i+1])) # Norm of residual
            # Check residual, compute x, and stop if possible
            if res[end] < abstol
                x = gmres_update(x, s, q, i, H)
                @goto exit
            end
        end
        x = gmres_update(x, s, q, m, H) # Update x before the restart
    end
    @label exit
    println("GMRES i=$(length(res)), absres= $(res[end]) in ", time() - tic, " seconds")
    return x, res
end

function BiCGSTAB(x, b, M::Function; maxiter, abstol)
    # Preconditioned BiCGstab, two MV operation when compared to PCG
    # We can expect a factor two in iteration count when compared to PCG.
    # Wriggles/peaks make sense since no residual is being minimized.
    tic = time()
    un = x
    r = b - A(un)
    z = M(r)
    rb, p, v = r, r * 0.0, r * 0.0
    ρ, α, ω = 1.0, 1.0, 1.0
    res, res[1] = zeros(1), norm(z)
    for i = 1:maxiter
        append!(res, norm(z))
        res[i+1] < abstol && @goto exit
        δ = dot(rb, r)
        β = (δ * α) / (ρ * ω)
        p = (p - ω * v) * β + r
        yn = M(p)
        v = A(yn)
        α = δ / dot(rb, v)
        rs = α * v - r
        zn = M(rs)
        t = A(zn)
        tn = M(t)
        ω = dot(tn, zn) / dot(tn, tn)
        un += (α * yn + ω * zn) # xi = xim1 + α*y + ω*z
        r = rs - ω * t
        z = M(r)
        ρ = δ
    end
    @label exit
    println("BiCGstab i=$(length(res)), absres= $(res[end]) in ", time() - tic, " seconds")
    return un, res
end # BiCGSTAB

