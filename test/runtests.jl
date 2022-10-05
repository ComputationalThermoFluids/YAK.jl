using YAK
using LinearAlgebra
using Test

@testset "Laplacian" begin
    n = 64

    A = Tridiagonal(-ones(n-1), 2ones(n), -ones(n-1))
    b = rand(n)

    Pl = Bidiagonal(A, :U)
    Pr = Bidiagonal(A, :L)

    x = zero(b)
    ws = bicgstabws(x, A, b)
    history = eltype(ws)[]

    @test ≥(length(ws), 8)

    solved, _ = bicgstab!(x, A, b, ws; Pl, Pr, history)
    @test solved
end
