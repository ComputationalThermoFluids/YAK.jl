using YAK
using SparseArrays
using Test

@testset "Laplacian" begin
    n = 64

    A = spdiagm(-1 => -ones(n - 1),
                0 => 2ones(n),
                1 => -ones(n - 1))
    b = rand(n)

    x = zero(b)
    ws = bicgstabws(x, A, b)
    history = eltype(ws)[]

    @test ≥(length(ws), 8)

    solved, _ = bicgstab!(x, A, b, ws; history)
    @test solved
end
