module TestBicgstab

using YAK, Test, SparseArrays, LinearAlgebra

@testset " Bicgstab" begin

@testset "Small sparse Laplacian system" begin

    n = 100
    A = spdiagm(-1 => -ones(n - 1), 0 => 2ones(n), 1 => -ones(n - 1)); b = rand(n)
    tol = √eps(eltype(b))

    xref = A \ b
    @test norm(A * xref - b) ≤ tol

    x = xref
    YAK.bicgstab!(x, A, b)
    @test norm(A * x - b) ≤ tol

    x = rand(n)
    YAK.bicgstab!(x, A, b)
    @test norm(A * x - b) ≤ tol

    x = zeros(n)
    YAK.bicgstab!(x, A, b)
    @test norm(A * x - b) ≤ tol

    # Test with cholesky factorizaation as preconditioner should converge immediately
    x = zeros(n)
    F = LinearAlgebra.lu(A)
    YAK.bicgstab!(x, A, b; Pr = F)
    @test norm(A * x - b) ≤ tol

    # # All-zeros rhs should give all-zeros lhs
    # x = rand(n)
    # b = zeros(n)
    # YAK.bicgstab!(x, A, b)
    # norm(x)
    # @test norm(x) ≤ tol

end # Small sparse Laplacian system

end # Bicgstab

end # module TestBicgstab
