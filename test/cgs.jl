# # module TestCGS

# using YAK, Test, SparseArrays, LinearAlgebra

# # @testset "CGS" begin

# # @testset "Small sparse Laplacian system" begin

#     n = 100
#     A = spdiagm(-1 => -ones(n - 1), 0 => 2ones(n), 1 => -ones(n - 1)); b = rand(n)
#     tol = √eps(eltype(b))

#     xref = A \ b
#     @test norm(A * xref - b) ≤ tol

#     x = xref
#     YAK.cgs!(x, A, b)
#     @test norm(A * x - b) ≤ tol

#     #x = rand(n)
#     #YAK.cgs!(x, A, b)
#     #@test norm(A * x - b) ≤ tol

#     x = zeros(n)
#     YAK.cgs!(x, A, b)
#     @test norm(A * x - b) ≤ tol

#     # Test with cholesky factorizaation as preconditioner should converge immediately
#     x = zeros(n)
#     F = LinearAlgebra.lu(A)
#     YAK.cgs!(x, A, b; Pr = F)
#     @test norm(A * x - b) ≤ tol

#     # All-zeros rhs should give all-zeros lhs
#     x = rand(n)
#     b = zeros(n)
#     YAK.cgs!(x, A, b)
#     @test norm(x) ≤ tol

# #end # Small sparse Laplacian system

# #end # CGS

# #end # module TestCGS