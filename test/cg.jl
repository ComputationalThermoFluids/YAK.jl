module TestCG

using YAK
using Test
using IterativeSolvers

@testset "Conjugate Gradients" begin

    # @testset "SparseMatrixCSC{$T, $Ti}" for T in (Float64, Float32), Ti in (Int64, Int32)
    #     xCG = cg(A, rhs; reltol=reltol, maxiter=100)
    #     xJAC = cg(A, rhs; Pl=P, reltol=reltol, maxiter=100)
    #     @test norm(A * xCG - rhs) ≤ reltol
    #     @test norm(A * xJAC - rhs) ≤ reltol
    # end

end

end # module TestCG
