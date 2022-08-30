module TestCG

using YAK,Test,LinearAlgebra

#@testset "Small full system" begin
    n = 100

    #@testset "Matrix{$T}" for T in (Float32, Float64, ComplexF32, ComplexF64)
        T = Float64; abstol = √eps(float(T));
        A = rand(T, n, n); A = A' * A + I; b = rand(T, n);

        x = zeros(T, n); @time cg!(x,A,b)
        @time norm(A*x-b)

        x = zeros(T, n); @time cg2!(x,A,b)
        @time norm(A*x - b)

        #@test norm(A*x - b) / norm(b) ≤ reltol

        # If you start from the exact solution, you should converge immediately

        @time x = YAK.cg(A,b)
        @time norm(A*x - b)

        xref = A\b; norm(A*xref - b)
        x,his = YAK.cg(A,b;x0=xref,log=true)
        @test norm(A*x - b) ≤ abstol
        @test length(his) ≤ 2

        # Test with cholesky factorizaation as preconditioner should converge immediately
        F = LinearAlgebra.cholesky(A, Val(false))
        x,his = YAK.cg(A,b; Pr=F , log=true)
        @test norm(A*x - b) ≤ abstol

        # All-zeros rhs should give all-zeros lhs
        x0 = YAK.cg(A, zeros(T, n))
        @test x0 == zeros(T, n)
    #end
#end

end # module TestCG
