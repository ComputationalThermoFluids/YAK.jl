# YAK.jl

[![CI][github-img]][github-url] [![license][license-img]][license-url]

[github-img]: https://github.com/ComputationalThermoFluids/YAK.jl/actions/workflows/ci.yml/badge.svg
[github-url]: https://github.com/ComputationalThermoFluids/YAK.jl/actions/workflows/ci.yml

[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[license-url]: LICENSE.md
 
Yet Another Krylov (YAK.jl) is another Julia package that provides efficient iterative [Krylov subspace](https://en.wikipedia.org/wiki/Krylov_subspace) algorithms for solving dense or sparse large linear systems, developed entirely with  [BLAS and sparse BLAS](http://cali2.unilim.fr/intel-xe/mkl/mklman/GUID-707FB65C-D0D9-418A-B22E-CBDEFB163C02.htm)
[level-1](http://cali2.unilim.fr/intel-xe/mkl/mklman/GUID-A050F064-A146-49F7-B22E-BBB1E1DD6B3F.htm#GUID-A050F064-A146-49F7-B22E-BBB1E1DD6B3F) (vector-vector operations) and [level-2](http://cali2.unilim.fr/intel-xe/mkl/mklman/GUID-9B9E459C-4E87-4A5E-8BC3-2FE06C86D0F1.htm#GUID-9B9E459C-4E87-4A5E-8BC3-2FE06C86D0F1) (matrix-vector operations).

Thanks to [multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY), introducing the necessary [methods](https://docs.julialang.org/en/v1/manual/methods/) with support your data-type shall give you acess to the following algorithms:

- conjugate gradient method ([CG](https://en.wikipedia.org/wiki/Conjugate_gradient_method)),
- conjugate gradient squared method ([CGS](https://mathworld.wolfram.com/ConjugateGradientSquaredMethod.html)),
- generalized minimal residual method ([GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)),
- biconjugate gradient method ([BiCG](https://en.wikipedia.org/wiki/Biconjugate_gradient_method)),
- biconjugate gradient stabilized method ([BiCGSTAB](https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method)).

Subroutines are provided in both Julia __asigment__ or __mutation__ forms:

```julia
using LinearAlgebra, YAK    
A = rand(100, 100); b = rand(100);

# asigment form 
@time x = cg(A,b)
norm(A*x-b)

# mutation form
x = zeros(100); 
@time cg!(x,A,b)
norm(A*x-b)
```