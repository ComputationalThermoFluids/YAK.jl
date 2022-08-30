# YAK.jl

| **Build Status** | **License** |
|:----------------:|:-----------:|
[![CI][github-img]][github-url] | [![license][license-img]][license-url] |

[github-img]: https://github.com/ComputationalThermoFluids/YAK.jl/actions/workflows/ci.yml/badge.svg
[github-url]: https://github.com/ComputationalThermoFluids/YAK.jl/actions/workflows/ci.yml

[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[license-url]: LICENSE.md

Yet Another Krylov (YAK.jl) is another Julia package that provides efficient iterative [Krylov subspace](https://en.wikipedia.org/wiki/Krylov_subspace) algorithms for solving dense or sparse large linear systems, developed entirely with  [BLAS and sparse BLAS](http://cali2.unilim.fr/intel-xe/mkl/mklman/GUID-707FB65C-D0D9-418A-B22E-CBDEFB163C02.htm)
[level-1 routines](http://cali2.unilim.fr/intel-xe/mkl/mklman/GUID-A050F064-A146-49F7-B22E-BBB1E1DD6B3F.htm#GUID-A050F064-A146-49F7-B22E-BBB1E1DD6B3F) (vector-vector operations).

Available algorithms:
- conjugate gradient method ([CG](https://en.wikipedia.org/wiki/Conjugate_gradient_method)),
- conjugate gradient squared method ([CGS](https://mathworld.wolfram.com/ConjugateGradientSquaredMethod.html)),
- generalized minimal residual method ([GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)),
- biconjugate gradient method ([BiCG](https://en.wikipedia.org/wiki/Biconjugate_gradient_method)),
- biconjugate gradient stabilized method ([BiCGSTAB](https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method)).