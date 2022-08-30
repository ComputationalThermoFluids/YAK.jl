# YAK.jl


Yet Another Krylov (YAK.jl) solver is yet another Julia package that offers efficient iterative [Krylov subspace](https://en.wikipedia.org/wiki/Krylov_subspace) algorithms for solving large linear systems designed with [BLAS and Sparse BLAS](http://cali2.unilim.fr/intel-xe/mkl/mklman/GUID-707FB65C-D0D9-418A-B22E-CBDEFB163C02.htm)
[Level 1 Routines](http://cali2.unilim.fr/intel-xe/mkl/mklman/GUID-A050F064-A146-49F7-B22E-BBB1E1DD6B3F.htm#GUID-A050F064-A146-49F7-B22E-BBB1E1DD6B3F) (vector-vector operations).

Available algorithms:
- conjugate gradient method ([CG](https://en.wikipedia.org/wiki/Conjugate_gradient_method)),
- conjugate gradient squared method ([CGS](https://mathworld.wolfram.com/ConjugateGradientSquaredMethod.html)),
- generalized minimal residual method ([GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)),
- biconjugate gradient method ([BiCG](https://en.wikipedia.org/wiki/Biconjugate_gradient_method)),
- biconjugate gradient stabilized method ([BiCGSTAB](https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method)).


| **Build Status** | **License** |
|:----------------:|:-----------:|
[![CI][github-img]][github-url] | [![license][license-img]][license-url] |

[github-img]: https://github.com/ComputationalThermoFluids/YAK.jl/actions/workflows/ci.yml/badge.svg
[github-url]: https://github.com/ComputationalThermoFluids/YAK.jl/actions/workflows/ci.yml

[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[license-url]: LICENSE.md