module EMMREML

using Optim, RCall;
using Statistics, Distributions, NamedArrays;
using ForwardDiff, PositiveFactorizations, AltDistributions;
using LinearAlgebra, DataFrames, InvertedIndices, PedigreeBase;

include("emmremlJulia.jl")
include("emmremlMultivariate_Varcomp.jl")
include("makeGRM.jl")
include("makeRKHS.jl")
#include("calcHmat.jl")
include("emmremlMultiKernel.jl")
#include("emmreml_v2.jl") ## uncomment for using SuperLU in julia >=1.6 

export emmreml, emmremlMultivariate
export GRM, GRMinv, RKHS, RKHSinv, SqEuclid
export GRMwted, GRMwtedinv, GRMiter, GRMVariter, DOM
#export emmreml_LU
#export Hmat, computeA, Hmat2, pedMat, HmatNet, HmatEpis, Hmat_alt
export emmMK

end # module
