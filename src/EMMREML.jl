module EMMREML

using Optim;
using Statistics, Distributions;
using ForwardDiff, PositiveFactorizations;
using LinearAlgebra, DataFrames;

include("emmremlJulia.jl")
include("emmremlMultivariate_Varcomp.jl")
include("makeGRM.jl")
include("makeRKHS.jl")
include("calcHmat.jl")
#include("emmreml_v2.jl") ## uncomment for using SuperLU in julia >=1.6 

export emmreml, emmremlMultivariate
export GRM, GRMinv, RKHS, RKHSinv, SqEuclid
export GRMwted, GRMwtedinv, GRMiter, GRMVariter, DOM
#export emmreml_LU
export Hmat

end # module
