module EMMREML

using Optim;
using Statistics, Distributions;
using ForwardDiff, PositiveFactorizations;
using LinearAlgebra, DataFrames;

include("emmremlJulia.jl")
include("emmremlMultivariate_Varcomp.jl")
include("makeGRM.jl")
include("makeRKHS.jl")
include("emmreml_v2.jl")

export emmreml, emmremlMultivariate
export GRM, GRMinv, RKHS, RKHSinv, SqEuclid
export GRMwted, GRMwtedinv, GRMiter
export emmreml_LU

end # module
