###############################################################################
## EMMREML (multikernel univariate) translated to Julia for big data computing
## Translated by Uche Godfrey Okeke
## EMMREML was originally written in R by Deniz Akdemir and Uche G. Okeke
## This is a fast and big Data version with autodifferentiation in Julia...
###############################################################################


using Optim;
using SparseArrays;
using ForwardDiff, PositiveFactorizations;
using LinearAlgebra, DataFrames;

function emmMK(y, X, Zlist, Klist, linenames)

q = size(X,2);
n = size(y,1);
lz = length(Zlist)
spI = one(rand(n,n));
S = spI - ((X * inv(cholesky(Positive, X'*X))) * X')

Z =  hcat(collect(Zlist)...) ## cbind all elements in Zlist

function minimfunctionouter(weights)
         
         weights = weights/sum(weights)
         ZKZt = zeros(n, n);

         for i in 1:lz
         ZKZt = ZKZt + weights[i] * collect(Zlist)[i] * collect(Klist)[i] * (collect(Zlist)[i])'
         end

         offset = 0.000001;

         ZKZtandoffset = ZKZt + (offset * I);
         SZKZtSandoffset = (S * ZKZtandoffset)*S;


         #### I saw svd fail ... write a try-catch later but use positive eigen for now
         D, U = eigen(Positive, Hermitian(Matrix(SZKZtSandoffset)));
         U = reverse(U, dims=2); # reverse bcos PositiveFactorizations sorted eigen vectors and values from smallest to largest ie opposite of svd in R/Julia.
         D = reverse(D);
         Ur = U[:, :1:(n - q)];
         lambda = D[1:(n - q)] .- offset;
         eta = Ur'y;

         function minimfunc(delta)
         (n - q) * log.(sum(eta.^2 ./(lambda .+ delta))) + sum(log.(lambda .+ delta))
         end

         nvar = 1
         lower = ([0.00000000001])
         upper = ([Inf])

         od = OnceDifferentiable(vars -> minimfunc(vars[1]), ones(nvar); autodiff=:forward);
         inner_optimizer = LBFGS()
         optimout = optimize(od, lower, upper, ones(nvar), Fminbox(inner_optimizer), Optim.Options(show_trace=true))

         obj = Optim.minimum(optimout);  ### the objective
         #obj = reshape(obj)[1];

         return(obj)
  end



  nvar = lz
  lower = fill(0.00000000001, lz)
  upper = fill(Inf, lz)
  inner_optimizer = LBFGS()

  weights = optimize(minimfunctionouter, lower, upper, fill(1.0/lz, nvar), Fminbox(inner_optimizer), Optim.Options(show_trace=true))
  weights = Optim.minimizer(weights)


  weights = weights/sum(weights)
  ZKZt = zeros(n,n)

  Klistweighted = Klist;
  for i in 1:lz
    collect(Klistweighted)[i] .= weights[i] * collect(Klist)[i]
    ZKZt = ZKZt + weights[i] * collect(Zlist)[i] * collect(Klist)[i] * (collect(Zlist)[i])'
  end

 
 K = blockdiag([sparse(collect(Klistweighted)[i]) for i=1:lz]...)
 ZK = Z * K
 offset = 0.000001;
 ZKZtandoffset = ZKZt + (offset .* spI);
 SZKZtSandoffset = (S * ZKZtandoffset)*S;

 D, U = eigen(Positive, Hermitian(Matrix(SZKZtSandoffset)));
 U = reverse(U, dims=2); # reverse bcos PositiveFactorizations sorted eigen vectors and values from smallest to largest ie opposite of svd in R/Julia.
 D = reverse(D);
 Ur = U[:, :1:(n - q)];
 lambda = D[1:(n - q)] .- offset;
 eta = Ur'y;

function minimfunc(delta)
  (n - q) * log.(sum(eta.^2 ./(lambda .+ delta))) + sum(log.(lambda .+ delta))
end

nvar = 1
lower = ([0.00000000001])
upper = ([Inf])
od = OnceDifferentiable(vars -> minimfunc(vars[1]), ones(nvar); autodiff=:forward);
inner_optimizer = LBFGS()
optimout = optimize(od, lower, upper, ones(nvar), Fminbox(inner_optimizer), Optim.Options(show_trace=true))

deltahat = Optim.minimizer(optimout);
deltahat = reshape(deltahat)[1];


### use Positive Factorizations cholesky here
Hinvhat = cholesky(Positive, (ZKZt + (deltahat * spI))); Hinvhat = inv(Hinvhat);
XtHinvhat = X'Hinvhat;


#### Do cholesky solve for betahat
F = lu(sparse(XtHinvhat * X));
betahat = F \ XtHinvhat * y;
ehat = y .- (X * betahat);
Hinvhatehat = Hinvhat * ehat;
sigmausqhat = sum(eta.^2 ./(lambda .+ deltahat))/(n - q);
Vinv = (1/sigmausqhat) * Hinvhat;
sigmaesqhat = deltahat * sigmausqhat;
uhat = ZK'Hinvhatehat;
df = n - q;
loglik = -0.5 * (Optim.minimum(optimout) + df + df * log.(2 * pi/df));


namesuhat = Vector{String}();
  for i in 1:lz
  extN = linenames .* ("_" * (string.(i)))
  append!(namesuhat, extN)
  end

### reshape uhat to 2 col matrix
ic1 = length(linenames)
GEBVs = reshape(uhat, (ic1,lz))
GEBVs = sum(GEBVs, dims=2)
                                             
### also use LU solve here....
F = lu(sparse(X'Vinv * X));
jjj = F \ X'Vinv;
P = Vinv - Vinv * X * jjj;
varuhat = sigmausqhat.^2 * ZK'P * ZK;
PEVuhat = sigmausqhat * K - varuhat;

varbetahat = lu(sparse(X'Vinv * X));
varbetahat = inv(X'Vinv * X);

uhatLong = DataFrame(Lines=namesuhat, Uhat=uhat);
uhat = DataFrame(Lines=linenames, Uhat=GEBVs);                                            
Vu = sigmausqhat; Ve = sigmaesqhat;  
varuhat = diag(varuhat); varbetahat = diag(varbetahat); PEVuhat = diag(PEVuhat);
h2 = Vu ./(Vu + Ve); rel_long = 1 .- (PEVuhat ./(Vu * diag(K))); rel = reshape(rel_long, (ic1,lz)); rel = sum(rel, dims=2);

m11 =  
  Dict(
  :Vu => Vu,
  :Ve => Ve,
  :betahat => betahat,
  :uhat => uhat,
  :varuhat => varuhat,
  :varbetahat => varbetahat,
  :PEVuhat => PEVuhat,
  :loglik => loglik,
  :weights => weights,
  :uhatLong => uhatLong,                                                  
  :h2 => h2,
  :rel => rel
)

return(m11)
  
end


