
########################################################################
## EMMREML (univariate) translated to Julia for big data computing
## Translated by Uche Godfrey Okeke
## EMMREML was originally written in R by Deniz Akdemir and Uche G. Okeke
## This is a fast and big Data version with autodifferentiation in Julia...
######################################################################### 

using Optim;
#using RCall;
using ForwardDiff, PositiveFactorizations;
using LinearAlgebra, DataFrames;

function emmreml(y, X, Z, K, linenames)

q = size(X,2);
n = size(y,1);
spI = one(rand(n,n));

#### Hat matrix via SVD
#S = spI - (X * pinv(X'X))*X';
U, D, V = svd(X);
S = spI - U*U';
ZK = Z*K;
offset = 0.000001;
ZKZt = ZK*Z';
ZKZtandoffset = ZKZt + (offset * I);
SZKZtSandoffset = Matrix((S * ZKZtandoffset)*S);

### Change to use SVD or eigen decomposition here...
#U, D, V = svd(Hermitian(SZKZtSandoffset));
#Ur = U[:, :1:(n - q)];
#lambda = D[1:(n - q)] .- offset;
#eta = Ur'y;

#### I saw svd fail ... write a try-catch later but use positive eigen for now
D, U = eigen(Positive, Hermitian(SZKZtSandoffset));
Ur = U[:, :1:(n - q)];
lambda = D[1:(n - q)] .- offset;
eta = Ur'y;
	

############################################
##### Use R Optim to optimize
#@rput lambda; @rput n; @rput q; @rput eta;
#R"
#    minimfunc <- function(delta) {
#        (n - q) * log(sum(eta^2/{
#            lambda + delta
#        })) + sum(log(lambda + delta))
#    }
#    optimout <- optimize(minimfunc, lower = 9^(-9), upper = 9^9, 
#        tol = 1e-06)
#    deltahat <- optimout$minimum
#"
#@rget deltahat;
############################################

function minimfunc(delta)
	(n - q) * log.(sum(eta.^2 ./(lambda .+ delta))) + sum(log.(lambda .+ delta))
end

nvar = 1
lower = ([0.00000000001])
upper = ([Inf])
od = OnceDifferentiable(vars -> minimfunc(vars[1]), ones(nvar); autodiff=:forward);
inner_optimizer = LBFGS()
optimout = optimize(od, lower, upper, ones(nvar), Fminbox(inner_optimizer), Optim.Options(show_trace=true))
#opt = optimize(od, ones(nvar), zeros(1)*9.0^(-9), ones(1)*9.0^9)

#### use Simulated Annealing
#optimout = optimize(vars -> minimfunc(vars[1]), ones(nvar), SimulatedAnnealing(), Optim.Options(show_trace=true, allow_f_increases=true, g_tol=1e-6))

deltahat = Optim.minimizer(optimout);
deltahat = reshape(deltahat)[1];
#Hinvhat = pinv(ZKZt + (deltahat * spI));
Hinvhat = cholesky(Positive, (ZKZt + (deltahat * spI))); Hinvhat = inv(Hinvhat);
XtHinvhat = X'Hinvhat;
#betahat = XtHinvhat * X \ XtHinvhat * y;
#### Do cholesky solve for betahat
F, le = ldlt(Positive, XtHinvhat * X);
betahat = F \ XtHinvhat * y;
ehat = y .- (X * betahat);
Hinvhatehat = Hinvhat * ehat;
sigmausqhat = sum(eta.^2 ./(lambda .+ deltahat))/(n - q);
Vinv = (1/sigmausqhat) * Hinvhat;
sigmaesqhat = deltahat * sigmausqhat;
uhat = ZK'Hinvhatehat;
df = n - q;
loglik = -0.5 * (Optim.minimum(optimout) + df + df * log.(2 * pi/df));

#jjj = X'Vinv * X \ X'Vinv;
F, le = ldlt(Positive, X'Vinv * X); ### also use cholesky solve here....
jjj = F \ X'Vinv;
P = Vinv - Vinv * X * jjj;
varuhat = sigmausqhat.^2 * ZK'P * ZK;
PEVuhat = sigmausqhat * K - varuhat;
#varbetahat = pinv(X'Vinv * X);
varbetahat = cholesky(Positive, (X'Vinv * X)); varbetahat = inv(varbetahat);

#uhat = hcat(linenames, uhat); uhat = convert(DataFrame, uhat);
uhat = DataFrame(Lines=linenames, Uhat=uhat);
Vu = sigmausqhat; Ve = sigmaesqhat;  
varuhat = diag(varuhat); varbetahat = diag(varbetahat); PEVuhat = diag(PEVuhat);
h2 = Vu ./(Vu + Ve); rel = 1 .- (PEVuhat ./(Vu * diag(K)));

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
	:h2 => h2,
	:rel => rel
)

return(m11)
  
end

