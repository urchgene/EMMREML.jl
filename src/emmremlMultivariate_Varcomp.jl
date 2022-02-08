
#################################################
### EMMREML-MULTIVARIATE for Julia
### Modified and translated by Uche Godfrey Okeke
### Suitable for Big data purposes
### This is specifically for estimating Vg and Ve
##################################################


using LinearAlgebra;
using Distributions, AltDistributions;
using Statistics, PositiveFactorizations;

#BLAS.set_num_threads(20)
#my_BLAS_set_num_threads(n) =  ccall((:openblas_set_num_threads64_, Base.libblas_name), Cvoid, (Int32,), n)
#my_BLAS_set_num_threads(20)

function emmremlMultivariate(Y, X, Z, K)

## call with regular Y, X, K but transpose Z ###

	Z = Z; X = X'; Y = Y'; tolpar = 1e-06; tolparinv = 1e-06;


    function ECM1(ytl, xtl, Vgt, Vet, Bt, deltal)
        Vlt = deltal * Vgt + Vet
        invVlt = inv(Vlt + tolparinv * I)
        gtl = deltal * Vgt * invVlt * (ytl - Bt * xtl)
        Sigmalt = deltal * Vgt - deltal * Vgt * invVlt * (deltal * Vgt)
        return(Dict(:Vlt => Vlt, :gtl => gtl, :Sigmalt => Sigmalt))
    end

    function wrapperECM1(l)
        ytl = Yt[:, l]
        xtl = Xt[:, l]
        #deltal = eigZKZt.values[l]
	deltal = D[l]	
        return(ECM1(ytl, xtl, Vgt, Vet, Bt, deltal))
    end


    function Vgfunc(l, outfromECM1)
        Vgl = outfromECM1[l][:gtl] * outfromECM1[l][:gtl]'
        #return((1/n) * (1 ./ eigZKZt.values[l]) * (Vgl + outfromECM1[l][:Sigmalt]))
	return((1/n) * (1 ./ D[l]) * (Vgl + outfromECM1[l][:Sigmalt]))	
    end


    function Vefunc(l, outfromECM1)
        etl = Yt[:, l] - Bt * Xt[:, l] - outfromECM1[l][:gtl]
        return((1/n) * (etl*etl' + outfromECM1[l][:Sigmalt]))
    end

    if (!ismissing(Y))
        N = size(K,1)
        KZt = K * Z'
        ZKZt = Z * KZt
        ZKZt  = Matrix(ZKZt + 0.0001*I);
        #eigZKZt = eigen(ZKZt)
	eigZKZt = eigen(Positive, Hermitian(ZKZt));
	U = reverse(eigZKZt.vectors, dims=2); 
	D = reverse(eigZKZt.values);
		
        n = size(ZKZt, 1)
        d = size(Y, 1)
        #Yt = Y * eigZKZt.vectors
        #Xt = X * eigZKZt.vectors
	Yt = Y * U
        Xt = X * U
	
        Vgt = cov(Y')/2
        Vet = cov(Y')/2
        XttinvXtXtt = Xt' * pinv(Xt*Xt')
        Bt = Yt * XttinvXtXtt
        Vetm1 = Vet

        j = 1;
        while true
            println("iteration ...", j)
            outfromECM1 = [wrapperECM1(l) for l in 1:n]
            Vetm1 = Vet
            hh = [outfromECM1[i][:gtl] for i in 1:n]; 
            Gt = reduce(hcat, hh);
            Bt = (Yt - Gt) * XttinvXtXtt
            listVgts = [Vgfunc(l, outfromECM1) for l in 1:n]
            Vgt = reduce(+, listVgts)
            listVets = [Vefunc(l, outfromECM1) for l in 1:n]
            Vet = reduce(+, listVets)
            convnum = abs(sum(diag(Vet - Vetm1)))/abs(sum(diag(Vetm1)))

            if convnum < tolpar
                break
            end
            j += 1

        end

    end

    h2 = diag(Vgt ./ (Vgt + Vet))
     
    #### calculate loglik 

    n = size(Y', 1)
    ZKZt = Z*K*Z'
    R = zeros(n,n) + I; R = kron(R, Vet);
    V = R + kron(ZKZt, Vgt); V = V + 0.001*I; VL = Matrix(cholesky(Positive, V).L);
    XB = X'*Bt';
    #LL = MvNormal(vec(XB), Symmetric(V));
    #loglik = logpdf(LL, vec(Y));
    LL = AltMvNormal(Val(:L), vec(XB), VL);
    loglik = logpdf(LL, vec(Y));

    ### Write ouput in R for pretty compatible format
    
    m11 =  Dict(
			:Vg => Vgt,
			:Ve => Vet,
			:h2 => h2,
	                :loglik => loglik)

    return(m11)

end 



