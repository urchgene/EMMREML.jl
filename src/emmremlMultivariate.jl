
#################################################
### EMMREML-MULTIVARIATE for Julia
### Modified and translated by Uche Godfrey Okeke
### Suitable for Big data purposes
##################################################


using LinearAlgebra, PositiveFactorizations;
using RCall;
using Statistics;
#using ProgressMeter

#my_BLAS_set_num_threads(n) =  ccall((:openblas_set_num_threads64_, Base.libblas_name), Cvoid, (Int32,), n)
#my_BLAS_set_num_threads(20)

"""
Multi-trait single kernel VCOMP solver for EMMA algorithm. Very useful for 2-step procedures.
Assumes same fixed effect components for all traits...
No missing values allowed in Y trait matrix

Function call: emmremlMultivariate(Y, X, Z, K, linenames)

Y is dxn trait matrix for d traits and n records, 
X (qxn) and Z (lxn) are design matrices for (q) fixed and (l) individuals in random effetcs
K (lxl) is Known covariance matrix
linenames is a character vec for individuals in order of colnames of Z or col/row (names)of K. 

This is a fast solver with ability to handle a good chunk of data.
"""

function emmremlMultivariate(Y, X, Z, K, linenames)

## call with regular Y, X, K but transpose Z ###

	Z = Z'; X = X'; Y = Y'; tolpar = 1e-06; tolparinv = 1e-06;


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
        deltal = eigZKZt.values[l]
        return(ECM1(ytl, xtl, Vgt, Vet, Bt, deltal))
    end


    function Vgfunc(l, outfromECM1)
        Vgl = outfromECM1[l][:gtl] * outfromECM1[l][:gtl]'
        return((1/n) * (1 ./ eigZKZt.values[l]) * (Vgl + outfromECM1[l][:Sigmalt]))
    end


    function Vefunc(l, outfromECM1)
        etl = Yt[:, l] - Bt * Xt[:, l] - outfromECM1[l][:gtl]
        return((1/n) * (etl*etl' + outfromECM1[l][:Sigmalt]))
    end

    if (!ismissing(Y))
        N = size(K,1)
        KZt = K * Z'
        ZKZt = Z * KZt
        ZKZt  = ZKZt + 0.000001*I;
        #eigZKZt = eigen(ZKZt)
	eigZKZt = eigen(Positive, ZKZt); # Generalize so no problems occur
	### reverse eigen vectors and values so that highest is first like in R
	eigZKZt.vectors .= reverse(eigZKZt.vectors, dims=2)
	eigZKZt.values .= reverse(eigZKZt.values)
        n = size(ZKZt, 1)
        d = size(Y, 1)
        Yt = Y * eigZKZt.vectors
        Xt = X * eigZKZt.vectors
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


    HobsInv = inv(kron(ZKZt, Vgt) + kron(zeros(n,n) + 1.0*I, Vet) + tolparinv * zeros(n*d,n*d) + 1.0*I)
    ehat = vec(Y - Bt * X)
    HobsInve = HobsInv * ehat
    varvecG = kron(K, Vgt)
    gpred = varvecG * (kron(Z', zeros(d,d) + 1.0*I)) * HobsInve
    Gpred = reshape(gpred, d, n)
    uhat = hcat(linenames, Gpred')
    h2 = diag(Vgt ./ (Vgt + Vet))


    ### Write ouput in R for pretty compatible format
    
    m11 =  Dict(
			:Vg => Vgt,
			:Ve => Vet,
			:betahat => Bt,
			:uhat => uhat,
			:h2 => h2)

    return(m11)

end 



