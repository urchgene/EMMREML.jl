###### Julia implementation of VanRaden GRM

using LinearAlgebra
using PositiveFactorizations
using Statistics;

function GRM(M)

	M = Matrix(M);
	#M = 1.0 .+ Matrix(M); # change to 0,1,2
	P = mean(M, dims=1) ./2 ## freq
	varsum = 2 * sum(P .* (1.0 .- P))
	A = (M .- (2 .* P)) ;
	K = (A * A') .* (1/varsum);
	K = 0.95*K + 0.05*I
	return(K)

end


function GRMwted(M, D)

	M = Matrix(M);
	#M = 1.0 .+ Matrix(M); # change to 0,1,2
	P = mean(M, dims=1) ./2 ## freq
	varsum = 2 * sum(P .* (1.0 .- P))
	A = (M .- (2 .* P)) ;
	K = (A * D * A') .* (1/varsum);
	K = 0.99*K + 0.01*I
	return(K)

end

function GRMinv(M)

        M = Matrix(M);
        #M = 1.0 .+ Matrix(M);
        P = mean(M, dims=1) ./2
        varsum = 2 * sum(P .* (1.0 .- P))
        A = (M .- (2 .* P));
        K = (A * A') .* (1/varsum);
        K = 0.99*K + 0.01*I
	K = cholesky(Positive, K);
	K = inv(K);
        return(K)

end

function GRMwtedinv(M, D)

        M = Matrix(M);
        #M = 1.0 .+ Matrix(M);
        P = mean(M, dims=1) ./2
        varsum = 2 * sum(P .* (1.0 .- P))
        A = (M .- (2 .* P));
        K = (A * D * A') .* (1/varsum);
        K = 0.99*K + 0.01*I
	K = cholesky(Positive, K);
	K = inv(K);
        return(K)

end



### computation of large SNP data in pieces
function GRMiter(M, pieces)

       series = collect(1:size(M,2));
       xx = Iterators.partition(series, pieces) |> collect;

       f = 0.0;
       G = zeros(size(M,1), size(M,1));

       for i=1:length(xx)

       MM = Matrix(M[:, xx[i]])  ## extract column of iterator
       P = mean(MM, dims=1) ./2
       varsum = 2 * sum(P .* (1.0 .- P))
       f += varsum;
       A = (MM .- (2 .* P)) ;
       K = (A * A');
       G = G + K;
       end

       G = G .* (1/f);
       G = 0.99*G + 0.01*I;

       return(G)

end


### Calculate GRM weighted by SNP variances for large data
### computation of large SNP data in pieces
function GRMVariter(M, pieces)
	
       scale(X, d) = (X .- (2 .* d)) ./ sqrt.((size(X, 2) * 2) .* d .* (1 .- d))
	
       series = collect(1:size(M,2));
       xx = Iterators.partition(series, pieces) |> collect;

       f = 0.0;
       G = zeros(size(M,1), size(M,1));

       for i=1:length(xx)

       MM = Matrix(M[:, xx[i]])  ## extract column of iterator
       P = mean(MM, dims=1) ./2
       A = scale(MM, P);
       K = (A * A');
       G = G + K;
       end
	
       G = 0.99*G + 0.01*I;

       return(G)

end

#### Dominance + absence presence kernels

function DOM(M)
    
	# mean = 2P, var/scale = n * 2pq(1 - 2pq)
	scale(X, d) = (X .- (2 .* d .* (1 .- d))) ./ sqrt.(size(X, 2)  .*  2 .* d .* (1 .- d) .* (1  .- 2 .* d .* (1 .- d)))
	
	M = Matrix(M);
	P = mean(M, dims=1) ./2 ## freq
	A = scale(M, P);
	K = (A * A');
	K = 0.99*K + 0.01*I;
	return(K)

end

