
##### Hmat Julia 

using RCall, LinearAlgebra, InvertedIndices
using EMMREML, PedigreeBase, NamedArrays


##### Calculate NRM from numeric pedigree
#### All missing must be 0

function computeA(Progeny, Sire, Dam)

  n = length(Progeny);
  A = zeros(n,n) + I;

for i in 1:n

    if (Sire[i] == 0 && Dam[i] != 0)
    	for j in 1:(i-1)
        A[i, j] = A[j, i] = 0.5 * (A[j, Dam[i]]) 
        end

    elseif (Sire[i] != 0 && Dam[i] == 0)
        for j in 1:(i-1)
        A[i, j] = A[j, i] = 0.5 * (A[j, Sire[i]]) 
        end

    elseif (Sire[i] != 0 && Dam[i] != 0)
        for j in 1:(i-1)
        A[i, j] = A[j, i] = 0.5 * (A[j, Sire[i]] + A[j, Dam[i]]) 
        end

        A[i, i] = A[i, i] + 0.5 * (A[Sire[i], Dam[i]])
    end

end

  return(A)

end


function Hmat(tau, omega, M; input="input.Rdata", networks=false) 

## Amat & input are paths to their Rdata locations

        @rput input;
        R"load(input)";
        
        ### pedfilePath must be full Path where ped is located. File must have no column names
        #pedlist,idtable = read_ped(pedfilePath);
        #A = get_nrm(pedlist, sorthere=false);
        #@rput pedlist;

        #R"xx <- unlist(pedlist[2:length(pedlist)])";
        #R"xx <- xx[order(xx)]";
        #R"idA <- names(xx)[which(xx == 1) : length(xx)]";
        #@rget idA;
        
        @rget ped; @rget pednames; @rget pednum; @rget linenames; # Linenames ensure only Ped that has data is used for analysis...
        idA = pednames; @rput idA;
        
        A = computeA(Int64.(ped[:,1]), Int64.(ped[:,2]), Int64.(ped[:, 3]))
        A = NamedArray(A, (idA, idA));
  
        ##### Only use pedigree individuals that has data
        A = A[linenames, linenames];
        idA = linenames; @rput idA; ## idA becomes linenames - a subset of only ped with phenos

        #@rget M;
        R"if(exists('MB')){M = MB}";
        R"idG <- as.character(rownames(M))"; @rget idG;
        R"idP <- unique(as.character(setdiff(linenames, idG)))"; @rget idP;
        
        #R"index = which(is.na(match(idH, idG)))"; @rget index;
        #A11 = A[index, index];
        #A12 = A[index, Not(index)];
        #A21 = A[Not(index), index];
        #A22 = A[Not(index), Not(index)];
  
        A11 = A[idP, idP];
        A12 = A[idP, idG];
        A21 = A12';
        A22 = A[idG, idG];
  
        sorted = vcat(idP, idG);
        A = A[sorted, sorted];
        #### Do wted G or not ######

     #   if wtedG == true
     #      @rget D; M = Matrix(M); G = GRMwted(M, D); @rput G;
     #   else 
     #      M = Matrix(M); G =  GRM(M); #@rput G;
     #   end
  
  
        if networks == true
           M = Matrix(M); G = DOM(M);
        else 
           M = Matrix(M); G =  GRM(M);
        end

        G = NamedArray(G, (idG, idG));

        #G22 = G[idH[Not(index)], idH[Not(index)]]
        G22 = G;

        A22inv = inv(cholesky(Positive, A22)); A22inv = NamedArray(A22inv, (idG, idG));
        G22inv = inv(cholesky(Positive, G22)); G22inv = NamedArray(G22inv, (idG, idG));
        H22 = inv(cholesky(Positive, (tau * G22inv) + ((1 - omega) * A22inv))); H22 = NamedArray(H22, (idG, idG));
        H11 = A12 * A22inv * (H22 - A22) * A22inv * A21;
        H12 = A12 * A22inv * (H22 - A22);
        H21 = (H22 - A22) * A22inv * A21;
        H22 = (H22 - A22);
        nom1 = vcat(names(A11,1), names(A21, 1)); nom2 = vcat(names(A11,2), names(A12, 2));
        H = A + hcat(vcat(H11, H21), vcat(H12, H22));
        H = 0.90*H + 0.1*I;
        H = NamedArray(H, (nom1, nom2));
        H = H[linenames, linenames];

        return(Dict(:H => H,:names => linenames))

end


#### compute Hmat differently ...

function Hmat2(tau, omega, M; input="input.Rdata", epis=true) 

## Amat & input are paths to their Rdata locations

        @rput input;
        R"load(input)";
        
        
        @rget ped; @rget pednames; @rget pednum; @rget linenames; # Linenames ensure only Ped that has data is used for analysis...
        idA = pednames; @rput idA;
        
        A = computeA(Int64.(ped[:,1]), Int64.(ped[:,2]), Int64.(ped[:, 3]))
        A = NamedArray(A, (idA, idA));
  
        ##### Only use pedigree individuals that has data
        A = A[linenames, linenames];
        idA = linenames; @rput idA; ## idA becomes linenames - a subset of only ped with phenos

        #@rget M;
        R"if(exists('MB')){M = MB}";
        R"idG <- as.character(rownames(M))"; @rget idG;
        R"idP <- unique(as.character(setdiff(linenames, idG)))"; @rget idP;
        
  
        A11 = A[idP, idP];
        A12 = A[idP, idG];
        A21 = A12';
        A22 = A[idG, idG];
  
        sorted = vcat(idP, idG);
        A = A[sorted, sorted];
        #### Do wted G or not ######
  
  
        if epis == true
           M = Matrix(M); G = M * M';
        else 
           M = Matrix(M); G =  GRM(M);
        end

        G = NamedArray(G, (idG, idG));

        G22 = G;

        A22inv = inv(cholesky(Positive, A22)); A22inv = NamedArray(A22inv, (idG, idG));
        G22inv = inv(cholesky(Positive, G22)); G22inv = NamedArray(G22inv, (idG, idG));
        H22 = inv(cholesky(Positive, (tau * G22inv) + ((1 - omega) * A22inv))); H22 = NamedArray(H22, (idG, idG));
        H11 = A12 * A22inv * (H22 - A22) * A22inv * A21;
        H12 = A12 * A22inv * (H22 - A22);
        H21 = (H22 - A22) * A22inv * A21;
        H22 = (H22 - A22);
        nom1 = vcat(names(A11,1), names(A21, 1)); nom2 = vcat(names(A11,2), names(A12, 2));
        H = A + hcat(vcat(H11, H21), vcat(H12, H22));
        H = 0.90*H + 0.1*I;
        H = NamedArray(H, (nom1, nom2));
        H = H[linenames, linenames];

        return(Dict(:H => H,:names => linenames))

end

#### Another alternative for Hmat computation


function Hmat_alt(tau, omega, M; input="input.Rdata", networks=false) 

## Amat & input are paths to their Rdata locations

        @rput input;
        R"load(input)";
        
        
        @rget ped; @rget pednames; @rget pednum; @rget linenames; # Linenames ensure only Ped that has data is used for analysis...
        idA = pednames; @rput idA;
        
        A = computeA(Int64.(ped[:,1]), Int64.(ped[:,2]), Int64.(ped[:, 3]))
        A = NamedArray(A, (idA, idA));
  
        ##### Only use pedigree individuals that has data
        A = A[linenames, linenames];
        idA = linenames; @rput idA; ## idA becomes linenames - a subset of only ped with phenos

        #@rget M;
        R"if(exists('MB')){M = MB}";
        R"idG <- as.character(rownames(M))"; @rget idG;
        R"idP <- unique(as.character(setdiff(linenames, idG)))"; @rget idP;
        
  
        A11 = A[idP, idP];
        A12 = A[idP, idG];
        A21 = A12';
        A22 = A[idG, idG];
  
        sorted = vcat(idP, idG);
        A = A[sorted, sorted];
        #### Do wted G or not ######
  
  
        if networks == true
           M = Matrix(M); G = DOM(M);
        else 
           M = Matrix(M); G =  GRM(M);
        end

        G = NamedArray(G, (idG, idG));

        A22inv = inv(cholesky(Positive, A22)); A22inv = NamedArray(A22inv, (idG, idG));
        
        # create H components (4 quadrants)
        H11 = A11 + (A12 * A22inv * (G - A22) * A22inv * A21);
        H12 = A12 * A22inv * G;
        H21 = H12';
        H22 =  G;
  
        # put H together by peices
        H = hcat(vcat(H11, H21), vcat(H12, H22));
        nom1 = vcat(names(A11,1), names(A21, 1)); nom2 = vcat(names(A11,2), names(A12, 2));
        H = 0.95*H + 0.05*I;
        H = NamedArray(H, (nom1, nom2));
        H = H[linenames, linenames];

        return(Dict(:H => H,:names => linenames))

end



function pedMat(input) 


        @rput input;
        R"load(input)";

        
        @rget ped; @rget pednames; @rget pednum; @rget linenames; # Linenames ensure only Ped that has data is used for analysis...
        idA = pednames; @rput idA;
 
        A = computeA(Int64.(ped[:,1]), Int64.(ped[:,2]), Int64.(ped[:, 3]));
        A = NamedArray(A, (idA, idA));
  
        ##### Only use pedigree individuals that has data
        A = A[linenames, linenames];

        return(A)

end



function HmatNet(tau, omega, M, N; input="input.Rdata") 

## make sure rows of N & M match same individuals...

        @rput input;
        R"load(input)";
        
        
        @rget ped; @rget pednames; @rget pednum; @rget linenames; # Linenames ensure only Ped that has data is used for analysis...
        idA = pednames; @rput idA;
        
        A = computeA(Int64.(ped[:,1]), Int64.(ped[:,2]), Int64.(ped[:, 3]))
        A = NamedArray(A, (idA, idA));
  
        ##### Only use pedigree individuals that has data
        A = A[linenames, linenames];
        idA = linenames; @rput idA; ## idA becomes linenames - a subset of only ped with phenos

        R"if(exists('MB')){M = MB}";
        R"idG <- as.character(rownames(M))"; @rget idG;
        R"idP <- unique(as.character(setdiff(linenames, idG)))"; @rget idP;
        
  
        A11 = A[idP, idP];
        A12 = A[idP, idG];
        A21 = A12';
        A22 = A[idG, idG];
  
        sorted = vcat(idP, idG);
        A = A[sorted, sorted];

        #### Do wted G or not ######
  
  
        N = Matrix(N); KN = DOM(N);

        M = Matrix(M); KM =  GRM(M);

        G = KM + KN; #sum networks and GRM


        G = NamedArray(G, (idG, idG));

        G22 = G;

        A22inv = inv(cholesky(Positive, A22)); A22inv = NamedArray(A22inv, (idG, idG));
        G22inv = inv(cholesky(Positive, G22)); G22inv = NamedArray(G22inv, (idG, idG));
        H22 = inv(cholesky(Positive, (tau * G22inv) + ((1 - omega) * A22inv))); H22 = NamedArray(H22, (idG, idG));
        H11 = A12 * A22inv * (H22 - A22) * A22inv * A21;
        H12 = A12 * A22inv * (H22 - A22);
        H21 = (H22 - A22) * A22inv * A21;
        H22 = (H22 - A22);
        nom1 = vcat(names(A11,1), names(A21, 1)); nom2 = vcat(names(A11,2), names(A12, 2));
        H = A + hcat(vcat(H11, H21), vcat(H12, H22));
        H = 0.90*H + 0.1*I;
        H = NamedArray(H, (nom1, nom2));
        H = H[linenames, linenames];

        return(Dict(:H => H,:names => linenames))

end


#### Run Epistasis + main effect in ssGBLUP framework

function HmatEpis(tau, omega, M, N; input="input.Rdata") 

## make sure rows of N & M match same individuals...

        @rput input;
        R"load(input)";
        
        
        @rget ped; @rget pednames; @rget pednum; @rget linenames; # Linenames ensure only Ped that has data is used for analysis...
        idA = pednames; @rput idA;
        
        A = computeA(Int64.(ped[:,1]), Int64.(ped[:,2]), Int64.(ped[:, 3]))
        A = NamedArray(A, (idA, idA));
  
        ##### Only use pedigree individuals that has data
        A = A[linenames, linenames];
        idA = linenames; @rput idA; ## idA becomes linenames - a subset of only ped with phenos

        R"if(exists('MB')){M = MB}";
        R"idG <- as.character(rownames(M))"; @rget idG;
        R"idP <- unique(as.character(setdiff(linenames, idG)))"; @rget idP;
        
  
        A11 = A[idP, idP];
        A12 = A[idP, idG];
        A21 = A12';
        A22 = A[idG, idG];
  
        sorted = vcat(idP, idG);
        A = A[sorted, sorted];

        #### Do wted G or not ######
  
        pieces = Int64.(round(size(M,2)/10));

        N = Matrix(N); KN = N * N';

        M = Matrix(M); KM =  GRMVariter(M, pieces); #snp specific scaling

        G = KM + KN; #sum networks and GRM


        G = NamedArray(G, (idG, idG));

        G22 = G;

        A22inv = inv(cholesky(Positive, A22)); A22inv = NamedArray(A22inv, (idG, idG));
        G22inv = inv(cholesky(Positive, G22)); G22inv = NamedArray(G22inv, (idG, idG));
        H22 = inv(cholesky(Positive, (tau * G22inv) + ((1 - omega) * A22inv))); H22 = NamedArray(H22, (idG, idG));
        H11 = A12 * A22inv * (H22 - A22) * A22inv * A21;
        H12 = A12 * A22inv * (H22 - A22);
        H21 = (H22 - A22) * A22inv * A21;
        H22 = (H22 - A22);
        nom1 = vcat(names(A11,1), names(A21, 1)); nom2 = vcat(names(A11,2), names(A12, 2));
        H = A + hcat(vcat(H11, H21), vcat(H12, H22));
        H = 0.90*H + 0.1*I;
        H = NamedArray(H, (nom1, nom2));
        H = H[linenames, linenames];

        return(Dict(:H => H,:names => linenames))

end



