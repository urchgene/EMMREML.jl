
##### Hmat Julia 

using RCall, LinearAlgebra, InvertedIndices
using EMMREML, PedigreeBase, NamedArrays

function Hmat(tau, omega, pedfilePath; input="input.Rdata", wtedG=false) 

## Amat & input are paths to their Rdata locations

        @rput input;
        R"load(input)";
        
        ### pedfilePath must be full Path where ped is located. File must have no column names
        pedlist,idtable = read_ped(pedfilePath);
        A = get_nrm(pedlist, sorthere=false);
        @rput pedlist;

        R"xx <- unlist(pedlist[2:length(pedlist)])";
        R"xx <- xx[order(xx)]";
        R"idA <- names(xx)[which(xx == 1) : length(xx)]";
        @rget idA;


        @rget M;
        R"idG <- as.character(rownames(M))"; @rget idG;
        R"idH <- unique(c(idG, idA))"; R"idH <- rev(idH)";
        @rget idH; 
        #R"A <- as.matrix(A); A <- A[idH, idH];"; @rget A;
        A = NamedArray(A, (idA, idA)); A = A[idH, idH];
        R"index = which(is.na(match(idH, idG)))"; @rget index;
        A11 = A[index, index];
        A12 = A[index, Not(index)];
        A21 = A[Not(index), index];
        A22 = A[Not(index), Not(index)];

        #### Do wted G or not ######

        if wtedG == true
           @rget D; G = GRMwted(M, D); @rput G;
        else 
           G =  GRM(M); @rput G;
        end

        #R"colnames(G) <- rownames(G) <- idG";
        G = NamedArray(G, (idG, idG));
        #R"G22 <- G[idH[-(index)], idH[-(index)]]"; @rget G22;
        G22 = G[idH[Not(index)], idH[Not(index)]]

        A22inv = pinv(A22);
        G22inv = pinv(G22);
        H22 = pinv((tau * G22inv) + ((1 - omega) * A22inv));
        H11 = A12 * A22inv * (H22 - A22) * A22inv * A21;
        H12 = A12 * A22inv * (H22 - A22);
        H21 = (H22 - A22) * A22inv * A21;
        H22 = (H22 - A22);
        H = A + hcat(vcat(H11, H21), vcat(H12, H22));
        H = NamedArray(H, (idH, idH));

        return(Dict(:H => H,:names => idH))

end
