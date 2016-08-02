# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

#import time
#import pdb
#import math
import numpy as np
#from numpy import linalg as LA
#from scipy import random#, linalg, special

def knnCPU(R,Q,k):
    # Find the k nearest neighbors for each element of the query set Q among
    # the points in the reference set R

    # Q is Nxd numpy.array where N is the number of query points and d is the dimension
    # R is Mxd numpy.array where M is the number of reference points and d is the dimension
    # k is the number of nearest neighbors desired
    # maxMem is the number of GB of ram knnCPU can use

    # ds is Nxk and contains the distances to the k nearest neighbors.
    # inds is Nxk and contains the indices of the k nearest neighbors.
    # Author: Tyrus Berry (in MatLab)
    # Translated to Python by Marilyn Vazquez
    
    #MARILYN: THE INDEXING STARTS AT 0 AND AXIS=0 IS DOWNWARD (row-wise)
    maxMem = 2
    M = np.shape(R)
    M = M[0]
    N = np.shape(Q)
    N = N[0]
   
    maxArray = (maxMem*2500)**2;#squaring a number
    #making floats into integers
    blockSize = int(np.floor(maxArray/float(M)))#division with floats
    blocks = int(np.floor(N/float(blockSize)))#division with floats
     
    ds = np.zeros((N,k))
    inds = np.zeros((N,k))
    
    #column-wise sum
    Nr = np.sum(R**2,axis=1)
    Nq = np.sum(Q**2,axis=1)
    
    for b in range(1,blocks+1):#blocks=0, this gets skipped
        #matrix multiplication  
        dtemp = -2*R.dot(Q[(b-1)*blockSize:b*blockSize,:].T)  
        #element-wise addition with column array                     
        dtemp = dtemp+Nr[:,np.newaxis]
        #element-wise addition with row array 
        dtemp = dtemp+Nq[(b-1)*blockSize:b*blockSize]
        dst = np.sort(dtemp,axis=0)
        indst = np.argsort(dtemp,axis=0)
        ds[(b-1)*blockSize:b*blockSize,:] = dst[0:k,:].T
        inds[(b-1)*blockSize:b*blockSize,:] = indst[0:k,:].T
        
    if blocks*blockSize < N:
        #matrix multiplication
        dtemp = -2*R.dot(Q[blocks*blockSize:N,:].T)
        #element-wise addition with column array 
        dtemp = dtemp+Nr[:,np.newaxis]
        #element-wise addition with row array
        dtemp = dtemp+Nq[blocks*blockSize:N]
        #pdb.set_trace()
        #Sorting row-wise
        dst = np.sort(dtemp,axis=0)
        indst = np.argsort(dtemp,axis=0)
        ds[blocks*blockSize:N,:] = dst[0:k,:].T
        inds[blocks*blockSize:N,:] = indst[0:k,:].T
        
    ds = np.sqrt(np.absolute(ds))# CHANGE TO BETTER FUNCTION?
    inds = inds.astype(int)
    return ds, inds
    #return inds
    


# -*- coding: utf-8 -*-

#def VBDM(x,k,k2,nvars,operator,adhocEpsilon):
#### Inputs
#    ### x       - N-by-n data set with N data points in R^n
#    ### k       - number of nearest neighbors to use
#    ### k2      - number of nearest neighbors to use to determine the "epsilon"
#    ###             parameter
#    ### nvars   - number of eigenfunctions/eigenvalues to compute
#    ### operator- 1 - Laplace-Beltrami operator, 2 - Kolmogorov backward operator 
#    ### dim     - intrinsic dimension of the manifold lying inside R^n
#    ### epsilon - optionally choose an arbitrary "global" epsilon
#    
#### Outputs
#    ### q       - Eigenfunctions of the generator/Laplacian
#    ### b       - Eigenvalues
#    ### epsilon - scale, derived from the k2 nearest neighbors
#    ### peqoversample - Invariant measure divided by the sampling measure
#    ### peq     - Invariant measure
#    ### qest    - Sampling measure
#
#    ### Theory requires c2 = 1/2 - 2*alpha + 2*dim*alpha + dim*beta/2 + beta < 0 
#    ### The resulting operator will have c1 = 2 - 2*alpha + dim*beta + 2*beta
#    ### Thus beta = (c1/2 - 1 + alpha)/(dim/2+1), since we want beta<0,
#    ### natural choices are beta=-1/2 or beta = -1/(dim/2+1)
#
#    N = np.shape(x) #number of points
#    N = N[0]
#    
#    d,inds = knnCPU(x,x,k)
#
#    ### Build ad hoc bandwidth function by autotuning epsilon for each pt.
#    
#    epss = 2**np.arange(-30,31,0.1)
#
#    rho0 = np.sqrt(np.mean(d[:,1:k2]**2,axis=1))
    
    ### Pre-kernel used with ad hoc bandwidth only for estimating dimension
    ### and sampling density
#    dt = d.^2./(repmat(rho0,1,k).*rho0(inds))
#    
#    ### Tune epsilon on the pre-kernel
#    dpreGlobal=zeros(1,length(epss))
#    for i=1:length(epss)
#        dpreGlobal(i) = sum(sum(exp(-dt./(2*epss(i)))))/(N*k)    
#    
#   
#    [maxval,maxind] = max(diff(log(dpreGlobal))./diff(log(epss)));
#    figure;semilogx(epss(2:end),2*diff(log(dpreGlobal))./diff(log(epss)));
#    %if (nargin < 6)
#    dim=2*maxval;
#    %end
    
#    ### Use ad hoc bandwidth function, rho0, to estimate the density
#    dt = exp(-dt./(2*epss(maxind)))/((2*pi*epss(maxind))^(dim/2));
#    dt = sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(dt'),N*k,1),N,N,N*k)';
#    dt = (dt+dt')/2;
#
#    qest = (sum(dt,2))./(N*rho0.^(dim)); 
#    
#    clear dt;
#    
#    if (operator == 1)
#        ### Laplace-Beltrami, c1 = 0
#        beta = -1/2;
#        alpha = -dim/4 + 1/2;
#    elseif (operator == 2)
#        ### Kolmogorov backward operator, c1 = 1
#        beta = -1/2;
#        alpha = -dim/4;  
#    elseif (operator == 0)
#        ### Standard Diffusion Maps normalization
#        beta = 0;
#        alpha = -1;
#    end
#
#    c1 = 2 - 2*alpha + dim*beta + 2*beta;
#    c2=.5-2*alpha+2*dim*alpha+dim*beta/2+beta;
#    
#    d = d.^2;
#
#    ### Define the true bandwidth function from the density estimate
#    rho = qest.^(beta);
#    rho = rho/mean(rho);
#
#    d = d./repmat((rho),1,k);  % divide row j by rho(j)
#    d = d./rho(inds);
#    
#    ### Tune epsilon for the final kernel
#    for i=1:length(epss)
#        s(i) = sum(sum(exp(-d./(4*epss(i))),2))/(N*k);
#    end
#    [~,maxind] = max(diff(log(s))./diff(log(epss)));
#    epsilon = epss(maxind);
#    
#    if (nargin == 6)
#        epsilon = adhocEpsilon;
#    end
#   
#    d = exp(-d./(4*epsilon));
#
#    d = sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(d'),N*k,1),N,N,N*k)';
#    clear inds;
#    
#    ###%% d(inds(i),i) = d(i)
#    ###%% takes ith row, inds(i,:) and ith row, d(i,:)
#    ###%% places them in the first row of d, d(inds(i,j),j) = d(i,j)
#
#    d = (d+d')/2;   ### symmetrize since this is the symmetric formulation
#
#    qest = full((sum(d,2)./(rho.^dim)));
#
#    Dinv1 = spdiags(qest.^(-alpha),0,N,N);
#
#    d = Dinv1*d*Dinv1; % the "right" normalization
#    
#    peqoversample = full((rho.^2).*(sum(d,2)));
#
#    Dinv2 = spdiags(peqoversample.^(-1/2),0,N,N);
#    
#    d = Dinv2*d*Dinv2 - spdiags(rho.^(-2)-1,0,N,N); ### "left" normalization
#
#    opts.maxiter = 200;
#
#%     [q,b] = eigs(d,nvars,1,opts);
#    [q,b] = eigs(d,nvars,0.999,opts);
#
#    b = (diag(b));
#    [~,perm] = sort(b,'descend');    
#    b = b(perm).^(1/epsilon);
#    b=diag(b);
#    q = q(:,perm);
#    
#    q = Dinv2*q;
#      
#    ### Normalize qest into a density by dividing by m0
#
#    qest = qest/(N*(4*pi*epsilon)^(dim/2));
#    peqoversample = peqoversample;
#    peq = qest.*peqoversample;                ### Invariant measure of the system
#    peq = peq./mean(peq./qest);     ### normalization factor
#
#    ### Normalize the eigenfunctions so their L^2 norm is 1
#    ### Note that the eigenfunctions are orthogonal with respect to
#    ### p_{eq}=qest^{c1} but sampled according to qest so we weight the
#    ### integrant by p_{eq}/qest = qest^{c1-1}.
# 
#    for i = 1:nvars
#        q(:,i) = q(:,i)/sqrt(mean(q(:,i).^2.*(peq./qest)));
#    end