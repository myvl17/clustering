# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:39:31 2016

@author: myvl17
"""
import time
import numpy as np
from numpy import linalg as LA
from scipy import random#, linalg, special

import CutClusClass

npts = 1000;
dim = 5;
k = npts;
t = np.zeros([100,1])
for it in range(0,100):
    points = random.rand(npts,dim)
    #points = np.array([[1,0],[4,4],[0,2],[3,4]])
    fldis = np.zeros((npts,k))
    for i in range(0,npts):
        for j in range(0,k):
            fldis[i,j] = LA.norm(points[i,:]-points[j,:])#LA.norm(fourpoints[i,:]-fourpoints[j,:])
        
    sfldis=np.sort(fldis,axis=1)
    #type(tenpoints)
    temp = time.time()
    ds,inds = CutClusClass.knnCPU(points,points,k)
    t[it] = time.time() - temp
    
print(['Average time elapsed = ',np.mean(t),' secs'])
print(LA.norm(sfldis-ds))