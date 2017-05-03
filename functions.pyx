#cython: boundscheck=False, wraparound=False, nonecheck=False
from numpy cimport *
import numpy as np
import cython
from cython.parallel import *
cimport openmp
from libc.stdlib cimport malloc,free
#cdef ndarray[double_t, ndim=2] c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
cdef int_t[:,:] c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
cdef int_t q=9
#cdef ndarray[double_t, ndim=1]t = 1.0/36. * np.ones(q)
cdef double_t[:] t = 1.0/36. * np.ones(q)
t[1:5] = 1.0/9.0 ;t[0] = 4.0/9.0 


# @cython.boundscheck(False)
# @cython.wraparound(False)
cpdef sumf(ndarray[double_t, ndim=3] fin):
    return np.sum(fin, axis =0)
           
# @cython.boundscheck(False)
# @cython.wraparound(False)    

cpdef equ(ndarray[double_t, ndim=2] rho, ndarray[double_t, ndim=2] ux, ndarray[double_t,ndim=2] uy):
    #cu   = dot(c, u.transpose(1, 0, 2))
    # peeling loop to increase speed

    cdef double_t[:,:,:] cu = np.empty((q,rho.shape[0],rho.shape[1]))
    cdef int i
    cdef int j
    cdef int k
    dim0 = rho.shape[0]
    dim1 = rho.shape[1]
    
    with nogil, parallel(num_threads=2):
        for i in prange(dim0):    #schedule=static or dynamic?
            for j in range(dim1):
                cu[0,i,j] = (c[0,0]*ux[i,j] + c[0,1]*uy[i,j])
                cu[1,i,j] = (c[1,0]*ux[i,j] + c[1,1]*uy[i,j])
                cu[2,i,j] = (c[2,0]*ux[i,j] + c[2,1]*uy[i,j])
                cu[3,i,j] = (c[3,0]*ux[i,j] + c[3,1]*uy[i,j])
                cu[4,i,j] = (c[4,0]*ux[i,j] + c[4,1]*uy[i,j])
                cu[5,i,j] = (c[5,0]*ux[i,j] + c[5,1]*uy[i,j])
                cu[6,i,j] = (c[6,0]*ux[i,j] + c[6,1]*uy[i,j])
                cu[7,i,j] = (c[7,0]*ux[i,j] + c[7,1]*uy[i,j])
                cu[8,i,j] = (c[8,0]*ux[i,j] + c[8,1]*uy[i,j])
    
    
    cdef ndarray[double_t, ndim=2] usqr 
    usqr = (ux*ux+uy*uy)
    #usqr=ne.evaluate('sum(u**2,0)')
    cdef ndarray[double_t, ndim=3] feq
    feq = np.empty((q, dim0, dim1))
    
    with nogil, parallel(num_threads=2):
        for i in range(q):
            for j in prange(dim0): #schedule=static or dynamic?
                for k in range(dim1):        
                    feq[i, j, k] = rho[j,k]*t[i]*(1. + 3.0*cu[i,j,k] + 9*0.5*cu[i,j,k]*cu[i,j,k] - 3.0*0.5*usqr[j,k])
        
    
    return feq

cpdef ucprod(ndarray[int_t, ndim=2] c, ndarray[double_t, ndim=3] fin,ndarray[double_t, ndim=2] rho): 
    
    cdef int i
    cdef int j
    cdef int k
    dim0 = fin.shape[0]
    dim1 = fin.shape[1]
    dim2 = fin.shape[2]
    cdef ndarray[double_t, ndim=3] vel
    vel = np.empty((2, dim1, dim2))
    with nogil, parallel(num_threads=2):
        for i in prange(dim1):
            for j in range(dim2):
                    vel[0,i,j] = (c[0,0]*fin[0,i,j]+c[1,0]*fin[1,i,j]+c[2,0]*fin[2,i,j]+c[3,0]*fin[3,i,j]+c[4,0]*fin[4,i,j]+c[5,0]*fin[5,i,j]+c[6,0]*fin[6,i,j]+c[7,0]*fin[7,i,j]+c[8,0]*fin[8,i,j])/rho[i,j]
                    vel[1,i,j] = (c[0,1]*fin[0,i,j]+c[1,1]*fin[1,i,j]+c[2,1]*fin[2,i,j]+c[3,1]*fin[3,i,j]+c[4,1]*fin[4,i,j]+c[5,1]*fin[5,i,j]+c[6,1]*fin[6,i,j]+c[7,1]*fin[7,i,j]+c[8,1]*fin[8,i,j])/rho[i,j]

    return vel