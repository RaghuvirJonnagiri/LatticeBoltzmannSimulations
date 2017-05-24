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
cdef double_t omega = 0.0
cdef double_t uLatBo = 0.0
cdef double_t nuLB = 0.0
#cdef ndarray[double_t, ndim=1]t = 1.0/36. * np.ones(q)
cdef double_t[:] t = 1.0/36. * np.ones(q)
t[1:5] = 1.0/9.0 ;t[0] = 4.0/9.0 
# cdef int_t[:] bounce = np.array([0,3,4,1,2,7,8,5,6])
# cdef int_t[:] ynonpos = np.array([2,4,5,6,7,8])


# LeftStencil   = np.arange(q)[np.asarray([temp[0] <  0 for temp in c])]
# CentVStencil  = np.arange(q)[np.asarray([temp[0] == 0 for temp in c])]
# RightStencil  = np.arange(q)[np.asarray([temp[0] >  0 for temp in c])]
# TopStencil    = np.arange(q)[np.asarray([temp[1] >  0 for temp in c])]
# CentHStencil  = np.arange(q)[np.asarray([temp[1] == 0 for temp in c])]
# BotStencil    = np.arange(q)[np.asarray([temp[1] <  0 for temp in c])]
# np.seterr(all='raise')


# @cython.boundscheck(False)
# @cython.wraparound(False)
cpdef sumf(ndarray[double_t, ndim=3] fin):
    return np.sum(fin, axis =0)
           
# @cython.boundscheck(False)
# @cython.wraparound(False)    
        
cpdef set_omega(double_t uLB, int_t Re, int_t ysize):
    global uLatBo
    uLatBo = uLB
    nuLB = uLB*ysize/Re #viscosity coefficient
    global omega
    omega = 2.0 / (6.*nuLB+1)
    
cpdef allfunc(ndarray[double_t, ndim=2] rho, ndarray[double_t, ndim=3] u,ndarray[double_t, ndim=3] fin,ndarray[double_t, ndim=3] feq ):
    cdef int i
    cdef int j
    cdef int k
    cdef int temp
    cdef int dim0, dim1, dim2
    cdef int xsize_max 
    cdef int ysize_max 
    cdef double_t[:,:,:] cu = np.empty((q,rho.shape[0],rho.shape[1]))
    cdef double_t[:,:] usqr = np.empty((rho.shape[0],rho.shape[1])) 
    cdef double_t[:,:,:] fpost = np.empty((q,rho.shape[0],rho.shape[1]))
    cdef double_t[:,:,:] fin1 = np.zeros((q,rho.shape[0],rho.shape[1]))
#     cdef double_t[:,:,:] fin2 = np.zeros((q,rho.shape[0],rho.shape[1]))
    
    #fin1 = fin.copy()
    #np.copyto(fin1,fin)
    dim0 = fin.shape[0]
    dim1 = fin.shape[1]
    dim2 = fin.shape[2]
    xsize_max = dim1-1
    ysize_max = dim2-1
    rho = np.sum(fin, axis =0)
    #print(openmp.omp_get_max_threads())
    #if (1>0):
    with nogil, parallel(num_threads=4):    
        for i in prange(dim1):    #schedule=static or dynamic?
            for j in range(dim2):
                  
                u[0,i,j] = (c[0,0]*fin[0,i,j]+c[1,0]*fin[1,i,j]+c[2,0]*fin[2,i,j]+c[3,0]*fin[3,i,j]+c[4,0]*fin[4,i,j]+c[5,0]*fin[5,i,j]+c[6,0]*fin[6,i,j]+c[7,0]*fin[7,i,j]+c[8,0]*fin[8,i,j])/rho[i,j]
                u[1,i,j] = (c[0,1]*fin[0,i,j]+c[1,1]*fin[1,i,j]+c[2,1]*fin[2,i,j]+c[3,1]*fin[3,i,j]+c[4,1]*fin[4,i,j]+c[5,1]*fin[5,i,j]+c[6,1]*fin[6,i,j]+c[7,1]*fin[7,i,j]+c[8,1]*fin[8,i,j])/rho[i,j]
  
                if i == 0 or i == (dim1-1) or j == (dim2-1):
                    u[0,i,j] =0;u[1,i,j] =0    
                      
                if j == 0: 
                    rho[i,0] = fin[0,i,0]+fin[1,i,0]+fin[3,i,0]+2*(fin[2,i,0]+fin[5,i,0]+fin[6,i,0])
                    u[0,i,0]=uLatBo ; u[1,i,0] =0
  
                      
                usqr[i,j] = u[0,i,j]*u[0,i,j] + u[1,i,j]*u[1,i,j]
                   
                for k in range(dim0): 
                    cu[k,i,j] = (c[k,0]*u[0,i,j] + c[k,1]*u[1,i,j])       
                    feq[k,i,j] = rho[i,j]*t[k]*(1. + 3.0*cu[k,i,j] + 9*0.5*cu[k,i,j]*cu[k,i,j] - 3.0*0.5*usqr[i,j])
                    
                    #fpost[k,i,j] = fin[k,i,j] - omega*(fin[k,i,j]-feq[k,i,j])
#                      
                    if ( (i+c[k,0]>=0) and (i+c[k,0]<dim1) and (j-c[k,1]>=0) and (j-c[k,1]<dim2) ):
                        fin1[k,i+c[k,0],j-c[k,1]] = fin[k,i,j] - omega*(fin[k,i,j]-feq[k,i,j])  # j-c because -ve y axis  
                if (i ==1 ):
                    fin1[1,0,j] = feq[1,0,j]
                    fin1[5,0,j] = feq[5,0,j]
                    fin1[8,0,j] = feq[8,0,j]
                     
                elif (i==dim1-1):
                    fin1[3,dim1-1,j] = -feq[1,dim1-1,j] + feq[3,dim1-1,j] + fin1[1,dim1-1,j]
                    fin1[6,dim1-1,j] = -feq[8,dim1-1,j] + feq[6,dim1-1,j] + fin1[8,dim1-1,j]
                    fin1[7,dim1-1,j] = -feq[5,dim1-1,j] + feq[7,dim1-1,j] + fin1[5,dim1-1,j]
 
                 
                #print(str(k)+' '+str(i)+' '+str(j)+' '+str(np.max(fin)))              
#  
            fin1[2,i,dim2-1] = -feq[4,i,dim2-1] + feq[2,i,dim2-1] + fin1[4,i,dim2-1]
            fin1[5,i,dim2-1] = -feq[7,i,dim2-1] + feq[5,i,dim2-1] + fin1[7,i,dim2-1]
            fin1[6,i,dim2-1] = -feq[8,i,dim2-1] + feq[6,i,dim2-1] + fin1[8,i,dim2-1]
   
             
            fin1[4,i,0] = -feq[2,i,0] + feq[4,i,0] + fin1[2,i,0]
            fin1[7,i,0] = -feq[5,i,0] + feq[7,i,0] + fin1[5,i,0]
            fin1[8,i,0] = -feq[6,i,0] + feq[8,i,0] + fin1[6,i,0]

          
#     ftemp = np.pad(ftemp,((0,0),(1,1),(1,1)),'constant',constant_values=0.9)
#     fpost = np.pad(fpost,((0,0),(1,1),(1,1)),'constant',constant_values=0.9)
# 
#     with nogil, parallel(num_threads=3):
#         for i in prange(1,dim1):
#             for j in range(1,dim2):
#                 for k in range(1,dim0):
#                     fin[k,i,j] = fin1[k,i,j]
    #fin[:,:,:] = fin1[:,:,:]
#     with nogil, parallel(num_threads=4):
#         for i in prange(1,dim1):
#             for j in range(1,dim2):
#                 
#                 fin[0, i, j] = fpost[0,i, j]
#                       
#                 fin[1, i,j]      = fpost[1, i-1, j]
#                 fin[2, i, j-1]   = fpost[2, i, j]
#                 fin[3, i-1, j]   = fpost[3, i, j]
#                 fin[4, i, j]     = fpost[4, i, j-1]
#                      
#                 fin[5, i,   j-1] = fpost[5, i-1, j]
#                 fin[6, i-1, j-1] = fpost[6, i, j]
#                 fin[7, i-1, j]   = fpost[7, i, j-1]
#                 fin[8, i,j]      = fpost[8, i-1, j-1]
#                     
#                 if i ==1:
#                     fin[0,0,j]   = fpost[0, 0, j]
#                     fin[2,0,j-1] = fpost[2, 0, j]
#                     fin[4,0,j]   = fpost[4, 0, j-1]
#                     fin[1,0,j] = feq[1,0,j];fin[5,0,j] = feq[5,0,j];fin[8,0,j] = feq[8,0,j]#- feq[LeftStencil, 0, :] + fin[LeftStencil, 0, :]
#      
#                 if j==1:
#                     fin[0,i,0]   = fpost[0, i, 0]
#                     fin[1,i,0]   = fpost[1, i-1, 0]
#                     fin[3,i-1,0] = fpost[3, i, 0]
#                          
#                 if i == dim1-1:
#                     fin[3,dim1-1,j] = -feq[1,dim1-1,j] + feq[3,dim1-1,j] + fin[1,dim1-1,j]
#                     fin[6,dim1-1,j] = -feq[8,dim1-1,j] + feq[6,dim1-1,j] + fin[8,dim1-1,j]
#                     fin[7,dim1-1,j] = -feq[5,dim1-1,j] + feq[7,dim1-1,j] + fin[5,dim1-1,j]
#        
#                 
#             fin[2,i,dim2-1] = -feq[4,i,dim2-1] + feq[2,i,dim2-1] + fin[4,i,dim2-1]
#             fin[5,i,dim2-1] = -feq[7,i,dim2-1] + feq[5,i,dim2-1] + fin[7,i,dim2-1]
#             fin[6,i,dim2-1] = -feq[8,i,dim2-1] + feq[6,i,dim2-1] + fin[8,i,dim2-1]
#          
#             fin[4,i,0] = -feq[2,i,0] + feq[4,i,0] + fin[2,i,0]
#             fin[7,i,0] = -feq[5,i,0] + feq[7,i,0] + fin[5,i,0]        
#             fin[8,i,0] = -feq[6,i,0] + feq[8,i,0] + fin[6,i,0]        
#         
#         fin[0,0,0] = fpost[0,0,0]
#         fin[2,0,dim2-1] = -feq[4,0,dim2-1] + feq[2,0,dim2-1] + fin[4,0,dim2-1]
#         fin[5,0,dim2-1] = -feq[7,0,dim2-1] + feq[5,0,dim2-1] + fin[7,0,dim2-1]
#         fin[6,0,dim2-1] = -feq[8,0,dim2-1] + feq[6,0,dim2-1] + fin[8,0,dim2-1]
#                  
#         fin[4,0,0] = -feq[2,0,0] + feq[4,0,0] + fin[2,0,0]
#         fin[7,0,0] = -feq[5,0,0] + feq[7,0,0] + fin[5,0,0]        
#         fin[8,0,0] = -feq[6,0,0] + feq[8,0,0] + fin[6,0,0]
#               
#         fin[1,0,0] = feq[1,0,0];fin[5,0,0] = feq[5,0,0];fin[8,0,0] = feq[8,0,0]#- feq[LeftStencil, 0, :] + fin[LeftStencil, 0, :]
#         fin[3,dim1-1,0] = -feq[1,dim1-1,0] + feq[3,dim1-1,0] + fin[1,dim1-1,0]
#         fin[6,dim1-1,0] = -feq[8,dim1-1,0] + feq[6,dim1-1,0] + fin[8,dim1-1,0]
#         fin[7,dim1-1,0] = -feq[5,dim1-1,0] + feq[7,dim1-1,0] + fin[5,dim1-1,0]

#     fin[0, :, :] = fpost[0, :, :]
#        
#     fin[1, 1:xsize_max,   :]     = fpost[1, 0:xsize_max-1,  :]
#     fin[2,   :,   0:ysize_max-1] = fpost[2,   :,    1:ysize_max]
#     fin[3, 0:xsize_max-1, :]     = fpost[3, 1:xsize_max,    :]
#     fin[4,   :,   1:ysize_max]   = fpost[4,   :,    0:ysize_max-1]
#    
#     fin[5, 1:xsize_max,   0:ysize_max-1] = fpost[5, 0:xsize_max-1, 1:ysize_max]
#     fin[6, 0:xsize_max-1, 0:ysize_max-1] = fpost[6, 1:xsize_max,   1:ysize_max]
#     fin[7, 0:xsize_max-1, 1:ysize_max]   = fpost[7, 1:xsize_max,   0:ysize_max-1]
#     fin[8, 1:xsize_max,   1:ysize_max]   = fpost[8, 0:xsize_max-1, 0:ysize_max-1]
# #   
#     
  
    

#     with nogil, parallel(num_threads=2):
#         for i in prange(dim1):
#                
#             fin[2,i,dim2-1] = -feq[4,i,dim2-1] + feq[2,i,dim2-1] + fin[4,i,dim2-1]
#             fin[5,i,dim2-1] = -feq[7,i,dim2-1] + feq[5,i,dim2-1] + fin[7,i,dim2-1]
#             fin[6,i,dim2-1] = -feq[8,i,dim2-1] + feq[6,i,dim2-1] + fin[8,i,dim2-1]
#                  
#             fin[4,i,0] = -feq[2,i,0] + feq[4,i,0] + fin[2,i,0]
#             fin[7,i,0] = -feq[5,i,0] + feq[7,i,0] + fin[5,i,0]        
#             fin[8,i,0] = -feq[6,i,0] + feq[8,i,0] + fin[6,i,0]
#               
#             fin[1,0,i] = feq[1,0,i];fin[5,0,i] = feq[5,0,i];fin[8,0,i] = feq[8,0,i]#- feq[LeftStencil, 0, :] + fin[LeftStencil, 0, :]
#             fin[3,dim1-1,i] = -feq[1,dim1-1,i] + feq[3,dim1-1,i] + fin[1,dim1-1,i]
#             fin[6,dim1-1,i] = -feq[8,dim1-1,i] + feq[6,dim1-1,i] + fin[8,dim1-1,i]
#             fin[7,dim1-1,i] = -feq[5,dim1-1,i] + feq[7,dim1-1,i] + fin[5,dim1-1,i]
       
#     
#     fin[RightStencil, 0, :] =  feq[RightStencil, 0, :] #- feq[LeftStencil, 0, :] + fin[LeftStencil, 0, :]
#     fin[LeftStencil, dim1-1, :] = - feq[RightStencil, dim1-1, :] + (feq[LeftStencil, dim1-1, :] + fin[RightStencil, dim1-1, :])
#     fin[TopStencil, :, dim2-1] = - feq[BotStencil, :, dim2-1] + (feq[TopStencil, :, dim2-1] + fin[BotStencil, :, dim2-1])
#     fin[BotStencil, :, 0] = - feq[TopStencil, :, 0] + (feq[BotStencil, :, 0] + fin[TopStencil, :, 0])

    #print(np.shape(fin))
    #print('2 - '+str(np.max(rho))) 
    #print('2 - '+str(np.argmax(fin-fin1)))         
    return rho, u, np.array(fin1), feq 
    #return rho, u, fin, feq 
    




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

cpdef copyfunc(ndarray[double_t, ndim=3] fout, ndarray[double_t, ndim=3] fin):
    cdef int i
    cdef int j 
    cdef int k 
    with nogil, parallel(num_threads=2):
        for i in prange(fin.shape[0]):
            for j in range(fin.shape[1]):
                for k in range(fin.shape[2]):
                    fout[i,j,k] = fin[i,j,k]
    return fout
                    