'''
@author: raghuvir
#LidDrivenCavity - BGK and TRT Model
use sudo apt-get install python-matplotlib
sudo apt-get install python-vtk?

'''
#from numba import vectorize
import numpy as np
#import numpy as np
import math
import numexpr as ne
import matplotlib
from math import nan
matplotlib.use('Agg')
from matplotlib import pyplot
#from VTKWrapper import saveToVTK
import os
from timeit import default_timer as timer
import numba as nmb
from functools import lru_cache as cache

import pycuda.driver as cuda 
import pycuda.autoinit
from pycuda.compiler import SourceModule 
from sklearn.metrics import r2_score
tstart = timer()

GPU0 = cuda.Device(0)
maxTperB = GPU0.MAX_THREADS_PER_BLOCK

#os.system("taskset -p 0xff %d" % os.getpid())
# print('number of cores detected is '); print(ne.detect_number_of_threads())
#ne.set_num_threads(ne.detect_number_of_threads())

# Plot Settings
Pinterval = 3000000# iterations to print next output
SavePlot = True #save plots
SaveVTK = False # save files in vtk format
project = 'ldc'  #for naming output files 
OutputFolder = './output'
CurrentFolder = os.getcwd()

# Lattice Parameters
maxIt = 3000001 # time iterations
Re    = 5000.0 # Reynolds number 100 400 1000 3200 5000 7500 10000
RT = 'SRT' # choose relaxation time method : SRT, TRT or MRT
turb = 0 ; # 1 = smagorinsky model. 0 = no turbulence. 

count = 0 # to calculate number of iterations to stop for convergence 
#Number of cells
xsize = 32*10 # must be multiple of 32 for GPU processing
ysize = 32*10# must be multiple of 32 for GPU processing
xsize_max, ysize_max = xsize-1, ysize-1 # highest index in each direction
q = 9 # d2Q9

uLB = 0.08 # velocity in lattice units
#since uLB is fixed if dx is changed, dt is changed proportionally i.e., acoustic scaling

print('the value of uLB is ', uLB) # <0.1 for accuracy and < 0.4 stability
print('xsize value is ', xsize)
nuLB = uLB*ysize/Re #viscosity coefficient

omega = 2.0 / (6.*nuLB+1)
tauS = np.zeros((xsize,ysize)) # relaxation times for turbulent flows - SGS
tauS[:,:] = 1.0/omega
tau = np.empty((xsize,ysize)) 
tau[:,:] = 1.0/omega
print('Re chosen  is ', Re)
print('RT chosen is ', RT)
if turb ==1 :
    regime = 'Turbulent' #options : Laminar, Turbulent
    print('Turbulence is on')
else :
    regime = 'Laminar' 
    print('Turbulence is off')

BC = 'EB-NEBB ' # options : BB(half way link based), NEBB (wet node)
print('the value of tau(/Dt) is ', 1/omega) # some BCs need this to be 1

tau_check = 0.5 +(1/8.0)*uLB # minimum value for stability
#print('If it is SRT, It should not be less than', tau_check, '. Closer to 1, better.')

#TRT
omegap = omega  
delTRT = 1.0/3.5 # choose this based on the error that you want to focus on
omegam = 1.0/(0.5 + ( delTRT/( (1/omegap)-0.5 )) )


#MRT - relaxation time vector)
omega_nu = omega # from shear viscosity
omega_e =  1.0 # stokes hypothesis - bulk viscosity is zero
omega_eps , omega_q = 1.2,1.2 #0.71, 0.83 # randomly chosen
omega_vec = [0.0, omega_e, omega_eps, 0.0, omega_q, 0.0, omega_q, omega_nu, omega_nu]
#omega_vec = ones((q)); omega_vec[:] = omega
omega_diag = np.diag(omega_vec)

if RT == 'SRT':
    print(' the value of omega is ', omega)
elif RT == 'TRT':
    print('the value of deltaTRT is ', delTRT)
    print('omegap, omegam :', round(omegap,4),' , ', round(omegam,4))
elif RT == 'MRT':
    print('omega omegap omegam omega_nu omega_e omega_eps omega_q') # For reference
    print(omega, omegap, omegam, omega_nu, omega_e, omega_eps, omega_q)

## Plot Setup

#creating output directory
if not os.path.isdir(OutputFolder):
    try:
        os.makedirs(OutputFolder)
    except OSError:
        pass

#Grid Setup
Xgrid = np.arange(0,xsize,dtype='float64')
Ygrid = np.arange(0,ysize,dtype='float64')
Zgrid = np.arange(0, 1,dtype='float64')
grid = Xgrid, Ygrid, Zgrid

Xcoord = np.array([Xgrid,]*ysize).transpose() #row number corresponds to x value
Ycoord = np.array([Ygrid,]*xsize) #column number corresponds to y value
velZ = np.zeros((xsize,ysize,1)) # velocity in Z direction is zero

# axis for velocity plots
YNorm = np.arange(ysize,0,-1,dtype='float64')/ysize # y starts as 0 from top lid
XNorm = np.arange(0,xsize,1,dtype='float64')/xsize # x starts as 0 from left wall
# Ghia Data for Re 100 to Re 10000
GhiaData = np.genfromtxt('GhiaData.csv', delimiter = ",")[6:23,1:]
GhiaDataV = np.genfromtxt('GhiaData.csv',delimiter = ",")[25:39,2:9]
X_GhiaHor = GhiaData[:,9]
Y_GhiaVer = GhiaData[:,0]

Re_dict = {100:1, 400:2, 1000:3, 3200:4, 5000:5, 7500:6, 10000:7} # column numbers in csv file for Re
Ux_GhiaVer = GhiaData[:,Re_dict[Re]]
Uy_GhiaHor = GhiaData[:,Re_dict[Re]+9]
# Positions of vortices
X_Vor = GhiaDataV[0:7,Re_dict[Re]-1]
X_Vor = X_Vor[X_Vor!=0]
Y_Vor = GhiaDataV[7:14,Re_dict[Re]-1]
Y_Vor = Y_Vor[Y_Vor!=0]

NormErr_column = []; time_column = [] # initializing arrays to track LBM and Ghia differences
temp = np.array(Y_GhiaVer*ysize_max).astype(int) # LBM coordinates close to ghia's values
LBMy = temp[1:] #avoiding zero values at wall

# Lattice Constants
c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
cx = np.array([0,1,0,-1,0,1,-1,-1,1])
cy = np.array([0,0,1,0,-1,1,1,-1,-1])
cxb = cx.reshape(9,1,1)
cyb = cy.reshape(9,1,1)
# Lattice Weights
t = 1.0/36. * np.ones(q)
t[1:5] = 1.0/9.0
t[0] = 4.0/9.0 


tb = t.reshape(9,1,1) # reshaping for broadcasting 

# reverse index for bounceback condition
bounce = [0,3,4,1,2,7,8,5,6]

#indexed arrays for stencil sides
LeftStencil   = np.arange(q)[np.asarray([temp[0] <  0 for temp in c])]
CentVStencil  = np.arange(q)[np.asarray([temp[0] == 0 for temp in c])]
RightStencil  = np.arange(q)[np.asarray([temp[0] >  0 for temp in c])]
TopStencil    = np.arange(q)[np.asarray([temp[1] >  0 for temp in c])]
CentHStencil  = np.arange(q)[np.asarray([temp[1] == 0 for temp in c])]
BotStencil    = np.arange(q)[np.asarray([temp[1] <  0 for temp in c])]

#MRT GRAM SCHMIDT VELOCITY MATRIX
M_GS = np.ones((9,9))
M_GS[0,:] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
M_GS[1,:] = [-4, -1, -1, -1, -1, 2, 2, 2, 2]
M_GS[2,:] = [4, -2, -2, -2, -2, 1, 1, 1, 1]
M_GS[3,:] = [0, 1, 0, -1, 0, 1, -1, -1, 1]
M_GS[4,:] = [0, -2, 0, 2, 0, 1, -1, -1, 1]
M_GS[5,:] = [0, 0, 1, 0, -1, 1, 1, -1, -1]
M_GS[6,:] = [0, 0, -2, 0, 2, 1, 1, -1, -1]
M_GS[7,:] = [0, 1, -1, 1, -1, 0, 0, 0, 0]
M_GS[8,:] = [0, 0, 0, 0, 0, 1, -1, 1, -1]
#MRT GS VEL MATRIX INVERSE
M_GS_INV = np.ones((9,9))
M_GS_INV[0,:] = [1.0/9, -1.0/9, 1.0/9, 0, 0, 0, 0, 0, 0]
M_GS_INV[1,:] = [1.0/9, -1.0/36, -1.0/18, 1.0/6, -1.0/6, 0, 0, 1.0/4, 0]
M_GS_INV[2,:] = [1.0/9, -1.0/36, -1.0/18, 0, 0, 1.0/6, -1.0/6, -1.0/4, 0]
M_GS_INV[3,:] = [1.0/9, -1.0/36, -1.0/18, -1.0/6, 1.0/6, 0, 0, 1.0/4, 0]
M_GS_INV[4,:] = [1.0/9, -1.0/36, -1.0/18, 0, 0, -1.0/6, 1.0/6, -1.0/4, 0]
M_GS_INV[5,:] = [1.0/9, 1.0/18, 1.0/36, 1.0/6, 1.0/12, 1.0/6, 1.0/12, 0, 1.0/4]
M_GS_INV[6,:] = [1.0/9, 1.0/18, 1.0/36, -1.0/6, -1.0/12, 1.0/6, 1.0/12, 0, -1.0/4]
M_GS_INV[7,:] = [1.0/9, 1.0/18, 1.0/36, -1.0/6, -1.0/12, -1.0/6, -1.0/12, 0, 1.0/4]
M_GS_INV[8,:] = [1.0/9, 1.0/18, 1.0/36, 1.0/6, 1.0/12, -1.0/6, -1.0/12, 0, -1.0/4]

# OMEGA, M_GS, M_GS_INV are defined again in GPU functions. Any changes here should be copied there.

reg_val = 0 ; reg_val_past = 0# regression value

#compute density
#@nmb.jit('double[:,:](double[:,:,:])')
def sumf(fin):
    return np.sum(fin, axis =0)
    #return ne.evaluate('sum(fin, axis =0)') # using numexpr for multithreading
#compute equilibrium distribution

cu = np.empty((q,xsize,ysize))
#u = np.empty((2,xsize,ysize))
#u = np.zeros((2,xsize,ysize))
u = np.zeros((2,xsize,ysize), dtype=float).astype(np.float32)
u1 = np.empty((2,xsize,ysize))
fplus =np.empty((q,xsize,ysize))
fminus = np.empty((q,xsize,ysize))
feplus = np.empty((q,xsize,ysize))
feminus = np.empty((q,xsize,ysize))
#ftemp = np.empty((q,xsize,ysize))
#fin = np.zeros((q,xsize,ysize))
fin = np.zeros((9,xsize,ysize), dtype=float).astype(np.float32)
ftemp = np.zeros((9,xsize,ysize), dtype=float).astype(np.float32)
fin1 = np.empty((q,xsize,ysize))
rho1 = np.empty((xsize,ysize))
fpost = np.copy(fin)
u[:,:,:]=0.0
u_past = u.copy()
#rho = np.ones((xsize,ysize))
rho = np.ones((xsize,ysize), dtype=float).astype(np.float32)
rhob = rho[np.newaxis,:,:]
usqr = np.empty((xsize,ysize))
usqrb = np.empty((q,xsize,ysize))

#@nmb.jit(nmb.f8[:,:,:](nmb.f8[:,:], nmb.f8[:,:,:]))
#@cache(maxsize=None)
def equ(rho,u):
    #cu   = dot(c, u.transpose(1, 0, 2))
    # peeling loop to increase speed
    cu[0] = (c[0,0]*u[0] + c[0,1]*u[1])
    cu[1] = (c[1,0]*u[0] + c[1,1]*u[1])
    cu[2] = (c[2,0]*u[0] + c[2,1]*u[1])
    cu[3] = (c[3,0]*u[0] + c[3,1]*u[1])
    cu[4] = (c[4,0]*u[0] + c[4,1]*u[1])
    cu[5] = (c[5,0]*u[0] + c[5,1]*u[1])
    cu[6] = (c[6,0]*u[0] + c[6,1]*u[1])
    cu[7] = (c[7,0]*u[0] + c[7,1]*u[1])
    cu[8] = (c[8,0]*u[0] + c[8,1]*u[1])
       
    usqr = (u[0]*u[0]+u[1]*u[1])

    #feq = np.empty((q, xsize, ysize))
    feq = np.zeros((9,xsize,ysize), dtype=float).astype(np.float32)
    for i in range(q):
        feq[i, :, :] = rho*t[i]*(1. + 3.0*cu[i] + 9*0.5*cu[i]*cu[i] - 3.0*0.5*usqr)
    return feq

LeftWall = np.fromfunction(lambda x,y:x==0,(xsize,ysize))
RightWall = np.fromfunction(lambda x,y:x==xsize_max,(xsize,ysize))
BottomWall = np.fromfunction(lambda x,y:y==ysize_max,(xsize,ysize))


wall = np.logical_or(np.logical_or(LeftWall, RightWall), BottomWall)

# velocity initial/boundary conditions
InitVel = np.zeros((2,xsize,ysize))
InitVel[0,:,0] = uLB
#u[:] = InitVel[:]
# initial distributions
feq = equ(rho, InitVel)

np.copyto(fin,feq)
np.copyto(ftemp,feq)
np.copyto(fpost,feq)

# interactive figure mode
if (SavePlot):
    pyplot.ioff()
    f = pyplot.figure(figsize=(30,16))

os.chdir(OutputFolder)

u[0,:,:] = u[0,:,:].transpose()
u[1,:,:] = u[1,:,:].transpose()
for k in range(9):
    fin[k,:,:] = fin[k,:,:].transpose()
    feq[k,:,:] = feq[k,:,:].transpose()
    ftemp[k,:,:] = ftemp[k,:,:].transpose()
rho = rho.transpose()

t_h = np.zeros((9), dtype=float).astype(np.float32)
c_h = np.zeros((18), dtype=float).astype(np.int32)
t_h[0]=4.0/9.0 ;t_h[1]=1.0/9.0;t_h[2]=1.0/9.0;t_h[3]=1.0/9.0;t_h[4]=1.0/9.0;t_h[5]=1.0/36.0;t_h[6]=1.0/36.0;t_h[7]=1.0/36.0;t_h[8]=1.0/36.0;
c_h[0]=0;c_h[1]=0;c_h[2]=1;c_h[3]=0;c_h[4]=0;c_h[5]=1;c_h[6]=-1;c_h[7]=0;c_h[8]=0;
c_h[9]=-1;c_h[10]=1;c_h[11]=1;c_h[12]=-1;c_h[13]=1;c_h[14]=-1;c_h[15]=-1;c_h[16]=1;c_h[17]=-1;

#GPU parameters
warp         = 32
blockDimX   = min(xsize,warp)
blockDimY   = min(ysize,warp)
gridDimX    = (xsize+blockDimX-1)//blockDimX # just quotient
gridDimY    = (ysize+blockDimY-1)//blockDimY

fin = fin.astype(np.float32)
fpost = fpost.astype(np.float32)
c = c.astype(np.int32)
t = t.astype(np.float32)
tauS = tauS.astype(np.float32)
# uLB = uLB.astype(np.float32)
# omegap = omegap.astype(np.float32); omegam = omegam.astype(np.float32)
# omega_e = omega_e.astype(np.float32); omega_eps = omega_eps.astype(np.float32)
# omega_q = omega_q.astype(np.float32); omega_nu = omega_nu.astype(np.float32)

# Allocate memory on GPU
fin_g = cuda.mem_alloc(fin.size * fin.dtype.itemsize)
#ftemp_g = cuda.mem_alloc(fin.nbytes)
feq_g = cuda.mem_alloc(fin.size * fin.dtype.itemsize)
rho_g = cuda.mem_alloc(rho.size * rho.dtype.itemsize)
taus_g = cuda.mem_alloc(tauS.size * tauS.dtype.itemsize)
u_g = cuda.mem_alloc(u.nbytes)
c_g = cuda.mem_alloc(c.nbytes)
t_g = cuda.mem_alloc(t.nbytes)

#fpost_g = cuda.mem_alloc(fin.size * fin.dtype.itemsize)
#Pinning memory for faster transfers between cpu and gpu
#fin_pin = cuda.register_host_memory(fin)
#this didnt seem to make difference, so using fin only for transfers

cuda.memcpy_htod(fin_g,fin)
#cuda.memcpy_htod(ftemp_g,fin)
cuda.memcpy_htod(feq_g,fin)
cuda.memcpy_htod(rho_g,rho)
cuda.memcpy_htod(taus_g, tauS)
cuda.memcpy_htod(u_g,u)
#cuda.memcpy_htod(fpost_g,fin)
# cuda.memcpy_htod(c_g,c)
# cuda.memcpy_htod(t_g,t)

# start = cuda.Event();
# stop = cuda.Event();

# mod2 = SourceModule(
# '''
# __device__ __constant__ float t_c[9] ;
# __device__ __constant__ float c_c[18] ;
# '''    
# )
# t_c,_ = mod2.get_global("t_c")
# c_c,_ = mod2.get_global("c_c")
# 
# cuda.memcpy_htod(t_c,t_h)
# cuda.memcpy_htod(c_c,c_h)

mod = SourceModule ("""

    __device__ __constant__ float t_c[9] ;
    __device__ __constant__ int c_c[18] ;
    
    __global__ void funRT(float* fin_g, float* feq_g, float* rho_g, float* u_g, float* taus_g){
        int x     = threadIdx.x + blockIdx.x * blockDim.x;
        int y     = threadIdx.y + blockIdx.y * blockDim.y;
        int xsize    = blockDim.x * gridDim.x; 
        int ysize    = blockDim.y * gridDim.y; 
        int d = xsize * ysize;
        int i = x + y * xsize;
        int k = 0;
        //float uLB = 0.08; reducing register variables
        
        float omega = %s;
        //int turb = 0; float product =0;// float product2 =0;float Qmf =0; decreasing register variables
        //float Csbulk = 0.16; decreasing register variables
        //float nuLB = (1.0/(3.0*omega)) - (1.0/6.0); int xsize_max = xsize-1;int ysize_max = ysize-1;
        //Only for Turbulence float visc_inv=0; float Zplus=0; float Cs2 = 0.0; float tau = 1.0/omega ;// float Cs=0;
        
        //float usqr = 0.0;
        float cu = 0.0;
        float rho_l=rho_g[i];
        //float t_c[9] ;
        //int c_c[18] ;
        float fin_l[9]; float u_l[2] ; // local fin to avoid multiple global memory access
        
        fin_l[0]=fin_g[0*d+i];fin_l[1]=fin_g[1*d+i];fin_l[2]=fin_g[2*d+i];fin_l[3]=fin_g[3*d+i];fin_l[4]=fin_g[4*d+i];
        fin_l[5]=fin_g[5*d+i];fin_l[6]=fin_g[6*d+i];fin_l[7]=fin_g[7*d+i];fin_l[8]=fin_g[8*d+i];
        
        rho_l = fin_l[0]+fin_l[1]+fin_l[2]+fin_l[3]+fin_l[4]+fin_l[5]+fin_l[6]+fin_l[7]+fin_l[8];
        
        u_l[0] = (c_c[0]*fin_l[0]+c_c[2]*fin_l[1]+c_c[4]*fin_l[2]+c_c[6]*fin_l[3]+c_c[8]*fin_l[4]+c_c[10]*fin_l[5]+c_c[12]*fin_l[6]+c_c[14]*fin_l[7]+c_c[16]*fin_l[8])/rho_l;
        u_l[1] = (c_c[1]*fin_l[0]+c_c[3]*fin_l[1]+c_c[5]*fin_l[2]+c_c[7]*fin_l[3]+c_c[9]*fin_l[4]+c_c[11]*fin_l[5]+c_c[13]*fin_l[6]+c_c[15]*fin_l[7]+c_c[17]*fin_l[8])/rho_l;
     
        // BCs left wall, right wall, bottom wall and top wall
        if ( x == 0 or x == xsize-1 or y == ysize-1){
            u_l[0] =0;
            u_l[1] =0;    
        }           
        if (y==0){ 
            rho_l = fin_l[0]+fin_l[1]+fin_l[3]+2*(fin_l[2]+fin_l[5]+fin_l[6]);
            u_l[0]= 0.08 ; // uLB; 
            u_l[1] =0;
        }
        //usqr = u_l[0]*u_l[0] + u_l[1]*u_l[1];
        u_g[0*d+i] = u_l[0]; u_g[1*d+i] = u_l[1];
        rho_g[i] = rho_l;       
        __syncthreads();
        for (k=0;k<9;k=k+1){ 
            cu = (c_c[k*2]*u_l[0] + c_c[k*2+1]*u_l[1]) ;      
            feq_g[k*d+i] = rho_l*t_c[k]*(1. + 3.0*cu + 9*0.5*cu*cu - 3.0*0.5*(u_l[0]*u_l[0] + u_l[1]*u_l[1]));
              
            if ( (x+c_c[k*2]>=0) and (x+c_c[k*2]<xsize) and (y-c_c[k*2+1]>=0) and (y-c_c[k*2+1]<ysize) ){
                fin_g[k*d+x+c_c[k*2]+(y-c_c[k*2+1])*xsize] = fin_l[k] - omega*(fin_l[k]-feq_g[k*d+i]) ; // j-c because -ve y axis  
            }
        }

        
        if (x ==0 ){
            fin_g[1*d+i] = feq_g[1*d+i] - feq_g[3*d+i] + fin_g[3*d+i];
            fin_g[5*d+i] = feq_g[5*d+i] - feq_g[7*d+i] + fin_g[7*d+i];
            fin_g[8*d+i] = feq_g[8*d+i] - feq_g[6*d+i] + fin_g[6*d+i];
        }else if(x==xsize-1){
            fin_g[3*d+i] = -feq_g[1*d+i] + feq_g[3*d+i] + fin_g[1*d+i];
            fin_g[6*d+i] = -feq_g[8*d+i] + feq_g[6*d+i] + fin_g[8*d+i];
            fin_g[7*d+i] = -feq_g[5*d+i] + feq_g[7*d+i] + fin_g[5*d+i];
        }
                 
        if (y==ysize-1){   
            fin_g[2*d+i] = -feq_g[4*d+i] + feq_g[2*d+i] + fin_g[4*d+i];
            fin_g[5*d+i] = -feq_g[7*d+i] + feq_g[5*d+i] + fin_g[7*d+i];
            fin_g[6*d+i] = -feq_g[8*d+i] + feq_g[6*d+i] + fin_g[8*d+i];
        }else if (y==0){     
            fin_g[4*d+i] = -feq_g[2*d+i] + feq_g[4*d+i] + fin_g[2*d+i];
            fin_g[7*d+i] = -feq_g[5*d+i] + feq_g[7*d+i] + fin_g[5*d+i];
            fin_g[8*d+i] = -feq_g[6*d+i] + feq_g[8*d+i] + fin_g[6*d+i];
        } 
        
          
                    
    }
""" % (omega)
)

#    funRT = funRT % (uLB, omega, turb)   

    
#mod         = SourceModule(funRT+funBC)
funRT        = mod.get_function("funRT")
#funBC        = mod.get_function("funBC")
t_c,_ = mod.get_global("t_c")
c_c,_ = mod.get_global("c_c")
# 
cuda.memcpy_htod(t_c,t_h)
cuda.memcpy_htod(c_c,c_h)


# Time Loop
for It in range(maxIt):
  
    #cuda.memcpy_htod(fin_g, fin)
    #cuda.memcpy_htod(fpost_g,fpost)

#    start.record() 
    #start.synchronize()
    
    funRT(fin_g, feq_g, rho_g, u_g, taus_g, block=(blockDimX,blockDimY,1), grid=(gridDimX,gridDimY)) 

    #funBC(ftemp_g, feq_g, fin_g,block=(blockDimX,blockDimY,1), grid=(gridDimX,gridDimY))     
#     
#     stop.record()
#     stop.synchronize()
    #cudaDeviceSynchronize();

#    time_gpu = start.time_till(stop)
    
    if( (It%Pinterval == 0) & (SaveVTK | SavePlot)) :
        
#         print('time spent in gpu (in millisecs) at this iteration is ',time_gpu)
#         print('So, Gpu bandwidth ( in GBPS) is ', 35*8*0.000001/(time_gpu))
#         
        u_past = u.copy()
        #cuda.memcpy_dtoh(fin,fin_g)
        cuda.memcpy_dtoh(rho,rho_g)
        cuda.memcpy_dtoh(u,u_g)
        rho = rho.transpose()
        u[0,:,:] = u[0,:,:].transpose()
        u[1,:,:] = u[1,:,:].transpose()   
         
        print ('current iteration :', It)
        #print (np.mean(u[0,:,0])/uLB)
        Usquare = u[0,:,:]**2 + u[1,:,:]**2
        
        Usquare = Usquare/(uLB**2)
        BCoffset = int(xsize/40)
        # replacing all boundaries with nan to get location of vortices
        Usquare[0:BCoffset,:] = nan ; Usquare[:,0:BCoffset] = nan
        Usquare[xsize_max-BCoffset:xsize,:] = nan;Usquare[:,ysize_max-BCoffset:ysize] = nan
        Loc1 = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)
        #print(Loc1)
        #print(Usquare[Loc1[0], Loc1[1]])
        # finding other vortices
        Usquare[Loc1[0]-BCoffset:Loc1[0]+BCoffset,Loc1[1]-BCoffset:Loc1[1]+BCoffset] = nan
        Loc2 = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)
        #print(Loc2)
        #print(Usquare[Loc2[0], Loc2[1]])        
              
        if (SavePlot):
            
            f.clear()
            subplot1 = pyplot.subplot2grid((2,15),(0,0), colspan=4, rowspan=1)
            subplot2 = pyplot.subplot2grid((2,15),(0,5), colspan=4, rowspan=1)
            subplot3 = pyplot.subplot2grid((2,15),(0,10), colspan=5, rowspan=1)
            subplot4 = pyplot.subplot2grid((2,15),(1,0), colspan=10, rowspan=1)
            #subplot5 = pyplot.subplot2grid((2,15),(1,5), colspan=4, rowspan=1)
            #subplot6 = pyplot.subplot2grid((2,15),(1,10), colspan=5, rowspan=1)          
            
            matplotlib.rcParams.update({'font.size': 15})
            Ux = u[0,int(xsize/2),:]/uLB 
            subplot1.plot(Ux, YNorm, label="LBM")
            subplot1.plot(Ux_GhiaVer, Y_GhiaVer, 'g*' , label="Ghia")
            subplot1.set_title('Ux on middle column', fontsize = 20, y =1.02)
            subplot1.legend(loc = 'center right')
            subplot1.set_xlabel('Ux', fontsize = 20);subplot1.set_ylabel('Y-position', fontsize = 20)
            
            Uy = u[1,:,int(ysize/2)]/uLB
            subplot2.plot(XNorm, Uy, label="LBM")
            subplot2.plot(X_GhiaHor,Uy_GhiaHor, 'g*', label='Ghia')
            subplot2.set_title('Uy on middle row', fontsize = 20, y=1.02)
            subplot2.legend(loc = 'upper right')
            subplot2.set_xlabel('X-position', fontsize = 20);subplot2.set_ylabel('Uy', fontsize = 20)
            
          
            #subplot1.imshow(sqrt(u[0]**2+u[1]**2).transpose(), pyplot.set_cmap('jet') , vmin = 0, vmax = 0.02)
            color1 = (np.sqrt(u[0,:,:]**2+u[1,:,:]**2)/uLB ).transpose()
            strm = subplot3.streamplot(XNorm,YNorm,(u[0,:,:]).transpose(),(u[1,:,:]).transpose(), color =color1,cmap=pyplot.cm.jet)#,norm=matplotlib.colors.Normalize(vmin=0,vmax=1)) 
            cbar = pyplot.colorbar(strm.lines, ax = subplot3)
            subplot3.plot(Loc1[0]/xsize,(ysize_max-Loc1[1])/ysize,'ro', label='Vortex1')
            subplot3.plot(Loc2[0]/xsize,(ysize_max-Loc2[1])/ysize,'mo', label='Vortex2')
            subplot3.plot(X_Vor, Y_Vor,'ks', label='Ghia')
            subplot3.set_title('Velocity Streamlines - LBM', fontsize = 20, y =1.02)
            subplot3.margins(0.005) #subplot3.axis('tight')
            subplot3.set_xlabel('X-position', fontsize = 20);subplot3.set_ylabel('Y-position', fontsize = 20)
            
            #print((YNorm == array(Y_GhiaVer*ysize_max).astype(int)))
            Ux_temp = u[0,int(xsize/2),LBMy]/uLB
            Ux_temp = Ux_temp + 0.0001 # to avoid zero division errors
            #print(np.fliplr(np.atleast_2d(Ux_temp))[0])
            #print(Ux_GhiaVer[:-1])
            #temp = abs( ( abs(Ux_GhiaVer[:-1]) - abs(np.fliplr(np.atleast_2d(Ux_temp))[0])) / ( len(Ux_temp)*np.maximum(abs(Ux_GhiaVer[:-1]),abs(np.fliplr(np.atleast_2d(Ux_temp))[0]) )) )  #rsquare estimate
            reg_val_past = reg_val 
            reg_val = r2_score(Ux_GhiaVer[:-1],np.fliplr(np.atleast_2d(Ux_temp))[0])
            NormErr_column.append(reg_val)
            print('current regression value is ' + str(reg_val))
            print('current mean velocity value is ' + str(np.mean(u)/uLB) )
            #print ((1 - sumf(abs(fliplr(atleast_2d(Ux_GhiaVer[:-1]))[0] - Ux_temp)/(len(Ux_temp)*abs(Ux_GhiaVer[:-1])) )))
            #print(NormErr_column)
            #NormErr_column.append(1 - sumf(square(fliplr(atleast_2d(Ux_GhiaVer[:-1]))[0] - Ux_temp))/(len(Ux_temp)*var(Ux_GhiaVer[:-1])) ) #rsquare estimate
            #NormErr_column.append(1 - (1/len(LBMy))*sumf( square( (fliplr(atleast_2d(Ux_GhiaVer[:-1]))[0] - Ux_temp)/(Ux_GhiaVer[:-1])) )  ) #rsquare estimate
            #NormErr_column.append(abs(1 - math.log10((1/len(LBMy))*sumf( square( (fliplr(atleast_2d(Ux_GhiaVer[:-1]))[0] - Ux_temp)/(Ux_GhiaVer[:-1])) ))  )) #rsquare estimate

            #NormErr_column.append(1 - sumf(square(fliplr(atleast_2d(Ux_GhiaVer))[0] - Ux_temp))/sumf(square(Ux_GhiaVer)) ) #rsquare estimate

            time_column.append(It)
            subplot4.plot(time_column,NormErr_column,)
            subplot4.set_title('Regression value - Ux_MiddleColumn' )  
            subplot4.set_xlabel('time iteration', fontsize = 20);subplot4.set_ylabel('Regression value', fontsize = 20)
            pyplot.figtext(0.5,0.3,'Current Regression value is')
            pyplot.figtext(0.5,0.28,str(round(NormErr_column[-1],4)))
            
            pyplot.figtext(0.65,0.45,"Square dots in above figure represent vortex locations from Ghia data")
            pyplot.figtext(0.65,0.43,"Circular dots represent vortex locations of current simulation")
            pyplot.figtext(0.65,0.35,'LBM parameters: '+RT, fontsize=20)
            pyplot.figtext(0.65,0.31,'Grid size: '+str(xsize)+'*'+str(ysize))
            pyplot.figtext(0.65,0.29,'Re: '+str(Re)+'    '+'BoundaryCondition: '+BC)
            pyplot.figtext(0.65,0.27,'Lid velocity in LB units: '+str(uLB)+'    dx* and dt* hardcoded as 1')
            pyplot.figtext(0.65,0.25,'tau - related to dynamic viscosity: '+str(round(1.0/omega,3)))
            if (RT=='SRT'):
                data_out = 'omega: '+str(round(omega,2))
                pyplot.figtext(0.65,0.23,data_out)
            elif (RT=='TRT'):
                data_out = 'omega_plus, omega_minus, delta: '+str(round(omegap,3))+' , '+str(round(omegam,3))+' , '+str(round(delTRT,3)) 
                pyplot.figtext(0.65,0.23,data_out)
            elif (RT=='MRT'):
                data_out = 'omega_nu, omega_e, omega_eps, omega_q: '
                pyplot.figtext(0.65,0.23,data_out)
                data_out = str(round(omega_nu,3))+' , '+str(round(omega_e,3))+' , '+str(round(omega_eps,3))+' , '+str(round(omega_q,3))
                pyplot.figtext(0.65,0.21,data_out)
            if( regime=='Turbulent'):
                data_out = 'Smagorinsky constant, Cs = '+ str(Cs[int(xsize/2),0])+' at wall to '+str(Csbulk)+' at bulk'
                pyplot.figtext(0.65,0.17,data_out)
                data_out = 'Mean relaxation time, tau+tau_turbulent, is  '+ str(mean(tauS))
                pyplot.figtext(0.65,0.15,data_out)
            
            f.suptitle('Lid Driven Cavity - Re'+str(int(Re))+' '+regime+' '+RT+' '+BC+' '+str(xsize)+'*'+str(ysize), fontsize = 30, y =1.04)
            pyplot.savefig(project + "_" + str(int(It/Pinterval)).zfill(5) + ".png",bbox_inches = 'tight', pad_inches = 0.4)

        if ( SaveVTK ):
            # convert 2d data to 3d arrays
            Vel = reshape(u, (2, xsize, ysize, 1))
            Vel3D = (Vel[0, :, :, :], Vel[1, :, :, :], velZ)   
            Rho3D = reshape(rho, (xsize, ysize, 1))
            index = str(int(It/Pinterval)).zfill(5)
            saveToVTK(Vel3D, Rho3D, project, index, grid)
        tend = timer()

        print ( 'time elapsed is ', (tend-tstart), 'seconds' )
       
        if (abs(np.mean(u)-np.mean(u_past))/uLB < 0.00000001):
        #if (abs(reg_val-reg_val_past) < 0.0001):
            
            count = count + 1
            if ( count>5):
                print('breaking out of loop because of convergence')
                break
          
        if (It == maxIt-1):
            print('max iterations reached. More needed for convergence.')
            

os.chdir(CurrentFolder)
tend = timer()

print ( 'TOTAL time elapsed is ', (tend-tstart), 'seconds' )
print ( 'MLUPS : ', xsize*ysize*maxIt*0.000001/(tend-tstart) )
