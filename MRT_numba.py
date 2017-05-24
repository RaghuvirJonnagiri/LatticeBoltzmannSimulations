'''
@author: raghuvir
#LidDrivenCavity - BGK and TRT Model
use sudo apt-get install python-matplotlib
sudo apt-get install python-vtk?

'''
#@profile

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


tstart = timer()

#os.system("taskset -p 0xff %d" % os.getpid())

# print('number of cores detected is '); print(ne.detect_number_of_threads())
#ne.set_num_threads(ne.detect_number_of_threads())

# Plot Settings
Pinterval = 1000# iterations to print next output
SavePlot = True #save plots
SaveVTK = False # save files in vtk format
project = 'ldc'  #for naming output files 
OutputFolder = './output'
CurrentFolder = os.getcwd()

# Lattice Parameters
maxIt = 3000 # time iterations
Re    = 1000.0 # Reynolds number 100 400 1000 3200 5000 7500 10000

#Number of cells
xsize, ysize = 160, 160
xsize_max, ysize_max = xsize-1, ysize-1 # highest index in each direction
q = 9 # d2Q9

uLB = 0.08 # velocity in lattice units
#since uLB is fixed if dx is changed, dt is changed proportionally i.e., acoustic scaling

print('the value of uLB is ', uLB) # <0.1 for accuracy and < 0.4 stability
nuLB = uLB*ysize/Re #viscosity coefficient

omega = 2.0 / (6.*nuLB+1)
tauS = np.empty((xsize,ysize)) # relaxation times for turbulent flows - SGS
tauS[:,:] = 1.0/omega
tau = np.empty((xsize,ysize)) 
tau[:,:] = 1.0/omega
print('the value of tau(/Dt) is ', 1/omega) # some BCs need this to be 1
tau_check = 0.5 +(1/8.0)*uLB # minimum value for stability
#print('If it is SRT, It should not be less than', tau_check, '. Closer to 1, better.')

#TRT
omegap = omega  
delTRT = 1.0/3.5 # choose this based on the error that you want to focus on
omegam = 1.0/(0.5 + ( delTRT/( (1/omegap)-0.5 )) )

#MRT - relaxation time vector
omega_nu = omega # from shear viscosity
omega_e =  1.0 # stokes hypothesis - bulk viscosity is zero
omega_eps , omega_q = 1.0,1.2 #0.71, 0.83 # randomly chosen
omega_vec = [0.0, omega_e, omega_eps, 0.0, omega_q, 0.0, omega_q, omega_nu, omega_nu]
#omega_vec = ones((q)); omega_vec[:] = omega
omega_diag = np.diag(omega_vec)

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
#     Ux_GhiaVertAxis = GhiaRefUx_100()
#     Uy_GhiaVertAxis = GhiaRefUy_100()
#     Ux_GhiaVertAxis = GhiaRefUx_400()
#     Uy_GhiaVertAxis = GhiaRefUy_400()
#     Ux_GhiaVertAxis = GhiaRefUx_1000()
#     Uy_GhiaVertAxis = GhiaRefUy_1000()
#     Ux_GhiaVertAxis = GhiaRefUx_3200()
#     Uy_GhiaVertAxis = GhiaRefUy_3200()
#     Ux_GhiaVertAxis = GhiaRefUx_5000()
#     Uy_GhiaVertAxis = GhiaRefUy_5000()
#     Ux_GhiaVertAxis = GhiaRefUx_7500()
#     Uy_GhiaVertAxis = GhiaRefUy_7500()
#     Ux_GhiaVertAxis = GhiaRefUx_10000()
#     Uy_GhiaVertAxis = GhiaRefUy_10000()


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

#compute density
#@nmb.jit('double[:,:](double[:,:,:])')
def sumf(fin):
    return np.sum(fin, axis =0)
    #return ne.evaluate('sum(fin, axis =0)') # using numexpr for multithreading
#compute equilibrium distribution

cu = np.empty((q,xsize,ysize))
u = np.empty((2,xsize,ysize))
u1 = np.empty((2,xsize,ysize))
fplus =np.empty((q,xsize,ysize))
fminus = np.empty((q,xsize,ysize))
feplus = np.empty((q,xsize,ysize))
feminus = np.empty((q,xsize,ysize))
ftemp = np.empty((q,xsize,ysize))
fin = np.zeros((q,xsize,ysize))
fin1 = np.empty((q,xsize,ysize))
rho1 = np.empty((xsize,ysize))
fpost = np.empty((q,xsize,ysize))


rho = np.ones((xsize,ysize))
rhob = rho[np.newaxis,:,:]
usqr = np.empty((xsize,ysize))
usqrb = np.empty((q,xsize,ysize))

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

    feq = np.empty((q, xsize, ysize))
#     for i in range(q):
#         feq[i, :, :] = rho*t[i]*(1. + 3.0*cu[i] + 9*0.5*cu[i]*cu[i] - 3.0*0.5*usqr)
#     return feq

    
    rhob = rho[np.newaxis,:,:]
    usqrb = usqr[np.newaxis,:,:]
    feq = ne.evaluate('rhob*tb*( 1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usqrb)')
    return feq
    

#     feq[0, :, :] = rho*t[0]*(1. + 3.0*cu[0] + 9*0.5*cu[0]*cu[0] - 3.0*0.5*usqr)
#     feq[1, :, :] = rho*t[1]*(1. + 3.0*cu[1] + 9*0.5*cu[1]*cu[1] - 3.0*0.5*usqr)
#     feq[2, :, :] = rho*t[2]*(1. + 3.0*cu[2] + 9*0.5*cu[2]*cu[2] - 3.0*0.5*usqr)
#     feq[3, :, :] = rho*t[3]*(1. + 3.0*cu[3] + 9*0.5*cu[3]*cu[3] - 3.0*0.5*usqr)
#     feq[4, :, :] = rho*t[4]*(1. + 3.0*cu[4] + 9*0.5*cu[4]*cu[4] - 3.0*0.5*usqr)
#     feq[5, :, :] = rho*t[5]*(1. + 3.0*cu[5] + 9*0.5*cu[5]*cu[5] - 3.0*0.5*usqr)
#     feq[6, :, :] = rho*t[6]*(1. + 3.0*cu[6] + 9*0.5*cu[6]*cu[6] - 3.0*0.5*usqr)
#     feq[7, :, :] = rho*t[7]*(1. + 3.0*cu[7] + 9*0.5*cu[7]*cu[7] - 3.0*0.5*usqr)
#     feq[8, :, :] = rho*t[8]*(1. + 3.0*cu[8] + 9*0.5*cu[8]*cu[8] - 3.0*0.5*usqr)
# Set up

LeftWall = np.fromfunction(lambda x,y:x==0,(xsize,ysize))
RightWall = np.fromfunction(lambda x,y:x==xsize_max,(xsize,ysize))
BottomWall = np.fromfunction(lambda x,y:y==ysize_max,(xsize,ysize))


wall = np.logical_or(np.logical_or(LeftWall, RightWall), BottomWall)

# velocity initial/boundary conditions
InitVel = np.zeros((2,xsize,ysize))
InitVel[0,:,0] = uLB

# initial distributions
feq = equ(rho, InitVel)

np.copyto(fin,feq)
np.copyto(fin1,feq)
np.copyto(fpost,feq)

# interactive figure mode
if (SavePlot):
    pyplot.ioff()
    f = pyplot.figure(figsize=(30,16))

os.chdir(OutputFolder)

# Following are hardcoded to avoid if loops. Options are just for printing.
regime = 'Laminar' #options : Laminar, Turbulent
tmethod = 'SRT' # options : SRT, TRT, MRT
BC = 'EB-NEBB ' # options : BB(half way link based), NEBB (wet node)

t2 = nmb.f8[:,:];t3 = nmb.f8[:,:,:]
#@nmb.jit(nmb.f8[:,:,:](nmb.f8[:,:], nmb.f8[:,:,:]))
#@cache(maxsize=None)
#@nmb.jit(nmb.types.Tuple((t2,t3,t3,t3))(t2,t3,t3,t3,nmb.i8))
#@nmb.guvectorize(['void(float64[:,:],float64[:,:,:],float64[:,:,:],int64,float64[:,:,:])'],"(n,n),(l,n,n),(m,n,n),(),(m,n,n)->(m,n,n)", target='cpu')
@nmb.autojit((t2,t3,t3,nmb.i8,t3))
def allfunc(rho,u,feq, maxIt, fin):
    
    for It in range(maxIt):
        
        
        rho = np.sum(fin,axis=0)    
        
        #u = np.dot(c.transpose(), fin.transpose((1,0,2)))/rho
        #peeling the loop to increase the speed
        u[0] = (c[0,0]*fin[0]+c[1,0]*fin[1]+c[2,0]*fin[2]+c[3,0]*fin[3]+c[4,0]*fin[4]+c[5,0]*fin[5]+c[6,0]*fin[6]+c[7,0]*fin[7]+c[8,0]*fin[8])/rho
        u[1] = (c[0,1]*fin[0]+c[1,1]*fin[1]+c[2,1]*fin[2]+c[3,1]*fin[3]+c[4,1]*fin[4]+c[5,1]*fin[5]+c[6,1]*fin[6]+c[7,1]*fin[7]+c[8,1]*fin[8])/rho

        
        rho[:, 0] = np.sum(fin[CentHStencil, :, 0],axis=0)+2*np.sum(fin[TopStencil, :, 0],axis=0)
     
        u[:,0,1:]= 0 ; u[:,xsize_max,1:]= 0 ; u[:,:,ysize_max]= 0
        u[0,:,0]=uLB ; u[1,:,0] =0 #10 chosen randomly to not force corners
     
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
        

        #feq = equ(rho,u)
        
        for i in range(q):
           feq[i, :, :] = rho*t[i]*(1. + 3.0*cu[i] + 9*0.5*cu[i]*cu[i] - 3.0*0.5*usqr)
 
        #rhob = rho[np.newaxis,:,:]
        #usqrb = usqr[np.newaxis,:,:]
        #feq = ne.evaluate('rhob*tb*( 1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usqrb)')

        fpost = fin - omega*( fin-feq)
        
        #fpost = ne.evaluate('fin - omega*( fin-feq)')     

        #streaming
    
        fin[0, :, :] = fpost[0, :, :]
    
        fin[1, 1:xsize_max,   :]     = fpost[1, 0:xsize_max-1,  :]
        fin[2,   :,   0:ysize_max-1] = fpost[2,   :,    1:ysize_max]
        fin[3, 0:xsize_max-1, :]     = fpost[3, 1:xsize_max,    :]
        fin[4,   :,   1:ysize_max]   = fpost[4,   :,    0:ysize_max-1]

        fin[5, 1:xsize_max,   0:ysize_max-1] = fpost[5, 0:xsize_max-1, 1:ysize_max]
        fin[6, 0:xsize_max-1, 0:ysize_max-1] = fpost[6, 1:xsize_max,   1:ysize_max]
        fin[7, 0:xsize_max-1, 1:ysize_max]   = fpost[7, 1:xsize_max,   0:ysize_max-1]
        fin[8, 1:xsize_max,   1:ysize_max]   = fpost[8, 0:xsize_max-1, 0:ysize_max-1]

        fin[RightStencil, 0, :] =  feq[RightStencil, 0, :] #- feq[LeftStencil, 0, :] + fin[LeftStencil, 0, :]
        fin[LeftStencil, xsize_max, :] = - feq[RightStencil, xsize_max, :] + (feq[LeftStencil, xsize_max, :] + fin[RightStencil, xsize_max, :])
        fin[TopStencil, :, ysize_max] = - feq[BotStencil, :, ysize_max] + (feq[TopStencil, :, ysize_max] + fin[BotStencil, :, ysize_max])
        fin[BotStencil, :, 0] = - feq[TopStencil, :, 0] + (feq[BotStencil, :, 0] + fin[TopStencil, :, 0])
        
     
   
fin = allfunc(rho,u,feq, maxIt,fin) 

# # Time Loop for It in range(maxIt):
#     # macro density
#     #np.copyto(ftemp,fin)
# #     ftemp1 = fin1
#     #rho = np.add.reduce(fin)
#     rho = np.sum(fin,axis=0)
#     
#     
#     #TRT
#     #fplus = 0.5*(fin[:,:,:] + fin[bounce[:], :,:])
#     #fminus = 0.5*(fin[:,:,:] - fin[bounce[:], :,:])
#     #simplifying to increase speed
# #     fplus[TopStencil]  = 0.5*(fin[TopStencil] + fin[BotStencil]); fplus[BotStencil] = fplus[TopStencil]
# #     fplus[1]  = 0.5*(fin[1] + fin[3]); fplus[3] = fplus[1];fplus[0] = fin[0]
# #     fminus[TopStencil]  = 0.5*(fin[TopStencil] - fin[BotStencil]); fminus[BotStencil] = -fminus[TopStencil]
# #     fminus[1]  = 0.5*(fin[1] - fin[3]); fminus[3] = -fminus[1];fminus[0] = 0
# #     
# #     fplus[2]  = 0.5*(fin[2] + fin[4]); fplus[4] = fplus[2]
# #     fplus[5]  = 0.5*(fin[5] + fin[7]); fplus[7] = fplus[5]
# #     fplus[6]  = 0.5*(fin[6] + fin[8]); fplus[8] = fplus[6]
# #     fplus[1]  = 0.5*(fin[1] + fin[3]); fplus[3] = fplus[1];fplus[0] = fin[0]
# #     fminus[2]  = 0.5*(fin[2] - fin[4]); fminus[4] = -fminus[2]
# #     fminus[5]  = 0.5*(fin[5] - fin[7]); fminus[7] = -fminus[5]
# #     fminus[6]  = 0.5*(fin[6] - fin[8]); fminus[8] = -fminus[6]
# #     fminus[1]  = 0.5*(fin[1] - fin[3]); fminus[3] = -fminus[1];fminus[0] = 0  
#     
#     
#     #print(It)
# 
# #     rho1 = sumf(fin1)
#     #u = np.dot(c.transpose(), fin.transpose((1,0,2)))/rho
#     #peeling the loop to increase the speed
# 
#     u[0] = (c[0,0]*fin[0]+c[1,0]*fin[1]+c[2,0]*fin[2]+c[3,0]*fin[3]+c[4,0]*fin[4]+c[5,0]*fin[5]+c[6,0]*fin[6]+c[7,0]*fin[7]+c[8,0]*fin[8])/rho
#     u[1] = (c[0,1]*fin[0]+c[1,1]*fin[1]+c[2,1]*fin[2]+c[3,1]*fin[3]+c[4,1]*fin[4]+c[5,1]*fin[5]+c[6,1]*fin[6]+c[7,1]*fin[7]+c[8,1]*fin[8])/rho
# 
#   
# #     u[0] = np.sum(cxb*fin/rho,axis=0)
# #     u[1] = np.sum(cyb*fin/rho,axis=0)
# #     
# #     rhob = rho[np.newaxis,:,:]
# #     u[0] = ne.evaluate('sum(cxb*fin/rhob,axis=0)')
# #     u[1] = ne.evaluate('sum(cyb*fin/rhob,axis=0)')  
# 
# 
# #     u1[0,:,:] = (c[0,0]*fin1[0]+c[1,0]*fin1[1]+c[2,0]*fin1[2]+c[3,0]*fin1[3]+c[4,0]*fin1[4]+c[5,0]*fin1[5]+c[6,0]*fin1[6]+c[7,0]*fin1[7]+c[8,0]*fin1[8])/rho
# #     u1[1,:,:] = (c[0,1]*fin1[0]+c[1,1]*fin1[1]+c[2,1]*fin1[2]+c[3,1]*fin1[3]+c[4,1]*fin1[4]+c[5,1]*fin1[5]+c[6,1]*fin1[6]+c[7,1]*fin1[7]+c[8,1]*fin1[8])/rho
# 
# 
# # 
#     rho[:, 0] = sumf(fin[CentHStencil, :, 0])+2.*sumf(fin[TopStencil, :, 0])
# #     #rho1[:, 0] = sumf(ftemp1[CentHStencil, :, 0])+2.*sumf(ftemp1[TopStencil, :, 0])
# #     #u[:,:,0]=InitVel[:,:,0]
# #     
#     u[:,0,1:]= 0 ; u[:,xsize_max,1:]= 0 ; u[:,:,ysize_max]= 0
#     u[0,:,0]=uLB ; u[1,:,0] =0 #10 chosen randomly to not force corners
# #     
# 
#     feq = equ(rho,u)
# #     feq1 = equ(rho1,u1)
#     
#     #MRT
# #     m_GS = np.dot(M_GS,fin.transpose(1,0,2))  # warning : m_GS is not same as M_GS
# #     jx = m_GS[3]; jy = m_GS[5]
# #     np.copyto(m_GS_eq,m_GS)# initiating m_GS equilibrium
# #     jx = m_GS[3]; jy = m_GS[5] 
# #     m_GS_eq[0,:,:] = rho
# #     m_GS_eq[1,:,:] = -2.0*rho + 3.0*(jx*jx + jy*jy)
# #     m_GS_eq[2,:,:] =  - 3.0*(jx*jx + jy*jy) + rho + 9*(jx*jx*jy*jy )
# #     m_GS_eq[4,:,:] = - jx + 3.0*(jx**3)
# #     m_GS_eq[6,:,:] = - jy + 3.0*(jy**3)
# #     m_GS_eq[7,:,:] = jx*jx - jy*jy
# #     m_GS_eq[8,:,:] = jx*jy
#     #TRT    
#     #feplus = 0.5*(feq[:,:,:] + feq[bounce[:], :,:])
#     #feminus = 0.5*(feq[:,:,:] - feq[bounce[:], :,:])
#     #simplifying to increase speed
# #     feplus[TopStencil]  = 0.5*(feq[TopStencil] + feq[BotStencil]); feplus[BotStencil] = feplus[TopStencil]
# #     feplus[1]  = 0.5*(feq[1] + feq[3]); feplus[3] = feplus[1];feplus[0] = feq[0]
# #     feminus[TopStencil]  = 0.5*(feq[TopStencil] - feq[BotStencil]); feminus[BotStencil] = -feminus[TopStencil]
# #     feminus[1]  = 0.5*(feq[1] - feq[3]); feminus[3] = -feminus[1];feminus[0] = 0    
#    
# #     feplus[2]  = 0.5*(feq[2] + feq[4]); feplus[4] = feplus[2]
# #     feplus[5]  = 0.5*(feq[5] + feq[7]); feplus[7] = feplus[5]
# #     feplus[6]  = 0.5*(feq[6] + feq[8]); feplus[8] = feplus[6]
# #     feplus[1]  = 0.5*(feq[1] + feq[3]); feplus[3] = feplus[1];feplus[0] = feq[0]
# #     feminus[2]  = 0.5*(feq[2] - feq[4]); feminus[4] = -feminus[2]
# #     feminus[5]  = 0.5*(feq[5] - feq[7]); feminus[7] = -feminus[5]
# #     feminus[6]  = 0.5*(feq[6] - feq[8]); feminus[8] = -feminus[6]
# #     feminus[1]  = 0.5*(feq[1] - feq[3]); feminus[3] = -feminus[1];feminus[0] = 0 
# #     
#     #Collision - MRT    
# #     m_GS = m_GS - dot(omega_diag, transpose((m_GS-m_GS_eq), (1,0,2)))
# #     fpost = dot(M_GS_INV , m_GS.transpose(1,0,2))
#     #temp = dot(omega_diag, transpose((m_GS-m_GS_eq), (1,0,2)))
#     #fpost = fin - dot(M_GS_INV , transpose(temp, (1,0,2)) ) 
#     
#        
#     #TRT
#     #Collision - TRT
# #     omegaS = 1.0/tauS
# #     fpost = fin - omegaS*(fplus-feplus) - omegam*( fminus-feminus)
#     #fpost = fin - omegap*(fplus-feplus) - omegam*( fminus-feminus)
#    
#     #Collision - SRT
#     #omegaS= 1.0/tauS
#     #fpost = fin - omegaS*(fin-feq)
#     #fpost = ne.evaluate('fin - omegaS*(fin-feq)')
#     #fpost = fin - omega*( fin-feq)
#     fpost = ne.evaluate('fin - omega*( fin-feq)')     
#     
# 
# 
#     #print(mean(fpost))
# 
#     #streaming
#     
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
#     
# #     fin1[0, :, :] = fpost1[0, :, :]
# #      
# #     fin1[1, 1:xsize_max,   :]     = fpost1[1, 0:xsize_max-1,  :]
# #     fin1[2,   :,   0:ysize_max-1] = fpost1[2,   :,    1:ysize_max]
# #     fin1[3, 0:xsize_max-1, :]     = fpost1[3, 1:xsize_max,    :]
# #     fin1[4,   :,   1:ysize_max]   = fpost1[4,   :,    0:ysize_max-1]
# #  
# #     fin1[5, 1:xsize_max,   0:ysize_max-1] = fpost1[5, 0:xsize_max-1, 1:ysize_max]
# #     fin1[6, 0:xsize_max-1, 0:ysize_max-1] = fpost1[6, 1:xsize_max,   1:ysize_max]
# #     fin1[7, 0:xsize_max-1, 1:ysize_max]   = fpost1[7, 1:xsize_max,   0:ysize_max-1]
# #     fin1[8, 1:xsize_max,   1:ysize_max]   = fpost1[8, 0:xsize_max-1, 0:ysize_max-1]
#      
#  
# 
#     # boundary condition at walls
#     
# 
# #Simple Bounceback - half way link based - works only when tau/Dt is around 0.93
# #     for value in LeftStencil: fin[value, RightWall ] = fpost[bounce[value], RightWall] 
# #     for value in RightStencil: fin[value, LeftWall ] = fpost[bounce[value], LeftWall]  
# #     for value in TopStencil: fin[value, BottomWall ] = fpost[bounce[value], BottomWall] 
# #     # Bouzidi condition for top lid
# #  
# #     fin[4, 1:xsize_max-1,0 ] = fpost[2, 1:xsize_max-1,0] 
# #     fin[7, 1:xsize_max-1,0 ] = fpost[5, 1:xsize_max-1,0] - array(1/6.0)*uLB
# #     fin[8, 1:xsize_max-1,0 ] = fpost[6, 1:xsize_max-1,0] + array(1/6.0)*uLB
# 
# #     fin[LeftStencil, RightWall ] = ftemp[asarray( [bounce[i] for i in LeftStencil]) , RightWall]
# #     fin[TopStencil, BottomWall ] = ftemp[asarray( [bounce[i] for i in TopStencil]) , BottomWall]
# #     fin[RightStencil, LeftWall ] = ftemp[asarray( [bounce[i] for i in RightStencil]) , LeftWall]
#   
#     #Accounting for moving wall using zou-he condition 
#     # NEBB for walls
#     
#     fin[RightStencil, 0, :] =  feq[RightStencil, 0, :] #- feq[LeftStencil, 0, :] + fin[LeftStencil, 0, :]
#     fin[LeftStencil, xsize_max, :] = - feq[RightStencil, xsize_max, :] + (feq[LeftStencil, xsize_max, :] + fin[RightStencil, xsize_max, :])
#     fin[TopStencil, :, ysize_max] = - feq[BotStencil, :, ysize_max] + (feq[TopStencil, :, ysize_max] + fin[BotStencil, :, ysize_max])
#     fin[BotStencil, :, 0] = - feq[TopStencil, :, 0] + (feq[BotStencil, :, 0] + fin[TopStencil, :, 0])
# #     
#   #   fin1[RightStencil, 0, :] = - feq1[LeftStencil, 0, :] + (feq1[RightStencil, 0, :] + ftemp1[LeftStencil, 0, :])
#   #   fin1[TopStencil, :, ysize_max] = - feq1[BotStencil, :, ysize_max] + (feq1[TopStencil, :, ysize_max] + ftemp1[BotStencil, :, ysize_max])
# #     fin1[LeftStencil, xsize_max, :] = - feq1[RightStencil, xsize_max, :] + (feq1[LeftStencil, xsize_max, :] + ftemp1[RightStencil, xsize_max, :])
# #     fin1[BotStencil, :, 0] = - feq1[TopStencil, :, 0] + (feq1[BotStencil, :, 0] + ftemp1[TopStencil, :, 0])
# #        
# 
# #NEBB accounting for tangential velocities
#     #temp = uLB
#     #uLB = 0
#      
# #     fin[4,:,0] = fin[2,0,:]
# #     fin[7,:,0] = fin[5,0,:] + 0.5*(fin[1,:,0] - fin[3,:,0]) - 0.5*uLB
# #     fin[8,:,0] = fin[6,0,:] - 0.5*(fin[1,:,0] - fin[3,:,0]) + 0.5*uLB
# # #             
# # # # # #     #corners - upper left and then upper right
# # # # #  #   uLB = 0     
# #     fin[1,0,0] = fin[3,0,0] + (2.0/3.0)*uLB
# #     fin[4,0,0] = fin[2,0,0] 
# #     fin[8,0,0] = fin[6,0,0] + (1.0/6.0)*uLB
# #     fin[5,0,0] = +(1.0/12.0)*uLB
# #     fin[7,0,0] = -(1.0/12.0)*uLB
# #     fin[0,0,0] = 1.0 - sumf(fin[1:,0,0])
# #     fin[3,xsize_max,0] = fin[1,xsize_max,0] - (2.0/3.0)*uLB
# #     fin[4,xsize_max,0] = fin[2,xsize_max,0] 
# #     fin[7,xsize_max,0] = fin[5,xsize_max,0] - (1.0/6.0)*uLB
# #     fin[6,xsize_max,0] =  -(1.0/12.0)*uLB
# #     fin[8,xsize_max,0] =  +(1.0/12.0)*uLB
# #     fin[0,xsize_max,0] = 1.0 - sumf(fin[1:,xsize_max,0])    
#     #uLB = temp
#     
#     #Adding smagorinsky models
#     ##Cs2 = 0.01 #0.0289 , smagorinsky constant 
#     #Applying Van Driest damping to make Cs2 zero at walls
# #     visc_inv = sqrt( (u[0,int(xsize/2),0]-u[0,int(xsize/2),1])/nuLB ) #viscous length scale inverse assuming dy =1
# #     Zplus = minimum(Xcoord-0,minimum(xsize_max-Xcoord,minimum(Ycoord-0,ysize_max-Ycoord)))*visc_inv # closest distance to a wall
# #     Csbulk = 0.16
# #     Cs = Csbulk*(1 - exp(-Zplus/26))
# #     Cs2 = Cs*Cs
# #     product1 = c[0,0]*c[0,1]*fin[0,:,:] + c[1,0]*c[1,1]*fin[1,:,:] + c[2,0]*c[2,1]*fin[2,:,:] + c[3,0]*c[3,1]*fin[3,:,:] + c[4,0]*c[4,1]*fin[4,:,:] + c[5,0]*c[5,1]*fin[5,:,:] + c[6,0]*c[6,1]*fin[6,:,:] + c[7,0]*c[7,1]*fin[7,:,:] + c[8,0]*c[8,1]*fin[8,:,:] 
# #     product2 = c[0,0]*c[0,1]*feq[0,:,:] + c[1,0]*c[1,1]*feq[1,:,:] + c[2,0]*c[2,1]*feq[2,:,:] + c[3,0]*c[3,1]*feq[3,:,:] + c[4,0]*c[4,1]*feq[4,:,:] + c[5,0]*c[5,1]*feq[5,:,:] + c[6,0]*c[6,1]*feq[6,:,:] + c[7,0]*c[7,1]*feq[7,:,:] + c[8,0]*c[8,1]*feq[8,:,:] 
# #     Qmf = product1 - product2 # momentum flux
# #     #tauS = ne.evaluate('0.5*(tau + sqrt( (tau*tau + ( 18*1.4142*Cs2*abs(Qmf) )/rho ) ) )') # tau + tau_turbulent
# #     tauS = 0.5*(tau + sqrt( (tau*tau + ( 18*1.4142*Cs2*abs(Qmf) )/rho ) ) ) # tau + tau_turbulent
# 
#     
#     if( (It%Pinterval == 0) & (SaveVTK | SavePlot)) :
#        
#         print ('current iteration :', It)
#         print (np.mean(u[0,:,0])/uLB)
#         Usquare = u[0,:,:]**2 + u[1,:,:]**2
#         
#         Usquare = Usquare/(uLB**2)
#         BCoffset = int(xsize/40)
#         # replacing all boundaries with nan to get location of vortices
#         Usquare[0:BCoffset,:] = nan ; Usquare[:,0:BCoffset] = nan
#         Usquare[xsize_max-BCoffset:xsize,:] = nan;Usquare[:,ysize_max-BCoffset:ysize] = nan
#         Loc1 = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)
#         #print(Loc1)
#         #print(Usquare[Loc1[0], Loc1[1]])
#         # finding other vortices
#         Usquare[Loc1[0]-BCoffset:Loc1[0]+BCoffset,Loc1[1]-BCoffset:Loc1[1]+BCoffset] = nan
#         Loc2 = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)
#         #print(Loc2)
#         #print(Usquare[Loc2[0], Loc2[1]])        
#         
#         
#         
#         if (SavePlot):
#             
#             f.clear()
#             subplot1 = pyplot.subplot2grid((2,15),(0,0), colspan=4, rowspan=1)
#             subplot2 = pyplot.subplot2grid((2,15),(0,5), colspan=4, rowspan=1)
#             subplot3 = pyplot.subplot2grid((2,15),(0,10), colspan=5, rowspan=1)
#             subplot4 = pyplot.subplot2grid((2,15),(1,0), colspan=10, rowspan=1)
#             #subplot5 = pyplot.subplot2grid((2,15),(1,5), colspan=4, rowspan=1)
#             #subplot6 = pyplot.subplot2grid((2,15),(1,10), colspan=5, rowspan=1)          
#             
#             matplotlib.rcParams.update({'font.size': 15})
#             Ux = u[0,int(xsize/2),:]/uLB 
#             subplot1.plot(Ux, YNorm, label="LBM")
#             subplot1.plot(Ux_GhiaVer, Y_GhiaVer, 'g*' , label="Ghia")
#             subplot1.set_title('Ux on middle column', fontsize = 20, y =1.02)
#             subplot1.legend(loc = 'center right')
#             subplot1.set_xlabel('Ux', fontsize = 20);subplot1.set_ylabel('Y-position', fontsize = 20)
#             
#             Uy = u[1,:,int(ysize/2)]/uLB
#             subplot2.plot(XNorm, Uy, label="LBM")
#             subplot2.plot(X_GhiaHor,Uy_GhiaHor, 'g*', label='Ghia')
#             subplot2.set_title('Uy on middle row', fontsize = 20, y=1.02)
#             subplot2.legend(loc = 'upper right')
#             subplot2.set_xlabel('X-position', fontsize = 20);subplot2.set_ylabel('Uy', fontsize = 20)
#             
#             #subplot1.imshow(sqrt(u[0]**2+u[1]**2).transpose(), pyplot.set_cmap('jet') , vmin = 0, vmax = 0.02)
#             color1 = (np.sqrt(u[0,:,:]**2+u[1,:,:]**2)/uLB ).transpose()
#             strm = subplot3.streamplot(XNorm,YNorm,(u[0,:,:]).transpose(),(u[1,:,:]).transpose(), color =color1,cmap=pyplot.cm.jet)#,norm=matplotlib.colors.Normalize(vmin=0,vmax=1)) 
#             cbar = pyplot.colorbar(strm.lines, ax = subplot3)
#             subplot3.plot(Loc1[0]/xsize,(ysize_max-Loc1[1])/ysize,'ro', label='Vortex1')
#             subplot3.plot(Loc2[0]/xsize,(ysize_max-Loc2[1])/ysize,'mo', label='Vortex2')
#             subplot3.plot(X_Vor, Y_Vor,'ks')
#             subplot3.set_title('Velocity Streamlines - LBM', fontsize = 20, y =1.02)
#             subplot3.margins(0.005) #subplot3.axis('tight')
#             subplot3.set_xlabel('X-position', fontsize = 20);subplot3.set_ylabel('Y-position', fontsize = 20)
#             
#             #print((YNorm == array(Y_GhiaVer*ysize_max).astype(int)))
#             Ux_temp = u[0,int(xsize/2),LBMy]/uLB
#             temp = abs( ( abs(Ux_GhiaVer[:-1]) - abs(np.fliplr(np.atleast_2d(Ux_temp))[0])) / ( len(Ux_temp)*np.maximum(abs(Ux_GhiaVer[:-1]),abs(np.fliplr(np.atleast_2d(Ux_temp))[0]) )) )  #rsquare estimate
#             NormErr_column.append(1-sumf(temp))
#             #print ((1 - sumf(abs(fliplr(atleast_2d(Ux_GhiaVer[:-1]))[0] - Ux_temp)/(len(Ux_temp)*abs(Ux_GhiaVer[:-1])) )))
#             #print(NormErr_column)
#             #NormErr_column.append(1 - sumf(square(fliplr(atleast_2d(Ux_GhiaVer[:-1]))[0] - Ux_temp))/(len(Ux_temp)*var(Ux_GhiaVer[:-1])) ) #rsquare estimate
#             #NormErr_column.append(1 - (1/len(LBMy))*sumf( square( (fliplr(atleast_2d(Ux_GhiaVer[:-1]))[0] - Ux_temp)/(Ux_GhiaVer[:-1])) )  ) #rsquare estimate
#             #NormErr_column.append(abs(1 - math.log10((1/len(LBMy))*sumf( square( (fliplr(atleast_2d(Ux_GhiaVer[:-1]))[0] - Ux_temp)/(Ux_GhiaVer[:-1])) ))  )) #rsquare estimate
# 
#             #NormErr_column.append(1 - sumf(square(fliplr(atleast_2d(Ux_GhiaVer))[0] - Ux_temp))/sumf(square(Ux_GhiaVer)) ) #rsquare estimate
# 
#             time_column.append(It)
#             subplot4.plot(time_column,NormErr_column,)
#             subplot4.set_title('Regression value - Ux_MiddleColumn' )  
#             subplot4.set_xlabel('time iteration', fontsize = 20);subplot4.set_ylabel('Regression value', fontsize = 20)
#             pyplot.figtext(0.5,0.3,'Current Regression value is')
#             pyplot.figtext(0.5,0.28,str(round(NormErr_column[-1],3)))
#             
#             pyplot.figtext(0.65,0.45,"Square dots in above figure represent vortex locations from Ghia data")
#             pyplot.figtext(0.65,0.43,"Circular dots represent vortex locations of current simulation")
#             pyplot.figtext(0.65,0.35,'LBM parameters: '+tmethod, fontsize=20)
#             pyplot.figtext(0.65,0.31,'Grid size: '+str(xsize)+'*'+str(ysize))
#             pyplot.figtext(0.65,0.29,'Re: '+str(Re)+'    '+'BoundaryCondition: '+BC)
#             pyplot.figtext(0.65,0.27,'Lid velocity in LB units: '+str(uLB)+'    dx* and dt* hardcoded as 1')
#             pyplot.figtext(0.65,0.25,'tau - related to dynamic viscosity: '+str(round(1.0/omega,3)))
#             if (tmethod=='SRT'):
#                 data_out = 'omega: '+str(round(omega,2))
#                 pyplot.figtext(0.65,0.23,data_out)
#             elif (tmethod=='TRT'):
#                 data_out = 'omega_plus, omega_minus, delta: '+str(round(omegap,3))+' , '+str(round(omegam,3))+' , '+str(round(delTRT,3)) 
#                 pyplot.figtext(0.65,0.23,data_out)
#             elif (tmethod=='MRT'):
#                 data_out = 'omega_nu, omega_e, omega_eps, omega_q: '
#                 pyplot.figtext(0.65,0.23,data_out)
#                 data_out = str(round(omega_nu,3))+' , '+str(round(omega_e,3))+' , '+str(round(omega_eps,3))+' , '+str(round(omega_q,3))
#                 pyplot.figtext(0.65,0.21,data_out)
#             if( regime=='Turbulent'):
#                 data_out = 'Smagorinsky constant, Cs = '+ str(Cs[int(xsize/2),0])+' at wall to '+str(Csbulk)+' at bulk'
#                 pyplot.figtext(0.65,0.17,data_out)
#                 data_out = 'Mean relaxation time, tau+tau_turbulent, is  '+ str(mean(tauS))
#                 pyplot.figtext(0.65,0.15,data_out)
#             
#             f.suptitle('Lid Driven Cavity - Re'+str(int(Re))+' '+regime+' '+tmethod+' '+BC+' '+str(xsize)+'*'+str(ysize), fontsize = 30, y =1.04)
#             pyplot.savefig(project + "_" + str(int(It/Pinterval)).zfill(5) + ".png",bbox_inches = 'tight', pad_inches = 0.4)
# 
#         if ( SaveVTK ):
#             # convert 2d data to 3d arrays
#             Vel = reshape(u, (2, xsize, ysize, 1))
#             Vel3D = (Vel[0, :, :, :], Vel[1, :, :, :], velZ)   
#             Rho3D = reshape(rho, (xsize, ysize, 1))
#             index = str(int(It/Pinterval)).zfill(5)
#             saveToVTK(Vel3D, Rho3D, project, index, grid)
#         tend = timer()
# 
#         print ( 'time elapsed is ', (tend-tstart), 'seconds' )
#             

os.chdir(CurrentFolder)
tend = timer()

print ( 'TOTAL time elapsed is ', (tend-tstart), 'seconds' )

#def main():
#    print('the end')


#if __name__ == '__main__':
 #   main()
