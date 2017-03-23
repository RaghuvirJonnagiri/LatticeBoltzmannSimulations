'''
@author: raghuvir
#LidDrivenCavity - BGK and TRT Model
use sudo apt-get install python-matplotlib
sudo apt-get install python-vtk?

'''

from numpy import *
import numexpr as ne
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from VTKWrapper import saveToVTK
import os
from timeit import default_timer as timer
tstart = timer()
#from rexec import RHooks
os.system("taskset -p 0xff %d" % os.getpid())

#print('number of cores detected is'); ne.detect_number_of_threads()
ne.set_num_threads(ne.detect_number_of_threads())

# Plot Settings
Pinterval = 100# iterations to print next output
SavePlot = True #save plots
SaveVTK = True # save files in vtk format
project = 'ldc'  #for naming output files 
OutputFolder = './output'
CurrentFolder = os.getcwd()

# Lattice Parameters
maxIt = 200000 # time iterations
Re    = 1000.0 # Reynolds number

#Number of cells
xsize, ysize = 600, 600
xsize_max, ysize_max = xsize-1, ysize-1 # highest index in each direction
q = 9 # d2Q9

uLB = 0.08 # velocity in lattice units
nuLB = uLB*ysize/Re #viscosity coefficient

#SRT
omega = 2.0 / (6.*nuLB+1)

#MRT - relaxation time vector
omega_nu = 2.0 / (6.*nuLB+1) # from shear viscosity
omega_e =  1.0 # stokes hypothesis - bulk viscosity is zero
omega_eps , omega_q = 0.72,0.72 #0.71, 0.83 # randomly chosen
omega_vec = [1.0, omega_e, omega_eps, 1.0, omega_q, 1.0, omega_q, omega_nu, omega_nu]
omega_diag = diag(omega_vec)

## Plot Setup

#creating output directory
if not os.path.isdir(OutputFolder):
    try:
        os.makedirs(OutputFolder)
    except OSError:
        pass

#Grid Setup
Xgrid = arange(0,xsize,dtype='float64')
Ygrid = arange(0,ysize,dtype='float64')
Zgrid = arange(0, 1,dtype='float64')
grid = Xgrid, Ygrid, Zgrid

velZ = zeros((xsize,ysize,1)) # velocity in Z direction is zero

# axis for velocity plots
YNorm = arange(ysize,0,-1,dtype='float64')/ysize # y starts as 0 from top lid
XNorm = arange(0,xsize,1,dtype='float64')/xsize # x starts as 0 from left wall

GhiaData = genfromtxt('GhiaData.csv', delimiter = ",")[6:,1:]

# Ghia Data for Re 100 to Re 10000
X_GhiaHor = GhiaData[:,9]
Y_GhiaVer = GhiaData[:,0]

Re_dict = {100:1, 400:2, 1000:3, 3200:4, 5000:5, 7500:6, 10000:7} # column numbers in csv file for Re
Ux_GhiaVer = GhiaData[:,Re_dict[Re]]
Uy_GhiaHor = GhiaData[:,Re_dict[Re]+9]

NormErr_column = []; time_column = [] # initializing arrays to track LBM and Ghia differences

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
c = array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
# Lattice Weights
t = 1.0/36. * ones(q)
t[1:5] = 1.0/9.0
t[0] = 4.0/9.0 

# reverse index for no-slip bounceback condition
noslip = [0,3,4,1,2,7,8,5,6]

#indexed arrays for stencil sides
LeftStencil   = arange(q)[asarray([temp[0] <  0 for temp in c])]
CentVStencil  = arange(q)[asarray([temp[0] == 0 for temp in c])]
RightStencil  = arange(q)[asarray([temp[0] >  0 for temp in c])]
TopStencil    = arange(q)[asarray([temp[1] >  0 for temp in c])]
CentHStencil  = arange(q)[asarray([temp[1] == 0 for temp in c])]
BotStencil    = arange(q)[asarray([temp[1] <  0 for temp in c])]

#MRT GRAM SCHMIDT VELOCITY MATRIX
M_GS = ones((9,9))
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
M_GS_INV = ones((9,9))
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
def sumf(fin):
    #return sum(fin, axis =0)
     return ne.evaluate('sum(fin, axis =0)') # using numexpr for multithreading
#compute equilibrium distribution
def equ(rho,u):
    cu   = 3.0 * dot(c, u.transpose(1, 0, 2))
    #usqr = 3./2.*(u[0]**2+u[1]**2)
    temp1 = ne.evaluate('u**2'); temp2 = ne.evaluate('sum(temp1,0)'); usqr = ne.evaluate('(3.0/2.0)*temp2')
    feq = zeros((q, xsize, ysize))
    for i in range(q):
        feq[i, :, :] = rho*t[i]*(1. + cu[i] + 0.5*cu[i]**2 - usqr)
    return feq

# Set up
LeftWall = fromfunction(lambda x,y:x==0,(xsize,ysize))
RightWall = fromfunction(lambda x,y:x==xsize_max,(xsize,ysize))
BottomWall = fromfunction(lambda x,y:y==ysize_max,(xsize,ysize))

wall = logical_or(logical_or(LeftWall, RightWall), BottomWall)

# velocity initial/boundary conditions
InitVel = zeros((2,xsize,ysize))
InitVel[0,:,0] = uLB

# initial distributions
feq = equ(1.0, InitVel)
fin = feq.copy()
fpost = feq.copy()

# interactive figure mode
if (SavePlot):
    pyplot.ioff()
    f = pyplot.figure(figsize=(30,16))

os.chdir(OutputFolder)


# Time Loop
for It in range(maxIt):
    # macro density
    ftemp = fin
    
    #TRT
    fplus = 0.5*(fin[:,:,:] + fin[noslip[:], :,:])
    fminus = 0.5*(fin[:,:,:] - fin[noslip[:], :,:])
          
    # macro velocity
    
    #print(It)
    rho = sumf(fin)
    u = dot(c.transpose(), fin.transpose((1,0,2)))/rho
   #u = zeros((2,xsize,ysize))
   #for m in range(xsize):
        #for l in range(2):
     #       u[l,m,:] = dot( c[:,l].transpose(), fin[:,m,:])
           
    
    rho[:, 0] = sumf(ftemp[CentHStencil, :, 0])+2.*sumf(ftemp[TopStencil, :, 0])
    #u[:,:,0]=InitVel[:,:,0]
    
    feq = equ(rho,u)
    
    #MRT
#     m_GS = dot(M_GS,fin.transpose(1,0,2))  # warning : m_GS is not same as M_GS
#      
#     m_GS_eq = m_GS # initiating m_GS equilibrium
#     m_GS_eq[0,:,:] = rho    
#     m_GS_eq[1,:,:] = rho*(-2.0 + 3.0*(u[0,:,:]*u[0,:,:] + u[1,:,:]*u[1,:,:]))
#     m_GS_eq[2,:,:] =  - 3.0*rho*(u[0,:,:]*u[0,:,:] + u[1,:,:]*u[1,:,:]) + rho
#      
#     m_GS_eq[3,:,:] = rho*u[0,:,:]
#      
#     m_GS_eq[4,:,:] = - rho*u[0,:,:]
#     m_GS_eq[5,:,:] = rho*u[1,:,:]
#     m_GS_eq[6,:,:] = - rho*u[1,:,:]
#     m_GS_eq[7,:,:] = rho*(u[0,:,:]*u[0,:,:] - u[1,:,:]*u[1,:,:])
#     m_GS_eq[8,:,:] = rho*u[0,:,:]*u[1,:,:]
    #TRT
    feplus = 0.5*(feq[:,:,:] + feq[noslip[:], :,:])
    feminus = 0.5*(feq[:,:,:] - feq[noslip[:], :,:])
    
    #Collision - MRT
    
#    m_GS = m_GS - dot(omega_diag, transpose((m_GS-m_GS_eq), (1,0,2)))
#    fpost = dot(M_GS_INV , m_GS.transpose(1,0,2))
    
    #TRT
    #Collision - TRT
    omegap, omegam = 0.98, 1.2
    fpost = fin - omegap*(fplus-feplus) - omegam*( fminus-feminus)
        
    #Collision - SRT
    #fpost = fin - omega*( fin-feq)    
   
    

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
    
     
    #Accounting for moving wall using zou-he condition
    # boundary condition at walls
    
 
  
    for value in LeftStencil: fin[value, RightWall ] = ftemp[noslip[value], RightWall] 
    for value in TopStencil: fin[value, BottomWall ] = ftemp[noslip[value], BottomWall]  
    for value in RightStencil: fin[value, LeftWall ] = ftemp[noslip[value], LeftWall]  
#     fin[LeftStencil, RightWall ] = ftemp[asarray( [noslip[i] for i in LeftStencil]) , RightWall]
#     fin[TopStencil, BottomWall ] = ftemp[asarray( [noslip[i] for i in TopStencil]) , BottomWall]
#     fin[RightStencil, LeftWall ] = ftemp[asarray( [noslip[i] for i in RightStencil]) , LeftWall]
    
    #fin[BotStencil, 0, :] = - feq[TopStencil, 0, :] + (feq[BotStencil, 0, :] + ftemp[TopStencil, 0, :])

    temp = uLB
    uLB = 0

    fin[4,0,:] = fin[2,0,:]
    fin[7,0,:] = fin[5,0,:] + 0.5*(fin[1,0,:] - fin[3,0,:]) - 0.5*uLB
    fin[8,0,:] = fin[6,0,:] - 0.5*(fin[1,0,:] - fin[3,0,:]) + 0.5*uLB
     
    #corners - upper left and then upper right
    
    fin[1,0,0] = fin[3,0,0] + (2.0/3.0)*uLB
    fin[4,0,0] = fin[2,0,0] 
    fin[8,0,0] = fin[6,0,0] + (1.0/6.0)*uLB
    fin[5,0,0] = +(1.0/12.0)*uLB
    fin[7,0,0] = -(1.0/12.0)*uLB
    fin[0,0,0] = 1.0 - sumf(fin[1:,0,0])
    fin[3,0,xsize_max] = fin[1,0,xsize_max] - (2.0/3.0)*uLB
    fin[4,0,xsize_max] = fin[2,0,xsize_max] 
    fin[7,0,xsize_max] = fin[5,0,xsize_max] - (1.0/6.0)*uLB
    fin[6,0,xsize_max] =  -(1.0/12.0)*uLB
    fin[8,0,xsize_max] =  +(1.0/12.0)*uLB
    fin[0,0,xsize_max] = 1.0 - sumf(fin[1:,0,xsize_max])    
    uLB = temp
     
    if( (It%Pinterval == 0) & (SaveVTK | SavePlot)) :
       
        print ('current iteration :', It)
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
            subplot2.legend(loc = 'center right')
            subplot2.set_xlabel('X-position', fontsize = 20);subplot2.set_ylabel('Uy', fontsize = 20)
            
            #subplot1.imshow(sqrt(u[0]**2+u[1]**2).transpose(), pyplot.set_cmap('jet') , vmin = 0, vmax = 0.02)
            color1 = (sqrt(u[0,:,:]**2+u[1,:,:]**2)/uLB ).transpose()
            strm = subplot3.streamplot(XNorm,YNorm,(u[0,:,:]).transpose(),(u[1,:,:]).transpose(), color =color1,cmap=pyplot.cm.jet)#,norm=matplotlib.colors.Normalize(vmin=0,vmax=1)) 
            cbar = pyplot.colorbar(strm.lines, ax = subplot3)
            subplot3.set_title('Velocity Streamlines - LBM', fontsize = 20, y =1.02)
            subplot3.margins(0.005) #subplot3.axis('tight')
            subplot3.set_xlabel('X-position', fontsize = 20);subplot3.set_ylabel('Y-position', fontsize = 20)
            
            #print((YNorm == array(Y_GhiaVer*ysize_max).astype(int)))
            Ux_temp = u[0,int(xsize/2),array(Y_GhiaVer*ysize_max).astype(int)]
            NormErr_column.append(1 - sumf(square(flip(Ux_GhiaVer,0)- Ux_temp))/(len(Ux_GhiaVer)*var(Ux_GhiaVer)) ) #rsquare estimate
            time_column.append(It)
            subplot4.plot(time_column,NormErr_column,)
            subplot4.set_title('Normalized difference between LBM and Ghia Data')  
            subplot4.set_xlabel('time iteration', fontsize = 20);subplot4.set_ylabel('Normalized difference', fontsize = 20)

            f.suptitle('Lid Driven Cavity - Re' + str(int(Re)), fontsize = 30, y =1.04)
            pyplot.savefig(project + "." + str(It/Pinterval).zfill(4) + ".png",bbox_inches = 'tight', pad_inches = 0.4)

        if ( SaveVTK ):
            # convert 2d data to 3d arrays
            Vel = reshape(u, (2, xsize, ysize, 1))
            Vel3D = (Vel[0, :, :, :], Vel[1, :, :, :], velZ)   
            Rho3D = reshape(rho, (xsize, ysize, 1))
            index = str(It/Pinterval).zfill(4)
            saveToVTK(Vel3D, Rho3D, project, index, grid)
        tend = timer()

        print ( 'time elapsed is ', (tend-tstart), 'seconds' )
            

os.chdir(CurrentFolder)
tend = timer()

print ( 'time elapsed is ', (tend-tstart), 'seconds' )

def main():
    print('the end')


if __name__ == '__main__':
    main()
