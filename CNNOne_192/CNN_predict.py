from keras.models import Sequential, Model
from keras.layers.core import Dropout,Activation,Flatten
from keras.layers import Input, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers import concatenate
from keras import optimizers
#from keras.utils import  np_utils
from keras import backend as K
from keras.models import load_model
K.set_image_data_format('channels_first')
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from math import nan

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as error
from sklearn.metrics import r2_score

#loading the LBM functions and velocity output
Re = np.load("Re_range.npy")
feq = np.load("feq_initial.npy")
fun = np.load("f_final.npy")
vel = np.load("u_final.npy")

Re_test = 7500; 



num = vel.shape[0]
print("Shape of inputs : ")
print("Re : "+ str(Re.shape))
print("feq : "+ str(feq.shape))
print("fun : "+ str(fun.shape))
print("vel : "+ str(vel.shape))

print("Number of training samples is " + str(num))
print("Original resolution of input/output is " + str(vel.shape[2:]))

#decreasing the resolution by half to make it easier to train
# first removing middle rows and columns
feq = np.delete(feq,int(num/2),axis=1);feq = np.delete(feq,int(num/2),axis=2)
fun = np.delete(fun,int(num/2),axis=2);fun = np.delete(fun,int(num/2),axis=3)
vel = np.delete(vel,int(num/2),axis=2);vel = np.delete(vel,int(num/2),axis=3)
# choosing alternate rows and columns including boundaries
feq = feq[:,::2,::2] ; fun = fun[:,:,2,::2] ; vel = vel[:,:,::2,::2]


print("Current resolution of input/output is " + str(vel.shape[2:]))

vel_lbm = vel[np.array(np.where(Re==5000))[0]]

vel_lbm = vel_lbm.reshape(2,vel.shape[-2], vel.shape[-1])
# Converting the max value to 1
#saving max values for later reference
Re_max = np.max(Re);feq_max = np.max(feq);fun_max = np.max(fun);
vel_add = np.max(vel) # addign max vel to whole array to make it all positive

Re = Re/Re_max ; feq = feq/feq_max ; fun = fun/fun_max;
vel[:] = vel[:] + vel_add; vel_max = np.max(vel); vel = vel/vel_max

# print("testing min and max of Re " + str(np.max(Re)) + ' ' + str(np.min(Re)))
# print("testing min and max of feq " + str(np.max(feq)) + ' ' + str(np.min(feq)))
# print("testing min and max of fun " + str(np.max(fun)) + ' ' + str(np.min(fun)))
# print("testing min and max of vel " + str(np.max(vel)) + ' ' + str(np.min(vel)))

#ensuring compatibility with GPUs
Re = Re.astype('float32') ; feq = feq.astype('float32') ;fun = fun.astype('float32') 
vel = vel.astype('float32') 

feq = np.asarray([feq]*Re.shape[0])
fnet = np.append(feq,np.zeros((feq.shape[0],1,feq.shape[-2],feq.shape[-1])),axis=1)

Re_test = Re_test/Re_max; Re_test = Re_test.astype('float32')
fnet_test = np.empty((1,fnet.shape[1],fnet.shape[-2],fnet.shape[-1]))
vel_test = np.empty((1,vel.shape[1],vel.shape[-2],vel.shape[-1]))

fnet_test[0,:,:,:] = fnet[0,:,:,:]
fnet_test[0,-1,:,:] = Re_test

for i in np.arange(feq.shape[0]):
    fnet[i,-1,:,:] = Re[i]  

modelx = load_model('cnn1_x1e5.h5')
modely = load_model('cnn1_y2e5.h5')


velx = modelx.predict(fnet_test)   
vely = modely.predict(fnet_test)

print(velx.shape)
print(vely.shape)
print(vel_test.shape)


vel_test = vel_test.reshape(2,feq.shape[-2],feq.shape[-1])

vel_test[0] = velx.reshape(1,feq.shape[-2],feq.shape[-1])
vel_test[1] = vely.reshape(1,feq.shape[-2],feq.shape[-1])

vel_test = vel_test*vel_max; vel_test[:] = vel_test[:] - vel_add
#vel[:] = vel[:] + vel_add; vel_max = np.max(vel); vel = vel/vel_max

Re_test = Re_test*Re_max
uLB = vel_add
u = np.copy(vel_test)
u_lbm = np.copy(vel_lbm)
print('the max of output vel is '+str(np.max(u)))
print('the max of input vel BC was '+str(vel_add))

xsize = vel.shape[-2];ysize=vel.shape[-1];xsize_max=xsize-1;ysize_max=ysize-1
YNorm = np.arange(ysize,0,-1,dtype='float64')/ysize # y starts as 0 from top lid
XNorm = np.arange(0,xsize,1,dtype='float64')/xsize # x starts as 0 from left wall

# Ghia Data for Re 100 to Re 10000
GhiaData = np.genfromtxt('GhiaData.csv', delimiter = ",")[6:23,1:]
GhiaDataV = np.genfromtxt('GhiaData.csv',delimiter = ",")[25:39,2:9]
X_GhiaHor = GhiaData[:,9]
Y_GhiaVer = GhiaData[:,0]

Re_test = int(round(Re_test))
Re_dict = {100:1, 400:2, 1000:3, 3200:4, 5000:5, 7500:6, 10000:7} # column numbers in csv file for Re
Ux_GhiaVer = GhiaData[:,Re_dict[Re_test]]
Uy_GhiaHor = GhiaData[:,Re_dict[Re_test]+9]
# Positions of vortices
X_Vor = GhiaDataV[0:7,Re_dict[Re_test]-1]
X_Vor = X_Vor[X_Vor!=0]
Y_Vor = GhiaDataV[7:14,Re_dict[Re_test]-1]
Y_Vor = Y_Vor[Y_Vor!=0]

NormErr_column = []; time_column = [] # initializing arrays to track LBM and Ghia differences
temp = np.array(Y_GhiaVer*(ysize_max)).astype(int) # LBM coordinates close to ghia's values
LBMy = temp[1:] #avoiding zero values at wall

f = pyplot.figure(figsize=(30,16))
f.clear()
subplot1 = pyplot.subplot2grid((2,25),(0,0), colspan=9, rowspan=1)
subplot2 = pyplot.subplot2grid((2,25),(0,12), colspan=9, rowspan=1)
subplot3 = pyplot.subplot2grid((2,25),(1,0), colspan=9, rowspan=1)
subplot4 = pyplot.subplot2grid((2,25),(1,12), colspan=9, rowspan=1)

Usquare = u[0,:,:]**2 + u[1,:,:]**2
Usquare = Usquare/(uLB**2)
BCoffset = int(xsize/20)
offset2 = int(xsize/20) #to escape already located vortices
# replacing all boundaries with nan to get location of vortices
Usquare[0:BCoffset,:] = nan ; Usquare[:,0:BCoffset] = nan
Usquare[xsize_max-BCoffset:xsize,:] = nan;Usquare[:,ysize_max-BCoffset:ysize] = nan
Loc1 = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)
# finding other vortices
Usquare[Loc1[0]-offset2:Loc1[0]+offset2,Loc1[1]-offset2:Loc1[1]+offset2] = nan
Loc2 = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)
Usquare[Loc2[0]-offset2:Loc2[0]+offset2,Loc2[1]-offset2:Loc2[1]+offset2] = nan
Loc3 = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)
Usquare[Loc3[0]-offset2:Loc3[0]+offset2,Loc3[1]-offset2:Loc3[1]+offset2] = nan
Loc4 = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)

Usquare = u_lbm[0,:,:]**2 + u_lbm[1,:,:]**2
Usquare = Usquare/(uLB**2)
BCoffset = int(xsize/20)
# replacing all boundaries with nan to get location of vortices
Usquare[0:BCoffset,:] = nan ; Usquare[:,0:BCoffset] = nan
Usquare[xsize_max-BCoffset:xsize,:] = nan;Usquare[:,ysize_max-BCoffset:ysize] = nan
Loc1_lbm = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)
# finding other vortices
Usquare[Loc1_lbm[0]-offset2:Loc1_lbm[0]+offset2,Loc1_lbm[1]-offset2:Loc1_lbm[1]+offset2] = nan
Loc2_lbm = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)
Usquare[Loc2_lbm[0]-offset2:Loc2_lbm[0]+offset2,Loc2_lbm[1]-offset2:Loc2_lbm[1]+offset2] = nan
Loc3_lbm = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)
Usquare[Loc3_lbm[0]-offset2:Loc3_lbm[0]+offset2,Loc3_lbm[1]-offset2:Loc3_lbm[1]+offset2] = nan
Loc4_lbm = np.unravel_index(np.nanargmin(Usquare),Usquare.shape)

print(Loc1)
print(Loc2)
print(Loc3)
print(Loc4)
print('---')
print(Loc1_lbm)
print(Loc2_lbm)
print(Loc3_lbm)
print(Loc4_lbm)


color1 = (np.sqrt(u_lbm[0,:,:]**2+u_lbm[1,:,:]**2)/uLB ).transpose()
strm = subplot1.streamplot(XNorm,YNorm,(u_lbm[0,:,:]).transpose(),(u_lbm[1,:,:]).transpose(), color =color1,cmap=pyplot.cm.jet)#,norm=matplotlib.colors.Normalize(vmin=0,vmax=1)) 
cbar = pyplot.colorbar(strm.lines, ax = subplot1)
subplot1.plot(Loc1_lbm[0]/xsize,(ysize_max-Loc1_lbm[1])/ysize,'mo', label='Vortex1')
subplot1.plot(Loc2_lbm[0]/xsize,(ysize_max-Loc2_lbm[1])/ysize,'mo', label='Vortex2')
subplot1.plot(Loc3_lbm[0]/xsize,(ysize_max-Loc3_lbm[1])/ysize,'mo', label='Vortex2')
subplot1.plot(Loc4_lbm[0]/xsize,(ysize_max-Loc4_lbm[1])/ysize,'mo', label='Vortex2')
subplot1.plot(X_Vor, Y_Vor,'rs', label='Ghia',alpha=1)
subplot1.set_title('Velocity Streamlines - LBM', fontsize = 23, y =1.02)
subplot1.margins(0.005) #subplot3.axis('tight')
subplot1.set_xlabel('X-position', fontsize = 20);subplot1.set_ylabel('Y-position', fontsize = 20)

color2 = (np.sqrt(u[0,:,:]**2+u[1,:,:]**2)/uLB ).transpose()
strm = subplot2.streamplot(XNorm,YNorm,(u[0,:,:]).transpose(),(u[1,:,:]).transpose(), color =color2,cmap=pyplot.cm.jet)#,norm=matplotlib.colors.Normalize(vmin=0,vmax=1)) 
cbar = pyplot.colorbar(strm.lines, ax = subplot2)
subplot2.plot(Loc1[0]/xsize,(ysize_max-Loc1[1])/ysize,'mo', label='Vortex1')
subplot2.plot(Loc2[0]/xsize,(ysize_max-Loc2[1])/ysize,'mo', label='Vortex2')
subplot2.plot(Loc3[0]/xsize,(ysize_max-Loc3[1])/ysize,'mo', label='Vortex2')
subplot2.plot(Loc4[0]/xsize,(ysize_max-Loc4[1])/ysize,'mo', label='Vortex2')
subplot2.plot(X_Vor, Y_Vor,'rs', label='Ghia', alpha=1)
subplot2.set_title('Velocity Streamlines - CNN Prediction', fontsize = 23, y =1.02)
subplot2.margins(0.005) #subplot3.axis('tight')
subplot2.set_xlabel('X-position', fontsize = 20);subplot2.set_ylabel('Y-position', fontsize = 20)

matplotlib.rcParams.update({'font.size': 15})
Ux = u[0,int(xsize/2),:]/uLB
Ux_lbm =  u_lbm[0,int(xsize/2),:]/uLB
subplot3.plot(Ux, YNorm, label="CNN")
subplot3.plot(Ux_lbm, YNorm, ':m',label="LBM", alpha=1)
subplot3.plot(Ux_GhiaVer, Y_GhiaVer, 'r*' , label="Ghia")
subplot3.set_title('Ux on middle column - LBM and CNN', fontsize = 20, y =1.02)
subplot3.legend(loc = 'center right')
subplot3.set_xlabel('Ux', fontsize = 20);subplot3.set_ylabel('Y-position', fontsize = 20)


Uy = u[1,:,int(ysize/2)]/uLB
Uy_lbm =  u_lbm[1,:,int(ysize/2)]/uLB
subplot4.plot(XNorm, Uy, label="CNN")
subplot4.plot(XNorm, Uy_lbm, ':m',label="LBM", alpha=1)
subplot4.plot(X_GhiaHor,Uy_GhiaHor, 'r*', label='Ghia')
subplot4.set_title('Uy on middle row - LBM and CNN', fontsize = 20, y=1.02)
subplot4.legend(loc = 'upper right')
subplot4.set_xlabel('X-position', fontsize = 20);subplot4.set_ylabel('Uy', fontsize = 20)


f.suptitle('Comparison of LBM and CNN Prediction for LDC at Re '+str(int(round(Re_test)))+' for grid size of '+str(xsize)+'*'+str(ysize), fontsize = 30, x = 0.44, y =0.97)
pyplot.savefig('CNN_predict' + "_Re" + str(int(round(Re_test))) + ".png",bbox_inches = 'tight', pad_inches = 0.4)


# 
# print(y_p.shape)
# #print((y_p[5,10:15,10:15,0]))
# #print((y_test[5,10:15,10:15,0]))
# y_rand = np.random.uniform(low=0.01, high=0.1, size=y_p.shape)
# 
# print(y_rand.shape)
# print(y_rand.shape[0])
# print(y_rand.shape[1])
# print(y_rand.shape[2])
# 
# y_test = y_test.reshape(y_p.shape[0],y_p.shape[1]*y_p.shape[2])
# y_p = y_p.reshape(y_p.shape[0],y_p.shape[1]*y_p.shape[2])
# y_rand = y_rand.reshape(y_rand.shape[0],y_rand.shape[1]*y_rand.shape[2])
# 
# print(r2_score(y_test,y_p))
# print(r2_score(y_test,y_rand))
# 
# 
# 
#       