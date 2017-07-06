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

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as error
from sklearn.metrics import r2_score

#loading the LBM functions and velocity output
Re = np.load("Re_range.npy")
feq = np.load("feq_initial.npy")
fun = np.load("f_final.npy")
vel = np.load("u_final.npy")

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

Re_test = np.array([2575]) ; 

Re_test = Re_test/Re_max; Re_test = Re_test.astype('float32')
fnet_test = np.empty((Re_test.shape[0],fnet.shape[1],fnet.shape[-2],fnet.shape[-1]))
vel_test = np.empty((Re_test.shape[0],vel.shape[1],vel.shape[-2],vel.shape[-1]))

for i in np.arange(Re_test.shape[0]):
    fnet_test[i,:,:,:] = fnet[i,:,:,:]
    fnet_test[i,-1,:,:] = Re_test[i]

for i in np.arange(feq.shape[0]):
    fnet[i,-1,:,:] = Re[i]  

modelx = load_model('cnn1_y2e5.h5')
modely = load_model('cnn1_x1e5.h5')


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
print('the max of output vel is '+str(np.max(u)))
print('the max of input vel BC was'+str(vel_add))

xsize = vel.shape[-2];ysize=vel.shape[-1]
YNorm = np.arange(ysize,0,-1,dtype='float64')/ysize # y starts as 0 from top lid
XNorm = np.arange(0,xsize,1,dtype='float64')/xsize # x starts as 0 from left wall

u[0,:,:] = u[0,:,:].transpose()
u[1,:,:] = u[1,:,:].transpose()
f = pyplot.figure(figsize=(30,16))
f.clear()
subplot3 = pyplot.subplot2grid((2,15),(0,10), colspan=5, rowspan=1)
color1 = (np.sqrt(u[0,:,:]**2+u[1,:,:]**2)/uLB ).transpose()
strm = subplot3.streamplot(XNorm,YNorm,(u[0,:,:]).transpose(),(u[1,:,:]).transpose(), color =color1,cmap=pyplot.cm.jet)#,norm=matplotlib.colors.Normalize(vmin=0,vmax=1)) 
#cbar = pyplot.colorbar(strm.lines, ax = subplot3)
subplot3.set_title('Velocity Streamlines - LBM', fontsize = 20, y =1.02)
subplot3.margins(0.005) #subplot3.axis('tight')
subplot3.set_xlabel('X-position', fontsize = 20);subplot3.set_ylabel('Y-position', fontsize = 20)
f.suptitle('CNN Prediction for Lid Driven Cavity - Re '+str(Re_test)+'size of '+str(xsize)+'*'+str(ysize), fontsize = 30, y =1.04)
pyplot.savefig('CNN_predict' + "_Re" + str(Re_test[0]).zfill(5) + ".png",bbox_inches = 'tight', pad_inches = 0.4)


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