from keras.models import Sequential, Model
from keras.layers.core import Dropout,Activation,Flatten
from keras.layers import Input, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers import concatenate
from keras import optimizers
#from keras.utils import  np_utils
from keras import backend as K
K.set_image_data_format('channels_first')
import numpy as np

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

for i in np.arange(feq.shape[0]):
    
    fnet[i,-1,:,:] = Re[i]  
  
num_batch = 5
num_epoch = 2
#x_conv = 3; y_conv = 3
#num_filters = 30
#x_pool = 2; y_pool=2
x_train, x_test, y_train, y_test = train_test_split(fnet,vel,test_size=0.2, random_state=4)

print('training Re against 2 velocity components. feq is hardcoded')
print('Shape of x_train, x_test, y_train, y_test :')
print(str(x_train.shape)+' ,'+str(x_test.shape)+' ,'+str(y_train.shape)+' ,'+str(y_test.shape)) 

main_input = Input(shape=(fnet.shape[1],fnet.shape[2],fnet.shape[3]))

layer1 = Convolution2D(128, (12, 12),strides=12, activation='relu')(main_input)
layer2 = Convolution2D(256, (4, 4), strides=4, activation='relu')(layer1)
layer3 = Convolution2D(512, (4, 4), activation='relu')(layer2)
layer4a = Conv2DTranspose(512,(8, 8), activation='relu')(layer3)
layer4b = Conv2DTranspose(512,(8, 8), activation='relu')(layer3)
layer5a = Conv2DTranspose(256,(3, 3),strides=3, activation='relu')(layer4a)
layer5b = Conv2DTranspose(256,(3, 3),strides=3, activation='relu')(layer4b)
layer6a = Conv2DTranspose(128,(2, 2), strides=2, activation='relu')(layer5a)
layer6b = Conv2DTranspose(128,(2, 2), strides=2, activation='relu')(layer5b)
layer7a = Conv2DTranspose(32, (2, 2), strides=2, activation='relu')(layer6a)
layer7b = Conv2DTranspose(32, (2, 2), strides=2, activation='relu')(layer6b)
layer8a = Conv2DTranspose(1, (2, 2), strides=2, activation='relu')(layer7a)
layer8b = Conv2DTranspose(1, (2, 2), strides=2, activation='relu')(layer7b)
layer9a = concatenate([main_input, layer8a], axis=1)
layer9b = concatenate([main_input, layer8b], axis=1)
layer10a = Convolution2D(10, (1, 1), activation='relu')(layer9a)
layer10b = Convolution2D(10, (1, 1), activation='relu')(layer9b)
outputx = Convolution2D(1,(1, 1), activation='relu' )(layer10a)
outputy = Convolution2D(1,(1, 1), activation='relu' )(layer10b)

model = Model(inputs=main_input, outputs=outputy)
optim = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.05)
model.compile(optimizer=optim, loss='mean_squared_error')
#model.fit(x_train, [y_train[:,0,:,:].reshape((80,1,192,192)), y_train[:,1,:,:].reshape((80,1,192,192))] , epochs=500, batch_size=5,verbose=1,validation_data=(x_test,[y_test[:,0,:,:].reshape((20,1,192,192)),y_test[:,1,:,:].reshape((20,1,192,192))]))
model.fit(x_train, y_train[:,1,:,:].reshape((80,1,feq.shape[-1],feq.shape[-1])), epochs=500, batch_size=5,verbose=1,validation_data=(x_test, y_test[:,1,:,:].reshape((20,1,feq.shape[-1],feq.shape[-1]))))

model.save('cnn1_y.h5')
print('model saved as cnn1.h5')

#print(" shape of layer1 output is " + str(layer1.get_output_shape_at(0)))
# model = Sequential()
# model.add(Convolution2D(num_filters,x_conv,y_conv, border_mode='same',activation='relu',input_shape=x_train.shape[1:]))
# model.add(Dropout(0.25))
# model.add(Conv2DTranspose(1,x_conv,y_conv, border_mode='same',activation='relu'))
# #model.add(MaxPooling2d(pool_size=(x_pool,y_pool)))
# 
# model.compile(loss='mean_squared_error',optimizer='RMSprop')
# 
# model.fit(x_train,y_train,batch_size=num_batch,epochs=num_epoch,verbose=1,validation_data=(x_test,y_test))
# 
# y_p = model.predict(x_test)   
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