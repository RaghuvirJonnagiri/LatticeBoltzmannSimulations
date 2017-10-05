from keras.models import Sequential, Model
from keras.layers.core import Dropout,Activation,Flatten
from keras.layers import Input, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras import optimizers
#from keras.utils import  np_utils
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
K.set_image_data_format('channels_first')
import numpy as np
from matplotlib import pyplot
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


#loading the LBM functions and velocity output
Re = np.load("Re_range.npy")
feq = np.load("feq_initial.npy")
fun = np.load("f_final.npy")
vel = np.load("u_final.npy")
velBC = vel.copy(); velBC[:,:,:,1:] = 0; # BC inputs to keep velocities zero and 1 at boundaries
velBCx = velBC[:,0,:,:].reshape(velBC.shape[0],1,velBC.shape[-2],velBC.shape[-1])/np.max(velBC);
velBCy = velBC[:,1,:,:].reshape(velBC.shape[0],1,velBC.shape[-2],velBC.shape[-1])/np.max(velBC);

re_scaler = MinMaxScaler(feature_range=(0.2, 0.7))
feq_scaler = MinMaxScaler(feature_range=(0.2, 0.7))
vel_scaler = MinMaxScaler(feature_range=(0.2, 0.7))

Re_scaled = re_scaler.fit_transform(Re.reshape(Re.shape[0],1))
feq_scaler.fit(feq.ravel()) 
vel_scaler.fit(vel.ravel())

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
# feq = np.delete(feq,int(num/2),axis=1);feq = np.delete(feq,int(num/2),axis=2)
# fun = np.delete(fun,int(num/2),axis=2);fun = np.delete(fun,int(num/2),axis=3)
# vel = np.delete(vel,int(num/2),axis=2);vel = np.delete(vel,int(num/2),axis=3)
# # choosing alternate rows and columns including boundaries
# feq = feq[:,::2,::2] ; fun = fun[:,:,2,::2] ; vel = vel[:,:,::2,::2]

print("Current resolution of input/output is " + str(vel.shape[2:]))

feq_scaled = feq.copy(); vel_scaled = vel.copy() 
for i in range(feq.shape[0]):
    feq_scaled[i,:] = feq_scaler.transform(feq[i,:])

for i in range(vel.shape[0]):
    vel_scaled[i,0,:] = vel_scaler.transform(vel[i,0,:])
    vel_scaled[i,1,:] = vel_scaler.transform(vel[i,1,:])

# Converting the max value to 1
#saving max values for later reference
Re_max = np.max(Re);Re_min = np.min(Re);feq_max = np.max(feq);fun_max = np.max(fun);
vel_add = np.max(vel) # addign max vel to whole array to make it all positive

Re = Re/Re_max ; feq = feq/feq_max ; fun = fun/fun_max;
vel[:] = vel[:] + vel_add; vel_max = np.max(vel); vel = vel/vel_max

# print("testing min and max of Re " + str(np.max(Re)) + ' ' + str(np.min(Re)))
# print("testing min and max of feq " + str(np.max(feq)) + ' ' + str(np.min(feq)))
# print("testing min and max of fun " + str(np.max(fun)) + ' ' + str(np.min(fun)))
# print("testing min and max of vel " + str(np.max(vel)) + ' ' + str(np.min(vel)))

Re = Re_scaled # ignoring previous arrays and using a scikit scaled version
feq = feq_scaled; vel = vel_scaled

print("Scaled Re of "+str(Re_min)+':'+str(Re_max)+' to '+str(round(np.min(Re),3))+':'+str(np.max(Re)))


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
velBCxtr, velBCxte, velBCytr, velBCyte  = train_test_split(velBCx,velBCy,test_size=0.2, random_state=4) 

print('training Re against 2 velocity components. feq is hardcoded')
print('Shape of x_train, x_test, y_train, y_test :')
print(str(x_train.shape)+' ,'+str(x_test.shape)+' ,'+str(y_train.shape)+' ,'+str(y_test.shape)) 

main_input = Input(shape=(fnet.shape[1],fnet.shape[2],fnet.shape[3]))
aux_input0 = Input(shape=(1,fnet.shape[2],fnet.shape[3]))
aux_input1 = Input(shape=(1,fnet.shape[2],fnet.shape[3]))

xy = 0 #choose 0 if x and 1 if y

layer0a = concatenate([main_input, aux_input0], axis=1)
layer0b = concatenate([layer0a, aux_input1], axis=1)
layer1a = Convolution2D(16, (2, 2),strides=1, padding = 'same')(layer0b); layer1a = LeakyReLU()(layer1a) #layer1b = BatchNormalization(axis=1)(layer1b)
layer1b = Convolution2D(16, (4, 4),strides=1, padding = 'same')(layer0b);layer1b = LeakyReLU()(layer1b)  #layer1b = BatchNormalization(axis=1)(layer1b)
layer1c = Convolution2D(16, (8, 8),strides=1, padding = 'same')(layer0b);layer1c = LeakyReLU()(layer1c)  #layer1c = BatchNormalization(axis=1)(layer1c)
layer1d = Convolution2D(16, (12, 12),strides=1, padding = 'same')(layer0b);layer1d = LeakyReLU()(layer1d)  #layer1d = BatchNormalization(axis=1)(layer1d)
layer2a = concatenate([layer1a,layer1b], axis=1); 
layer2b = concatenate([layer2a,layer1c], axis=1); 
layer2c = concatenate([layer2b,layer1d], axis=1); 
layer3 = Convolution2D(16, (2, 2),strides=2)(layer2c);layer3 = LeakyReLU()(layer3)  #layer3 = BatchNormalization(axis=1)(layer3)
layer4 = Convolution2D(64, (4, 4),strides=4)(layer3);layer4 = LeakyReLU()(layer4)  #layer4 = BatchNormalization(axis=1)(layer4)
layer5 = (Convolution2D(128, (3, 3),strides=3)(layer4));layer5 = LeakyReLU()(layer5)  #layer5 = BatchNormalization(axis=1)(layer5)
layer6 = (Convolution2D(256, (4, 4), strides=4)(layer5)); layer6 = LeakyReLU()(layer6) #layer6 = BatchNormalization(axis=1)(layer6)
layer7 = (Convolution2D(512, (4, 4))(layer6)); layer7 = LeakyReLU()(layer7) #layer7 = BatchNormalization(axis=1)(layer7)

if xy == 0:
    layer8a = (Conv2DTranspose(512,(4, 4))(layer7)); layer8a = LeakyReLU()(layer8a) #layer8a = BatchNormalization(axis=1)(layer8a)
    layer9a = (Conv2DTranspose(256,(4, 4),strides=4)(layer8a)); layer9a = LeakyReLU()(layer9a) #layer9a = BatchNormalization(axis=1)(layer9a)
    layer10a = (Conv2DTranspose(128,(3, 3), strides=3)(layer9a)); layer10a = LeakyReLU()(layer10a) #layer10a = BatchNormalization(axis=1)(layer10a)
    layer11a = (Conv2DTranspose(64,(3, 3), strides=1,padding = 'same')(layer10a)); layer11a = LeakyReLU()(layer11a) # layer11a = BatchNormalization(axis=1)(layer11a)
    layer12a = (Conv2DTranspose(32, (2, 2), strides=2)(layer11a)); layer12a = LeakyReLU()(layer12a) #layer12a = BatchNormalization(axis=1)(layer12a)
    layer13a = (Conv2DTranspose(16, (2, 2), strides=2)(layer12a)); layer13a = LeakyReLU()(layer13a) # layer13a = BatchNormalization(axis=1)(layer13a)
    layer14a =(Conv2DTranspose(2, (2, 2), strides=2)(layer13a)) ; layer14a = LeakyReLU()(layer14a) #layer14a = BatchNormalization(axis=1)(layer14a)
    layer15a = concatenate([main_input, layer14a], axis=1); 
    layer16a = concatenate([aux_input0, layer15a], axis=1); 
    layer17a = (Convolution2D(50, (1, 1))(layer16a)); layer17a = LeakyReLU()(layer17a) #layer17a = BatchNormalization(axis=1)(layer17a)
    outputx = (Convolution2D(1,(1, 1))(layer17a));  
    outputxy = [outputx, outputx]

if xy == 1:
    layer8b = (Conv2DTranspose(512,(4, 4))(layer7)); layer8b = LeakyReLU()(layer8b) #layer8b = BatchNormalization(axis=1)(layer8b)
    layer9b = (Conv2DTranspose(256,(4, 4),strides=4)(layer8b)); layer9b = LeakyReLU()(layer9b) #layer9b = BatchNormalization(axis=1)(layer9b)
    layer10b = (Conv2DTranspose(128,(3, 3), strides=3)(layer9b)); layer10b = LeakyReLU()(layer10b) #layer10b = BatchNormalization(axis=1)(layer10b)
    layer11b = (Conv2DTranspose(64,(3, 3), strides=1,padding = 'same')(layer10b)); layer11b = LeakyReLU()(layer11b) #layer11b = BatchNormalization(axis=1)(layer11b)
    layer12b = (Conv2DTranspose(32, (2, 2), strides=2)(layer11b)); layer12b = LeakyReLU()(layer12b) #layer12b = BatchNormalization(axis=1)(layer12b)
    layer13b = (Conv2DTranspose(16, (2, 2), strides=2)(layer12b)); layer13b = LeakyReLU()(layer13b) #layer13b = BatchNormalization(axis=1)(layer13b)
    layer14b =(Conv2DTranspose(2, (2, 2), strides=2)(layer13b)); layer14b = LeakyReLU()(layer14b) # layer14b = BatchNormalization(axis=1)(layer14b)
    layer15b = concatenate([main_input, layer14b], axis=1); 
    layer16b = concatenate([aux_input1, layer15b], axis=1);
    layer17b = (Convolution2D(50, (1, 1))(layer16b)); layer17b = LeakyReLU()(layer17b) #layer17b = BatchNormalization(axis=1)(layer17b)
    outputy = (Convolution2D(1,(1, 1))(layer17b))
    outputxy = [outputy, outputy]

xydict = {0:'x',1:'y'}

print("training "+str(xydict[xy]))
if xy == 0:
    model = Model(inputs=[main_input,aux_input0,aux_input1], outputs=outputxy[xy])
elif xy == 1:
    model = Model(inputs=[main_input,aux_input0,aux_input1], outputs=outputxy[xy])
#optim = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.02)
optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)

model.compile(optimizer=optim, loss='mean_squared_error')
#model.fit(x_train, [y_train[:,0,:,:].reshape((80,1,192,192)), y_train[:,1,:,:].reshape((80,1,192,192))] , epochs=500, batch_size=5,verbose=1,validation_data=(x_test,[y_test[:,0,:,:].reshape((20,1,192,192)),y_test[:,1,:,:].reshape((20,1,192,192))]))
if xy ==0 :
    history = model.fit([x_train, velBCxtr, velBCytr], y_train[:,xy,:,:].reshape(int(0.8*vel.shape[0]),1,feq.shape[-2],feq.shape[-1]), epochs=400, batch_size=20,verbose=1,validation_data=([x_test,velBCxte, velBCyte], y_test[:,xy,:,:].reshape(int(0.2*vel.shape[0]),1,feq.shape[-1],feq.shape[-1])))
elif xy == 1:
    history = model.fit([x_train, velBCxtr, velBCytr], y_train[:,xy,:,:].reshape(int(0.8*vel.shape[0]),1,feq.shape[-2],feq.shape[-1]), epochs=400, batch_size=20,verbose=1,validation_data=([x_test,velBCxte, velBCyte], y_test[:,xy,:,:].reshape(int(0.2*vel.shape[0]),1,feq.shape[-1],feq.shape[-1])))
    
f = pyplot.figure(figsize=(30,16))
pyplot.plot(history.history['loss']);pyplot.plot(history.history['val_loss'])
pyplot.title('model loss - cnn10_'+xydict[xy],fontsize = 40)
pyplot.xlabel('epoch',fontsize = 30) ; pyplot.ylabel('loss',fontsize = 30)
pyplot.xticks(fontsize=20);pyplot.yticks(fontsize=20)
pyplot.legend(['train', 'test'], loc='center right',fontsize = 30); 
pyplot.savefig('cnn10_'+xydict[xy] +'_history' + ".png",bbox_inches = 'tight', pad_inches = 0.4)

model.save('cnn10_'+xydict[xy] +'.h5')
print('model saved as cnn10_'+xydict[xy] +'.h5')

#pyplot.show()



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