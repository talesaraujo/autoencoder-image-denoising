# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import TensorBoard

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize,rescale
from os import listdir

def build_data():
    
    x_train = []
    
    for file in listdir('kdd/train'):
    
        img = imread('kdd/train/' + file)
        
        img = resize(img,(260,540),True)
        
        img = img.reshape(1,img.shape[0],img.shape[1])
    
        
        x_train.append(img)
        
    
    x_train = tuple(x_train)
    
    x_train = np.concatenate(x_train)
    
    y_train = [] 
    
    
    for file in listdir('kdd/train_cleaned'):
    
        img = imread('kdd/train_cleaned/' + file)
        
        img = resize(img,(260,540),True)
        
        y_train.append(img.reshape(1,img.shape[0],img.shape[1]))
    
    
    
    y_train = tuple(y_train)
    
    y_train = np.concatenate(y_train)
    
    x_test = []
    
    for file in listdir('kdd/test'):
    
        img = imread('kdd/test/' + file)
        
        img = resize(img,(260,540),True)
        
        x_test.append(img.reshape(1,img.shape[0],img.shape[1]))
    
    
    
    x_test = tuple(x_test)
    
    x_test = np.concatenate(x_test)
    
    return x_train,y_train,x_test
    
x_train, y_train, x_test = build_data()

x_train = np.reshape(x_train, (len(x_train), 260, 540, 1))
y_train = np.reshape(y_train, (len(x_train), 260, 540, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 260, 540, 1))  # adapt this if using `channels_first` image data format

x_validation = x_train[121:145]
x_train = x_train[0:120]

y_validation = y_train[121:145]
y_train = y_train[0:120]

input_img = Input(shape=(260,540,1))

downConv = Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
downConv = MaxPooling2D((2,2),padding='same')(downConv)
downConv = Conv2D(32,(3,3),activation='relu',padding='same')(downConv)

encoded = MaxPooling2D((2,2),padding='same')(downConv)



upConv = Conv2D(32,(3,3),activation='relu',padding='same')(encoded)
upConv = UpSampling2D((2,2))(upConv)
upConv = Conv2D(32,(3,3),activation='relu',padding='same')(upConv)
upConv = UpSampling2D((2,2))(upConv)


decoded = Conv2D(1,(3,3), activation='sigmoid', padding='same')(upConv)

autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

autoencoder.fit(x_train, y_train,
                epochs=100,
                batch_size=12,
                shuffle=True,
                validation_data=(x_validation, y_validation),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder/denoising/kdd', histogram_freq=0, write_graph=False)])



denoised_imgs = autoencoder.predict(x_test)

n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(260, 540))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(denoised_imgs[i].reshape(260, 540))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
