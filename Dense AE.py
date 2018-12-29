# -*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.layers import Input,Dense
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

encoding_dim = 32

input_img = Input(shape=(784,))

encoded = Dense(512,activation='relu')(input_img)
encoded = Dense(256,activation='relu')(encoded)
encoded = Dense(128,activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64,activation='relu')(encoded)
decoded = Dense(128,activation='relu')(decoded)
decoded = Dense(256,activation='relu')(decoded)
decoded = Dense(512,activation='relu')(decoded)
decoded = Dense(784,activation='sigmoid')(decoded)

autoencoder = Model(input_img,decoded)


#Encoder Model
#encoder = Model(input_img,encoded)

#Decoder Model
#encoded_input = Input(shape=(encoding_dim,))
#decoder_layer = autoencoder.layers[-1]
#decoder = Model(encoded_input,decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train,x_train,epochs=100,batch_size=256,shuffle=True,validation_data=(x_test,x_test))

#encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20,4))

for i in range(n):
    
    ax = plt.subplot(2,n,i + 1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
    