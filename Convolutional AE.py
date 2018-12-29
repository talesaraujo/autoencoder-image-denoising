# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import TensorBoard

import numpy as np
import matplotlib.pyplot as plt


input_img = Input(shape=(28,28,1))

downConv = Conv2D(16,(3,3),activation='relu',padding='same')(input_img)
downConv = MaxPooling2D((2,2),padding='same')(downConv)
downConv = Conv2D(8,(3,3),activation='relu',padding='same')(downConv)
downConv = MaxPooling2D((2,2),padding='same')(downConv)
downConv = Conv2D(8,(3,3),activation='relu',padding='same')(downConv)
encoded = MaxPooling2D((2,2),padding='same')(downConv)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

upConv = Conv2D(8,(3,3),activation='relu',padding='same')(encoded)
upConv = UpSampling2D((2,2))(upConv)
upConv = Conv2D(8,(3,3),activation='relu',padding='same')(upConv)
upConv = UpSampling2D((2,2))(upConv)
upConv = Conv2D(16,(3,3),activation='relu')(upConv)
upConv = UpSampling2D((2,2))(upConv)

decoded = Conv2D(1,(3,3), activation='sigmoid', padding='same')(upConv)

autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

autoencoder.fit(x_train,x_train,epochs=100,batch_size=128,shuffle=True,validation_data=(x_test,x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()