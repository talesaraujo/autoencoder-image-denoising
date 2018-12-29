# -*- coding: utf-8

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import TensorBoard

import numpy as np
import matplotlib.pyplot as plt


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(sigma)*eps
def sampling(args):
    
    z_mean,z_log_sigma = args
    
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    
    
    #Reparametrization trick
    epsilon = K.random_normal(shape=(batch,dim))
     
    
    return z_mean + K.exp(0.5*z_log_sigma) * epsilon

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# network hyperparameters
input_shape = (original_dim, )
#intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50

#encoder model
inputs = Input(shape=input_shape,name='encoder_inputs')
x = Dense(512,activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_sigma = Dense(latent_dim, name='z_log_sigma')(x)

z = Lambda(sampling,output_shape=(latent_dim,), name='z')([z_mean,z_log_sigma])



#encode inputs to latent space
encoder = Model(inputs,[z_mean,z_log_sigma,z],name='encoder')
encoder.summary()

#decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(512,activation='relu')(latent_inputs)
outputs = Dense(original_dim,activation='sigmoid')(x)


#decode latent space samples to outputs
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

outputs = decoder(encoder(inputs)[2])

#VAE Core Power


vae = Model(inputs,outputs,name='vae')


reconstruction_loss = binary_crossentropy(inputs,outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)

vae.compile(optimizer='adam')


vae.summary()

vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test,None))


z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
plt.figure(figsize=(12, 10))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
plt.colorbar()
plt.show()

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()