import theano
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import time

start = time.time()
# Initialize
encode_dim = 5
input_data = Input(shape=(16,))

# build up the autoencoder first
encoded = Dense(encode_dim, activation = 'relu')(input_data)
#encoded1 = Dense(8, activation = 'relu')(encoded)
#encoded2 = Dense(4, activation = 'relu')(encoded2)
decoded = Dense(16, activation = 'relu')(encoded)
autoencoder = Model(input = input_data, output = decoded)

# build up the encoder
encoder = Model(input = input_data, output = encoded)

# build up the decoder
encoded_input = Input(shape=(encode_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

x_train = np.eye(16)
x_test = np.eye(16)

autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')
autoencoder.fit(x_train, x_train, nb_epoch=4, batch_size=50)

encoded_res = encoder.predict(x_test)
#decoded_res = decoder.predict(encoded)

print(encoded_res)
#print(decoded_res)
print('running time is {}'.format(time.time() - start))
