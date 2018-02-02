from __future__ import division, print_function, absolute_import
import time
import tensorflow as tf
import numpy as np
from keras.layers import Dense
#%matplotlib inline

# Import data
data = np.diag([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

# Parameters
learning_rate = 0.01
training_epochs = 500
batch_size = 256
display_step = 1
examples_to_show = 10
tol = 0.01

# Network Parameters
n_hidden_1 = 4 #1st layer num features
n_input = 16 # data input (matrix shape: 16*16)

# initial run time
start = time.time()

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),

    'decoder_b1': tf.Variable(tf.random_normal([n_input]))
}
# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    encoded = Dense(n_hidden_1, activation = 'relu')(data)
    return encoded

# Building the d ecoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    decoded = Dense(16, activation = 'relu')
    return decoded

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer

print(encoder(X))
print("Run time is {.3f}".format(time.time() - start))
