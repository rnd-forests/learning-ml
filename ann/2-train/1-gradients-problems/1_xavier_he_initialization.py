"""
Xavier and He Initialization
----------------------------
To prevent vanishing/exploding gradients problems, we need to control the signal flow properly in both directions:
    - Forward (to make predictions)
    - Backward (when backpropagating gradients)

Conditions:
-> The variance of the outputs of each layer should be equal to the variance of its inputs
-> The gradients should have equal variance before and after flowing through a layer in the reversed direction

Xavier initialization (Glorot initialization) (using logistic activation function)
- Using normal distribution with mean 0 and stdv: sigma = sqrt(2 / (n_inputs + n_outputs))
- Using uniform distribution between -r and r with r = sqrt(6 / (n_inputs + n_outputs))

The number of input and output connections for the layer whose weights are being initialized are called
fan-in and fan-out respectively.

Based on Xavier initialization, other intialization techniques have been developed for other activation functions.
Initialization strategy for ReLU (and its variants) is called He initialization.

For other activation functions, refer to this table:
https://www.dropbox.com/s/4acydwm62csjvrp/Screenshot%20from%202017-08-28%2008-36-39.png?dl=0
"""

import tensorflow as tf

X = []
n_hidden1 = 100
# Tensorflow dense() layer uses Xavier initialization (with a uniform distribution) by default.
# To change to the He initialization, we can do as follows:
he_init = tf.contrib.layers.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                          kernel_initializer=he_init, name="hidden1")
