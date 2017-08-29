"""
Nonsaturating Activation Functions
----------------------------------
Vanishing/Exploding gradients problems can happen because of poor choice of activation function.

ReLU activation is an good choice for activation (compared to signmoid function) because it doesn't saturate for
positive class and faster to compute.

Problem of ReLU:
    - Dying ReLUs problem: neurons keep outputting 0 during training process. If a neuron's weights is updated such
      that the weighted sumof the neuron's inputs is negative, the output will be 0 -> gradient = 0 (negative input)

ReLU variants
-------------
* LeakyReLU(z) = max(alpha * z, z)
Ref: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs
Ref: https://arxiv.org/pdf/1505.00853.pdf
alpha hyperparameter defines the amount of "leaks" or the slope of the function when z < 0
The slope ensures LeakyReLU never die and always have chance to wake up -> resolve the 'dying ReLU problem'

RReLU - Randomized Leaky ReLU is similar to LeakyReLU but alpha is picked randomly in a given range during training,
and is fixed to an average value during testing. It reduces the risk of overfitting the training set (acting like a
regularizer)

PReLU - Parametric Leaky ReLU is another variant of ReLU where alpha is learned during training process rather than
be defined beforehand as a hyperparameter. It outperforms ReLU on large image datasets. However, it may overfit the
trainning set when the dataset is small.

* ELU - Exponential Linear Unit
Ref: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#ELUs
This activation function seems to outperform all variants of ReLU, reduce training time, and perform better on test set
Advantages over ReLU function:
    - It outputs negative values when z < 0 -> average output closer to 0 -> prevent vanishing graidents problem.
      alpha defines the value that the ELU function'll approach when z is a large negative number (usually set to 1)
    - Has nonzero graident for z < 0 -> prevent dying neurons problem
    - It's smooth everywhere -> speed up Gradient Descent

Main drawback of ELU activation function is that it's slower to compute than ReLU and associated variants (because ELU
uses exponential function)

How to choose activation function?
----------------------------------
Generally speaking: ELU > leaky ReLU (and its variants) > ReLU > tanh > logistic (signmoid)
Better runtim performance then leaky ReLUs > ELUs
Use the default value of alpha rather than tweaking it (0.01 for leaky ReLU, 1 for ELU)
If not enough time of computing power -> use cross-validation to evaluate other activation functions
(RReLU if your network is overfitting, PReLU if training set is huge)

ELU function is built-in in TensorFlow - tf.nn.elu
For leaky ReLU, we can define our own function :D
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


z = np.linspace(-5, 5, 200)


def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha * z, z)


def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)


plt.figure(figsize=(11, 4))
plt.subplot(121)
plt.plot(z, leaky_relu(z, 0.05), "b--", linewidth=2)
plt.plot([-5, 5], [0, 0], "k-")
plt.plot([0, 0], [-0.5, 4.2], "k-")
plt.grid(True)
props = dict(facecolor="black", shrink=0.1)
plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
plt.title("Leaky ReLU activation function", fontsize=14)
plt.axis([-5, 5, -0.5, 4.2])

plt.subplot(122)
plt.plot(z, elu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1, -1], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
props = dict(facecolor='black', shrink=0.1)
plt.title(r"ELU activation function ($\alpha=1$)", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

# plt.show()


"""Using leaky ReLU and ELU for activation function in TensorFlow"""


def tf_leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)


tf.reset_default_graph()
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    # hidden1 = tf.layers.dense(X, n_hidden1, activation=tf_leaky_relu, name="hidden1")
    # hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf_leaky_relu, name="hidden2")
    # logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.elu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xen = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xen, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()


n_epochs = 20
batch_size = 100
mnist = input_data.read_data_sets("/tmp/data/")

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(mnist.test.labels) // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
