"""
Training a DNN using TensorFlow's lower-level API
-------------------------------------------------

Problem: use DNN to classify digits using MNIST dataset (a classical problem :v)

Implement a DNN with the following structures:
    - 2 hidden layers: one with 300 neurons, the other with 100 neurons
    - ReLU activation function
    - Softmax output layer
    - Using Mini-batch Gradient Descent for optimization
"""

import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data


def reset_tf_graph(seed=42):
    """Reset TensorFlow default graph and set seed number for numpy"""
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


"""CONSTRUCTION PHASE"""

n_inputs = 28 * 28  # each image in MNIST dataset is 28x28 pixels
n_hidden1 = 300     # the number of neurons in the first hidden layer
n_hidden2 = 100     # the number of neurons in the second hidden layer
n_outputs = 10      # the number of neurons in the output layers (10 as we're classifying ten digits)

reset_tf_graph()

with tf.name_scope("input"):
    # Create placeholder nodes to represent training data and the targets (labels)

    # The shape of X is partially defined because we don't know how many training instances
    # will be used. X is a 2D tensor (or matrix) with instances along the first dimension
    # and features along the second dimension. The total number of features is 28*28 (784) as the size of the image.
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

    # Similar to X, we don't know the shape of y because we don't know exactly the number of
    # training instances will be fed in each batch during Mini-batch Gradient Descent.
    y = tf.placeholder(tf.int64, shape=None, name="y")


def variable_summaries(var, name):
    """Attach summaries to a Tensor"""
    with tf.name_scope(name + '-summary'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Create the neural network layer
# X is now the input layer. During the execution phase, it'll be replaced with one training batch
# at a time (all instances inside the batch will be processed simultaneously by the network).
# We need to create two hidden layers (differ by the number of neurons - inputs). The output layer
# is the same as hidden layer except that it uses 'softmax' activation function rather than 'ReLU'
# activation function.

def neuron_layer(X, n_neurons, name, activation=None):
    """Create a layer with specified number of neurons and activation function"""

    # Create a name scope using the name of the layer which contains all the computation nodes
    # for this layer. Name scope is not required but it makes the visualization process
    # inside TensorBoard be well organized.
    # Ref: https://www.tensorflow.org/api_docs/python/tf/name_scope
    with tf.name_scope(name=name):
        # Get the number of inputs using the input matrix's shape (the second dimension)
        n_inputs = int(X.get_shape()[1])

        # Create W variable which holds the weights matrix (layer's kernel).
        # It's 2D tensor containing all the connection weights between each input and each neuron
        # so its shape will be (n_inputs, n_neurons).
        #
        # W is initialized randomly using 'truncated normal' (Gaussian) distribution with the standard
        # deviation of 2/sqrt(n_inputs). Using this specific std makes the algorithm converge much faster
        # (explained vanishing and exploding gradient problems)
        #
        # Using a truncated normal rather than a regular normal distribution to prevent generating large weights
        # which could slow down the training process.
        #
        # We need to randomly initialize connection weights for all hidden layers in order to avoid any symmetries
        # that could make GD algorithm unable to break. For instance, if all connection weights are initialized to 0,
        # then all neurons will output 0, and the error gradient will be the same for all neurons in a hidden layer.
        # The GD step will update all weights in the same way for each layer, so the connection weights of all neurons
        # remain equal. Having hundreds of neurons in a layer is meaningless in this situation.
        #
        # Ref: https://www.tensorflow.org/api_docs/python/tf/truncated_normal
        #      https://www.tensorflow.org/api_docs/python/tf/matmul
        with tf.name_scope("weights"):
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name="kernel")
            variable_summaries(W, "kernel")

        # Create b variable for biases (initialized to 0)
        with tf.name_scope("biases"):
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")
            variable_summaries(b, "bias")

        # Create a sub-graph to compute the weighted sums of the inputs (plus the bias term)
        # for each and every neuron in the layer and for all instances in the batch.
        # Here we're using a vectorized implementation for efficiency (as TensorFlow will take
        # care of CPU & GPU utilization for operation like matrix multiplication).
        with tf.name_scope("Wx_plus_b"):
            Z = tf.matmul(X, W) + b

        # Apply the specified activation function on the weighted sums if available.
        if activation is not None:
            with tf.name_scope("activation"):
                act_Z = activation(Z)
                tf.summary.histogram('activations', act_Z)
                return act_Z
        else:
            tf.summary.histogram('pre_activations', Z)
            return Z


# Build the deep neural network
# The first hidden layer takes X as its input.
# The second hidden layer takes the output from the first hidden layer as its input
# The ouput layer takes the output of the second hidden layer as its input.

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X=X, n_neurons=n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = neuron_layer(X=hidden1, n_neurons=n_hidden2, name="hidden2", activation=tf.nn.relu)

    # Logit is the output of neural network before going through the softmax activation function.
    logits = neuron_layer(X=hidden2, n_neurons=n_outputs, name="outputs")

# We can use TensorFlow's built-in layer construction functions instead of defining them manually.
# tf.layers.dense function create a fully connected layer where all the inputs are connected to all
# the neurons in the layer. Weights, biases and activation are took care of internally.
# with tf.name_scope("dnn"):
#     hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
#     hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
#     logits = tf.layers.dense(hidden2, n_outputs, name="outputs")


# Define the cost function used to train the DNN
# We're going to use cross entropy cost function (penalize models that estimate a low probability
# for the target class)
with tf.name_scope("cross_entropy"):
    # Compute cross entropy based on the logits (the output of the network before going through
    # the softmax activation function). Labels should be in form of integers ranging from 0 to
    # the number classes-1. For MNIST dataset, labels will be the range [0, 9].
    # sparse_softmax_cross_entropy_with_logits() returns a 1D tensor containing the cross entropy
    # for each instance. This function is equivalent to applying softmax activation function then
    # computing the cross entropy, but it's more efficient and convenient.
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    # Compute mean cross entropy over all instances.
    with tf.name_scope("total"):
        loss = tf.reduce_mean(xentropy, name="loss")
tf.summary.scalar('cross_entropy', loss)


# Applying Gradient Descent Optimizer to tweak model parameters and minimize the cost function
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)


# Evaluate the model by calculating accuracy score
# We're going to use accuracy as performance measure
with tf.name_scope("evaluation"):
    # Determine if the neural network's prediction is correct by checking whether
    # or not the highest logit corresponds to the target class. in_top_k() function
    # returns 1D tensor of boolean values, so we need to cast these booleans to
    # floats and then compute the average to get the overall accuracy.
    with tf.name_scope("correct_predicion"):
        correct = tf.nn.in_top_k(logits, y, 1)
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


"""EXECUTION PHASE"""

# Load the MNIST dataset
mnist = input_data.read_data_sets("/tmp/data")

n_epochs = 100    # The number of training iterations
batch_size = 100  # The size of each mini batch

# The name of summary directory for each run of this file
summary_dir = "{}/run-{}".format('summary', datetime.utcnow().strftime("%Y%m%d%H%M%S"))

# Merge all the summaries and write them out to the log dir
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summary_dir + '/train', tf.get_default_graph())
test_writer = tf.summary.FileWriter(summary_dir + '/test', tf.get_default_graph())

# Initialize all variables and create a saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Train the define model
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        # Load the next mini batch and execute the training operation on that batch.
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            train_summary, _ = sess.run([merged, training_op], feed_dict={X: X_batch, y: y_batch})
            if iteration % 10 == 0:
                train_writer.add_summary(train_summary, epoch * n_batches + iteration)

        # Compute accuracy score on the last mini batch
        accuracy_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

        # Compute accuracy score on the full training set
        test_summary, accuracy_test = sess.run([merged, accuracy],
                                               feed_dict={X: mnist.test.images, y: mnist.test.labels})
        test_writer.add_summary(test_summary, epoch)

        # Log the train and test accuracy scores for each epoch
        print(epoch, "Train accuracy:", accuracy_train, "Test accuracy:", accuracy_test)

    # Save the model parameters
    save_path = saver.save(sess, "./saved/trained_dnn.ckpt")


# Using trained DNN
with tf.Session() as sess:
    # Load the model parameters from disk
    saver.restore(sess, "./saved/trained_dnn.ckpt")

    # Scaled images needed to classify
    X_new_scaled = mnist.test.images[:20]

    # Evaluate the logits node again
    Z = logits.eval(feed_dict={X: X_new_scaled})

    # Pick the highest logit value to get the predicted classes
    y_pred = np.argmax(Z, axis=1)

    # Print predicted classes and actual classes
    print("Predicted classes:", y_pred)
    print("Actual classes:", mnist.test.labels[:20])

reset_tf_graph()
train_writer.close()
test_writer.close()
