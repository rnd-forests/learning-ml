"""
Reuse Models from other Frameworks
----------------------------------
To reuse models from other frameworks, we need to load model parameters manually and assign them to appropriate
TensorFlow variables.
"""
import tensorflow as tf


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)


n_inputs = 2
n_hidden1 = 3


# Weights and biases from model trained in another framework rather than TensorFlow.
# This is weights and biases of one hidden layer.
original_w = [[1., 2., 3.], [4., 5., 6.]]
original_b = [7., 8., 9.]

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")

# Build other hidden layers here, just one hidden layer for demonstration

graph = tf.get_default_graph()

# Every TensorFlow variable has an associated assignment operation (Assign).
# Here we get that operation (variable name + /Assign)
assign_kernel = graph.get_operation_by_name("hidden1/kernel/Assign")
assign_bias = graph.get_operation_by_name("hidden1/bias/Assign")
# Get a handle on each assignment operation's second input. In case of assignment operation,
# the second input is the value which get assigned to the variable (initialization value)
init_kernel = assign_kernel.inputs[1]
init_bias = assign_bias.inputs[1]

# Get a handle on the variables of layer hidden1
with tf.variable_scope("", default_name="", reuse=True):
    hidden1_weights = tf.get_variable("hidden1/kernel")
    hidden1_biases = tf.get_variable("hidden1/bias")
# We can get a handle of the variables using get_collection() function and specifying the scope
tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden1")
# Or we can use get_tensor_by_name() function()
tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
tf.get_default_graph().get_tensor_by_name("hidden1/bias:0")


# Create new placeholders and assignment nodes
original_weights = tf.placeholder(tf.float32, shape=(n_inputs, n_hidden1))
original_biases = tf.placeholder(tf.float32, shape=n_hidden1)
assign_hidden1_weights = tf.assign(hidden1_weights, original_w)
assign_hidden1_biases = tf.assign(hidden1_biases, original_b)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Run the normal initialization process but we feed it the values (weights and biases)
    # that we want to reuse.
    sess.run(init, feed_dict={init_kernel: original_w, init_bias: original_b})

    # We can create new assignment operations and placeholders and
    # use them to set the values of the variables manually. However, this is not necessary.
    # We should use the above approach.
    sess.run(assign_hidden1_weights, feed_dict={original_weights: original_w})
    sess.run(assign_hidden1_biases, feed_dict={original_biases: original_b})

    # Training the model here
    print(hidden1.eval(feed_dict={X: [[10.0, 11.0]]}))
