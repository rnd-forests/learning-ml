"""
Freezing Lower Layers
---------------------
Generally, lower layers of DNN are used to learn low-level features, so we can just reuse these layers as they are.
It's better to freeze their weights (make their weights fixed) when training new DNN which makes higher layers easier
to train.

Solutions:
    - Give the optimizer the list of variables to train excluding variables from lower layers
    - Add stop_gradient() layer in the graph so that any layers below it will be frozen.
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


n_inputs = 784
n_hidden1 = 300  # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new
n_outputs = 10  # new
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")        # reused
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")  # reused
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")  # reused
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")  # new
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")                          # new

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Get the list of variables that will be used when training the model and feed them to the optimizer.
    # Here we get only hidden3, hidden4, and output layers.
    # hidden1 and hidden2 are now frozen layers.
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[34]|outputs")
    # Provide explicitly the variables to the optimizer for training using var_list parameter
    training_op = optimizer.minimize(loss, var_list=train_vars)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Create the saver to restore the first three hidden layers
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden[123]")
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
restore_saver = tf.train.Saver(reuse_vars_dict)

n_epochs = 20
batch_size = 200
mnist = input_data.read_data_sets("/tmp/data/")

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")


# Using stop_gradient() function
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")  # reused frozen
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")  # reused frozen & cached
    hidden2_stop = tf.stop_gradient(hidden2)
    hidden3 = tf.layers.dense(hidden2_stop, n_hidden3, activation=tf.nn.relu, name="hidden3")  # reused, not frozen
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")  # new!
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")  # new!

# Caching the frozen layers
# Because the frozen layers won't change, it's possible to cache the output
# of the TOPMOST frozen layer for each training instance.
n_batches = mnist.train.num_examples // batch_size
with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")
    # Here hidden2 is the topmost frozen layer.
    # We're going to cache its output
    h2_cache = sess.run(hidden2, feed_dict={X: mnist.train.images})
    h2_cache_test = sess.run(hidden2, feed_dict={X: mnist.test.images})

    for epoch in range(n_epochs):
        shuffled_idx = np.random.permutation(mnist.train.num_examples)
        # Build up the batches using output from the hidden layer 2
        # instead of using training instances
        hidden2_batches = np.array_split(h2_cache[shuffled_idx], n_batches)
        y_batches = np.array_split(mnist.train.labels[shuffled_idx], n_batches)
        for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
            sess.run(training_op, feed_dict={hidden2: hidden2_batch, y: y_batch})

        accuracy_val = accuracy.eval(feed_dict={hidden2: h2_cache_test, y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")
