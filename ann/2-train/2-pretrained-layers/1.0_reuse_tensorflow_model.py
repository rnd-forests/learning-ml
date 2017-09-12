"""
It's not a good idea to train large DNN from scratch. We should find an existing NN that does the similar task
to the one that we're trying to accomplish. Then, we reuse the lower layers of that network.
This technique is called TRANSFER LEARNING. It speeds up training process and requires much less training data.
Transfer learning will only work well if the inputs have similar low-level features.

Reusing a TensorFlow model
--------------------------
"""

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

trained_dir = "~/python/machine-learning/ann/2-train/1-gradients-problems/saved/"
# Use import_meta_graph() function to import the operations into the default graph.
# This function returns a Saver instance.
saver = tf.train.import_meta_graph(os.path.expanduser(trained_dir + "batch_normalization_dnn.ckpt.meta"))

# List all the operations inside the graph using get_operations() function.
# It's better to use tool like TensorBoard to explore the structure of the graph.
# for op in tf.get_default_graph().get_operations():
#     print(op.name)

# Load tensors and operations from loaded graph using
# get_tensor_by_name() and get_operation_by_name() functions

# X is inside 'input' name scope and it is the first input
X = tf.get_default_graph().get_tensor_by_name("input/X:0")
# y is inside 'input' name scope and it is the first input
y = tf.get_default_graph().get_tensor_by_name("input/y:0")
# 'accuracy' is inside 'evaluation' name scope and it is the first input
accuracy = tf.get_default_graph().get_tensor_by_name("evaluation/accuracy:0")
# GradientDescent operation is inside 'train' name scope
training_op = tf.get_default_graph().get_operation_by_name("train/GradientDescent")

# Load tensors and operations from the defined collection
# X, y, accuracy, training_op = tf.get_collection("important_ops")

n_epochs = 10
batch_size = 200
mnist = input_data.read_data_sets("/tmp/data/")

with tf.Session() as sess:
    # Restore model's state and train on new data
    saver.restore(sess, os.path.expanduser(trained_dir + "batch_normalization_dnn.ckpt"))

    # Train using the loaded model
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)
