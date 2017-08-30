"""
If we can access to the original Python code that build the graph, we should use it directly without using
import_meta_graph() function.

If we want to reuse part of the original model (lower layers typically), just pick the layers that matter to us.

In the following example, we're going to keep the first three pretrained layers and add a new hidden layer.
We also need to build a new output layer, the loss for the new output, a new optimizer, new saver, and new
initialization step.
"""

import os
import tensorflow as tf
from functools import partial
from tensorflow.examples.tutorials.mnist import input_data


n_hidden4 = 20
n_outputs = 10

trained_dir = "~/python/machine-learning/ann/2-train/1-gradients-problems/saved/"
saver = tf.train.import_meta_graph(os.path.expanduser(trained_dir + "batch_normalization_dnn.ckpt.meta"))

X = tf.get_default_graph().get_tensor_by_name("input/X:0")
y = tf.get_default_graph().get_tensor_by_name("input/y:0")

hidden3 = tf.get_default_graph().get_tensor_by_name("dnn/elu_bn3:0")

momentum = 0.9
learning_rate = 0.01
he_init = tf.contrib.layers.variance_scaling_initializer()
dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)
training = tf.placeholder_with_default(False, shape=(), name="training")
batch_normalization_layer = partial(tf.layers.batch_normalization, training=training, momentum=momentum)

hidden4 = dense_layer(hidden3, n_hidden4, name="hidden4")
bn4 = batch_normalization_layer(hidden4, name="bn4")
bn4_act = tf.nn.elu(bn4, name="elu_bn4")

new_logits_before_bn = dense_layer(hidden4, n_outputs, name="new_outputs")
new_logits = batch_normalization_layer(new_logits_before_bn, name="bn5")

with tf.name_scope("new_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("new_eval"):
    correct = tf.nn.in_top_k(new_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("new_train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


n_epochs = 10
batch_size = 200
init = tf.global_variables_initializer()
mnist = input_data.read_data_sets("/tmp/data/")

with tf.Session() as sess:
    init.run()
    saver.restore(sess, os.path.expanduser(trained_dir + "batch_normalization_dnn.ckpt"))

    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={training: True, X: X_batch, y: y_batch})

        accuracy_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        accuracy_test = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", accuracy_train, "Test accuracy:", accuracy_test)
