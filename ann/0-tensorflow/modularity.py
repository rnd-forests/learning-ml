import tensorflow as tf
from datetime import datetime

# Implement the ReLU
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs/relu'
logdir = "{}/relu-{}".format(root_logdir, now)


# Not good code
# w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weight1")
# w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weight2")
# b1 = tf.Variable(0.0, name="bias1")
# b2 = tf.Variable(0.0, name="bias2")
#
# z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
# z2 = tf.add(tf.matmul(X, w2), b2, name="z2")
#
# relu1 = tf.maximum(z1, 0., name="relu1")
# relu2 = tf.maximum(z2, 0., name="relu2")
#
# output = tf.add(relu1, relu2, name="output")


# Better code
def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, 0., name="relu")


relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
summary_writer.close()
