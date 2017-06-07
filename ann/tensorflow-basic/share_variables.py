import tensorflow as tf
from datetime import datetime

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs/relu'
logdir = "{}/relu-{}".format(root_logdir, now)


# def relu(X):
#     with tf.variable_scope("relu", reuse=True):
#         threshold = tf.get_variable("threshold")
#         w_shape = (int(X.get_shape()[1]), 1)
#         w = tf.Variable(tf.random_normal(w_shape), name="weights")
#         b = tf.Variable(0.0, name="bias")
#         z = tf.add(tf.matmul(X, w), b, name="z")
#         return tf.maximum(z, threshold, name="relu")

# with tf.variable_scope("relu"):
#     threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
#
# relus = [relu(X) for i in range(5)]
# output = tf.add_n(relus, name="output")



def relu(X):
    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, threshold, name="relu")

relus = []
for relu_index in range(5):
    with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))

summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
summary_writer.close()
