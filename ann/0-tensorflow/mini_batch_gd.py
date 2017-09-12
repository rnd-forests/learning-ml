import numpy as np
import tensorflow as tf
import numpy.random as rnd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Placeholder node testing
# A = tf.placeholder(dtype=tf.float32, shape=(None, 3))
# B = A + 5
# with tf.Session() as sess:
#     B_val = B.eval(feed_dict={A: [[1, 2, 3], [4, 5, 6]]})
#
# print(B_val)

housing = fetch_california_housing()
m, n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

learning_rate = 0.01

X = tf.placeholder(dtype=tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform((n + 1, 1), -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()


def fetch_batch(epoch, batch_index, batch_size):
    rnd.seed(epoch * n_batches * batch_index)
    indices = rnd.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch


n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()

print('Best theta:')
print(best_theta)
