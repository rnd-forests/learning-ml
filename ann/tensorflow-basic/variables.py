import tensorflow as tf

# Define the variables
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

# Device placement (CPU, GPU)
with tf.device("/cpu:0"):
    v = tf.Variable(0.1, dtype=tf.float32)

# Initialize from another variable
w2 = tf.Variable(weights.initialized_value(), name="w2")
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")

# Initialize all variables
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

# Saving and Restoring variable
with tf.Session() as sess:
    sess.run(init_op)

    # save_path = saver.save(sess, "tf_logs/model.ckpt")
    # print("Model saved in file: %s" % save_path)

    saver.restore(sess, "tf_logs/model.ckpt")
    print("Model restored.")
