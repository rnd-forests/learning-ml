import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x * x * y + y + 2

# sess = tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# result = sess.run(f)
# print(result)
# sess.close()


# with tf.Session() as sess:
#     # session is automatically closed inside the with block
#     print(tf.get_default_session())
#
#     x.initializer.run() # tf.get_default_session().run(x.initializer)
#     y.initializer.run()
#     result = f.eval() # tf.get_default_session().run(f)
#     print(result)

# Initialize all global variables
init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     print(f.eval())

# Interactive session (automatically sets itself as the default session -> dont't need with block)
sess = tf.InteractiveSession()
init.run()
print(f.eval())
sess.close()

# To build a Tensorflow program we need to build the computational graph first, then execute that graph.
