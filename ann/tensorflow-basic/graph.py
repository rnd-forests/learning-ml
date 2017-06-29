import tensorflow as tf

# Normally, add created nodes are added to the default graph of Tensorflow.
x = tf.Variable(1)
print(x.graph is tf.get_default_graph())

# Manually creating graph
graph = tf.Graph()
with graph.as_default():
    y = tf.Variable(2)
    print(y.graph is graph)
    print(graph is tf.get_default_graph())
print(y.graph is tf.get_default_graph())

# Reset the default graph
tf.reset_default_graph()
