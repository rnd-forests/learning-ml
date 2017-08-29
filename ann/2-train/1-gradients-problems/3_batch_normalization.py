"""
Batch Normalization
-------------------
Ref: https://arxiv.org/pdf/1502.03167v3.pdf

Using He initialization and ELU activation function may prevent vanishing/exploding gradients problems at the BEGINNING
of the training process; however, they may come back DURING training process.

Specifically, the distribution of each layer's inputs changes during training as the parameters of the previous layers
change (Internal Covariate Shift problem - read more in the paper). It's a primary cause of
vanishing/exploding gradients.

Batch normalization is the solution
-----------------------------------
- Adding an additional operation before the activation function of each layer.
- This operation zero-centers and normalizes the inputs, then it scales and shifts using two additional parameters (one
  for scaling, the other for shifting)
- Details: https://www.dropbox.com/s/jxva2iqhuo4ssmq/Screenshot%20from%202017-08-29%2009-23-16.png?dl=0
- All calculations are performed on a mini-batch not the whole training set
- During testing time, empirical mean and standard deviation are not calculated on mini-batch (because there's no
  mini-batch). Instead, we use the whole training set's mean and standard deviation (calculated during training using
  moving average)
- There are four parameters that are learned for each batch-normalized layer: γ (scale factor),
  β (offset - shift factor), μ (mean), and σ (standard deviation).

Results:
- Reduce the V/E gradients problems to the point that we can use saturating activation functions (like tanh or sigmoid).
- Less sensitive to weight initialization.
- Possible to use much larger learning rate to speed up the training process.
- BN acts as regularizer.

Batch normalization adds some complexity to the model and makes calculations much slower -> slower when making
predictions.
Trying to use ELU + He initialization first before attemping to use BN
"""

import numpy as np
import tensorflow as tf
from datetime import datetime
from functools import partial
from tensorflow.examples.tutorials.mnist import input_data

n_inputs = 784
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01
batch_momentum = 0.9


def reset_tf_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_tf_graph()

with tf.name_scope("input"):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    # The training placeholder. During training, we'll set it to True.
    # It's used to tell the tf.layers.batch_normalization() function
    # whether it should use the current mini-batch or the whole trainset
    # mean and standard deviation (when testing - set it to False)
    training = tf.placeholder_with_default(False, shape=(), name="training")

with tf.name_scope("dnn"):
    """
    We can use tf.nn.batch_normalization() to center and normalize the inputs; however,
    we've to compute the mean and standard deviation manually for each mini-batch or for
    the testing set. We also have to handle the creation of scaling and offset parameters.
    Afther all calculations, pass results as the parameters to tf.nn.batch_normalization()
    To simplify things, we may use tf.layers.batch_normalization() which hanles all the
    calculations behind the scene.
    """

    he_init = tf.contrib.layers.variance_scaling_initializer()
    dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)
    batch_normalization_layer = partial(tf.layers.batch_normalization, training=training, momentum=batch_momentum)

    # Omit the 'activation' parameter as we're going to apply it manually
    hidden1 = dense_layer(X, n_hidden1, name="hidden1")
    # Calculate batch normalization for the inputs of the second hidden layer
    # The BN algorithm uses 'exponential decay' to compute the running averages.
    # 'momentum' parameter is used for this purpose.
    # Given a value v then the running average v.hat is upated as follow:
    #   v.hat <- v.hat * momentum + v * (1 - momentum)
    #
    # Normally, momentum value is greater or equal to 0.9 (very close to 1)
    # Larger datasets, smaller mini-batches -> bigger value of momentum
    # bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=batch_momentum)
    bn1 = batch_normalization_layer(hidden1)
    tf.summary.histogram("batch_normalization", bn1)
    bn1_act = tf.nn.elu(bn1)
    tf.summary.histogram("activations", bn1_act)

    hidden2 = dense_layer(bn1_act, n_hidden2, name="hidden2")
    bn2 = batch_normalization_layer(hidden2)
    tf.summary.histogram("batch_normalization", bn2)
    bn2_act = tf.nn.elu(bn2)
    tf.summary.histogram("activations", bn2_act)

    hidden3 = dense_layer(bn2_act, n_hidden2, name="hidden3")
    bn3 = batch_normalization_layer(hidden3)
    tf.summary.histogram("batch_normalization", bn3)
    bn3_act = tf.nn.elu(bn3)
    tf.summary.histogram("activations", bn3_act)

    logits_before_bn = dense_layer(bn3_act, n_outputs, name="outputs")
    logits = batch_normalization_layer(logits_before_bn)
    tf.summary.histogram("batch_normalization", logits)


with tf.name_scope("cross_entropy"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    with tf.name_scope("total"):
        loss = tf.reduce_mean(xentropy, name="loss")
tf.summary.scalar('cross_entropy', loss)


with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # batch_normalization() function creates operations which must be evaluated at
    # each step during training to update the moving averages. These operations are
    # automatically added to the UPDATE_OPS collection.
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Use tf.control_dependencies() to specify a list of operations or tensor objects
    # that must be executed or computed before running the operations defined in the
    # context.
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss)


with tf.name_scope("evaluation"):
    with tf.name_scope("correct_predicion"):
        correct = tf.nn.in_top_k(logits, y, 1)
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


def buil_summary_writer(path, graph=tf.get_default_graph()):
    path = "{}/run-{}".format('summary', datetime.utcnow().strftime("%Y%m%d%H%M%S")) + '/' + path
    return tf.summary.FileWriter(path, graph)


n_epochs = 100
batch_size = 200

merged = tf.summary.merge_all()
train_writer = buil_summary_writer('train')
test_writer = buil_summary_writer('test')

saver = tf.train.Saver()
init = tf.global_variables_initializer()
mnist = input_data.read_data_sets("/tmp/data/")

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # For operations depending on batch normalization, set the training placeholder to True
            train_summary, _ = sess.run([merged, training_op], feed_dict={training: True, X: X_batch, y: y_batch})
            if iteration % 10 == 0:
                train_writer.add_summary(train_summary, epoch * n_batches + iteration)

        accuracy_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        test_summary, accuracy_test = sess.run([merged, accuracy],
                                               feed_dict={X: mnist.test.images, y: mnist.test.labels})
        test_writer.add_summary(test_summary, epoch)
        print(epoch, "Train accuracy:", accuracy_train, "Test accuracy:", accuracy_test)
    saver.save(sess, "./saved/batch_normalization_dnn.ckpt")

train_writer.close()
test_writer.close()
