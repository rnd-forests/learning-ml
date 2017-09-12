"""
Training a MLP using TensorFlow's TF.Learn
TF.Learn (tf.contrib.learn) is a Tensorflow high-level API which offers ScikitLearn-compatible API
"""

import tensorflow as tf
from sklearn.metrics import accuracy_score, log_loss
from tensorflow.examples.tutorials.mnist import input_data

# Download and extract MNIST dataset
mnist = input_data.read_data_sets("/tmp/data/")

# Construct the trainset and the testset
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

# Set the seed number to create uniform randomization across the runs
config = tf.contrib.learn.RunConfig(tf_random_seed=42)

# Create a set of real valued columns from the training set
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

# Construct a DNN with two hidden layers (one with 300 neurons and the other with 100 neurons)
# and a softmax output layer with 10 neurons (10 classes - number from 0->9)
# DNNClassifier creates all neuron layers with ReLU activation function (as default)
# We can change the default activation function using activation_fn hyperparameter
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100],
                                         n_classes=10,
                                         feature_columns=feature_cols)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)

# Perform the training process with 40 thousands instances and batch size of 50 images
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)

y_pred = dnn_clf.predict(X_test)
accuracy_score(y_test, y_pred['classes'])

y_pred_proba = y_pred['probabilities']
log_loss(y_test, y_pred_proba)
