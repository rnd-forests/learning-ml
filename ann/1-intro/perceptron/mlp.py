"""
Multi-layer Perceptron and Backpropagation
------------------------------------------
MLP consists of an input layer, one or more LTU layers - hidden layers, and one final LTU layer called output layer.
Bias neuron is included in every layer except the output layer
Layers are fully connected together
DNN (Deep Neural Network) is an ANN with two or more hidden layers.

How to train MLPs?
-> Using Backpropagation (Gradient Descent using reverse-mode autodiff)
   Ref: https://ayearofai.com/rohan-lenny-1-neural-networks-the-backpropagation-algorithm-explained-abf4609d4f9d

General ideas behind backpropagation algorithm:
    - Each training instance is used to make prediction (forward pass)
    - Measuring the error
    - Go through each layer in reversed order to measure the error contribution from each connection (reverse pass)
    - Tweak the connection weights to reduce the error (Gradient Descent)

In order for the backpropagation algorithm to work properly, we need to change the activation function
from a simple 'step function' to 'logistic function (sigmoid function)'. The reason is that Gradient Descent cannot
work with sign function which contains only flat segments.

Common activation functions:
    - Logistic function / Sigmoid function: https://en.wikipedia.org/wiki/Sigmoid_function
    - Hyperbolic tangent function: http://functions.wolfram.com/ElementaryFunctions/Tanh/introductions/Tanh/ShowAll.html
    - ReLU function (Rectifier Linear Unit): https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

- MLPs are mainly used for classification tasks
- Each output is corresponding to a different binary class
- If the classes are exclusive, the output layer is modified by replacing the individual activation
  functions by a shared softmax function. The output of each neuron corresponds to the estimated
  probability of the corresponding class. In Scikit Learn, softmax layer is used when we execute
  predict_proba() method.

FNN (Feedforward neural network): the signal goes in only one direction from the input layer to output layer.
"""

import numpy as np
import matplotlib.pyplot as plt


def logit(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps)) / (2 * eps)


z = np.linspace(-5, 5, 200)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(z, np.sign(z), "r--", linewidth=2, label="Step")
plt.plot(z, logit(z), "g--", linewidth=2, label='Logit')
plt.plot(z, np.tanh(z), "b-", linewidth=2, label='Tanh')
plt.plot(z, relu(z), "m-.", linewidth=2, label='ReLU')
plt.grid(True)
plt.legend(loc="center right", fontsize=14)
plt.title("Activation Functions", fontsize=14)
plt.axis([-5, 5, -1.2, 1.2])

plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Step")
plt.plot(0, 0, "ro", markersize=5)
plt.plot(0, 0, "rx", markersize=10)
plt.plot(z, derivative(logit, z), "g--", linewidth=2, label="Logit")
plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
plt.title("Derivatives", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])

plt.show()
