"""
Perceptron
----------
One of the simplest ANN architectures
Based on a type of artificial neuron called 'linear threshold unit' (LTU)
    - inputs and outputs are numbers
    - input connection is associated with a weight
    - compute the weighted sum of its inputs (z = w1.x1 + w2.x2 + ... + wn.xn = w.T.x <-> vectorized)
    - apply the step function to the above sum and output the result
        + Heaviside step function: https://en.wikipedia.org/wiki/Heaviside_step_function
        + Sign function: https://en.wikipedia.org/wiki/Sign_function

A single LTU can be used for linear binary classification by computing a linear combination of the inputs.
If the result exceeds a threshold, it outputs the positive class and negative class otherwise.

About bias terms: https://ayearofai.com/rohan-5-what-are-bias-units-828d942b4f52

A perceptron is just a single layer of LTU. Each LTU is connected to all input neurons.
A special bias neuron (output 1 all the time) is also connected to all LTU.

Perceptron can be used for multiclass classification tasks.
Perceptron cannot be used to learn complex patterns. It is the best fit for linearly separable datasets.

How to train a Perceptron !?
Ref: https://www.slideshare.net/AndresMendezVazquez/14-machine-learning-single-layer-perceptron
     https://www.slideshare.net/MohammedBennamoun/artificial-neural-network-lect4-single-layer-perceptron-classifiers

-> Using Hebb's rule or Hebb Learning: the connection weight between two neurons is increased
   whenever they have the same output.
Feed the Perceptron with one training instance at a time and make the predictions for that instance.
For every output neuron produced a wrong prediction, it reinforces the connection weights from the
inputs that would have contributed to the correct prediction.

Think of Perceptron as a resembling version of SGD
In Scikit Learn Perceptron is just a SGD classifier with the following hyperparameters:
    - loss="perceptron"
    - learning_rate="constant"
    - eta0=1 (the actual learning rate used by perceptron)
    - penalty=None

Perceptron doesn't output the class probabilities, it makes predictions based on a hard threshold.
As a result, we should prefer Logistic Regression over Perceptron.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length and petal width
y = (iris.target == 0).astype(np.int)

perceptron_clf = Perceptron(random_state=42)
perceptron_clf.fit(X, y)
y_pred = perceptron_clf.predict([[2, 0.5]])
print(y_pred)
print(perceptron_clf.coef_)
print(perceptron_clf.intercept_)

a = -perceptron_clf.coef_[0][0] / perceptron_clf.coef_[0][1]
b = -perceptron_clf.intercept_ / perceptron_clf.coef_[0][1]

axes = [0, 5, 0, 2]
x0, x1 = np.meshgrid(np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
                     np.linspace(axes[2], axes[3], 500).reshape(-1, 1))

X_new = np.c_[x0.ravel(), x1.ravel()]
y_pred = perceptron_clf.predict(X_new)
zz = y_pred.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 0, 0], X[y == 0, 1], "bs", label="Not Iris-Setosa")
plt.plot(X[y == 1, 0], X[y == 1, 1], "yo", label="Iris-Setosa")

# Decision boundary
plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-", linewidth=3)

plt.contourf(x0, x1, zz, cmap=ListedColormap(['#9898ff', '#fafab0']), linewidth=5)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.axis(axes)

plt.show()
