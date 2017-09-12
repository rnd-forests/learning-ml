"""
Vanishing and Exploding Gradients Problems
------------------------------------------
The Backpropagation algorithm works by going from the output layer to the input layer and propagating error gradient
in its way. After the gradients of the cost function with regards to each parameter in the network are computed, it uses
these gradients to update each parameter with a Gradient Descent step.

Problem: Gradients getting smaller when going back to lower layers.
         As a result, gradient descent update leaves the lower layer connection weights unchanged.
-> Training process cannot converge to a good solution.
-> Causing vanihsing gradients problem.

If gradients getting bigger -> a lot of large weight updates -> algorithm dgiverges -> exploding gradients problems
(usually happening in RNN)

Main causes:
- Sigmoid activation function.
- Weight initialization technique (random initialization using normal distribution with mean = 0 and stdv = 1)
Using the above two techniques makes the variance of the outputs of each layer much greater than the variance
of its inputs. The variance keeps increasing after each layer as we go forward in the network until the activation
function saturates at the top layers.

Note that the sigmoid function has the mean of 0.5 not 0 as our weight initialization process.

Note that for sigmoid activation function, when the inputs are large the function saturates at 0 or 1
-> Derivative is very close to 0 (getting smaller as we go back from the output layer)
-> Backpropagation algorithm has no gradients to work with in lower layers.
"""

import numpy as np
import matplotlib.pyplot as plt


def logit(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-5, 5, 200)

plt.plot([-5, 5], [0, 0], "k-")
plt.plot([-5, 5], [1, 1], "k--")
plt.plot([0, 0], [-0.2, 1.2], "k-")
plt.plot([-5, 5], [-3 / 4, 7 / 4], "g--")
plt.plot(z, logit(z), "b-", linewidth=2)
props = dict(facecolor='black', shrink=0.1)
plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
plt.grid(True)
plt.title("Sigmoid activation function", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])

plt.show()
