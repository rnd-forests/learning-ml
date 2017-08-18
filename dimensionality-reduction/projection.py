"""
In real life problems, training instances may contain thousands or millions of features.
As a result, this could slow down the training process as well as reduce the performance
of algorithms in general.

In order to solve that problem, we'll use dimensionality reduction techniques to turn an
intractable problem to a tractable one.

Reducing dimensionality lose some information.
It speeds up the training process, but it may cause system to perform slightly worse and
become more complex.

DR is useful for data visualization (plotting high-dimensional dataset in two/three-dimensional
space to gain insights about the dataset such as patterns or clusters)

The curse of dimensionality
---------------------------
High-dimensional dataset is usually very spare. A new instance would also very far from the training
instances. As a result predictions are less reliable.

More dimensions -> increase the risk of overfitting

There are two main approaches for DR: Projection and Manifold Learning

Projection
----------

"""

import numpy as np
import numpy.random as rnd

rnd.seed(0)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * rnd.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * rnd.randn(m)

