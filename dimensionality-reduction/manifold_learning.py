"""
Manifold Learning
-----------------
A d-dimensional manifold is a part of a n-dimensional space (d < n) which locally
resembles a d-dimensional hyperplane.
For example: Swiss roll dataset is actually a 2D plan twisted in the third dimension. (d=2, n=3)

Manifold Learning relies on the manifold assumption (or hypothesis) which states that most
real-word high-dimensional datasets lie close to a lower-dimensional manifold.

The classification and regression tasks would be simpler if expressed in the lower-dimensional
space of the manifold.

Be aware that reducing the dimensionality of the training set does speed up the training process; however,
it may not always lead to a better and simpler solution (it depends on the dataset)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # This is required to render 3D graph !?
from sklearn.datasets import make_swiss_roll

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# Plot the swiss roll dataset
# axes = [-11.5, 14, -2, 23, -12, 15]
# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.viridis)
# ax.view_init(10, -70)
# ax.set_xlabel("$x_1$", fontsize=18)
# ax.set_ylabel("$x_2$", fontsize=18)
# ax.set_zlabel("$x_3$", fontsize=18)
# ax.set_xlim(axes[0:2])
# ax.set_ylim(axes[2:4])
# ax.set_zlim(axes[4:6])

# Illustrate the projection problems
# plt.figure(figsize=(11, 4))
# plt.subplot(121)
# plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.viridis)
# plt.axis(axes[:4])
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$x_2$", fontsize=18, rotation=0)
# plt.grid(True)
#
# plt.subplot(122)
# plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.viridis)
# plt.axis([4, 15, axes[2], axes[3]])
# plt.xlabel("$z_1$", fontsize=18)
# plt.grid(True)

# Illustrate the decision boundary (using manifold or not)
# Note that it doesn't always lead to a better solution.
# axes = [-11.5, 14, -2, 23, -12, 15]
# x2s = np.linspace(axes[2], axes[3], 10)
# x3s = np.linspace(axes[4], axes[5], 10)
# x2, x3 = np.meshgrid(x2s, x3s)
# positive_class = X[:, 0] > 5

# fig = plt.figure(figsize=(6, 5))
# ax = plt.subplot(111, projection='3d')
# X_pos = X[positive_class]
# X_neg = X[~positive_class]
# ax.view_init(10, -70)
# ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
# ax.plot_wireframe(5, x2, x3, alpha=0.5)
# ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
# ax.set_xlabel("$x_1$", fontsize=18)
# ax.set_ylabel("$x_2$", fontsize=18)
# ax.set_zlabel("$x_3$", fontsize=18)
# ax.set_xlim(axes[0:2])
# ax.set_ylim(axes[2:4])
# ax.set_zlim(axes[4:6])


# fig = plt.figure(figsize=(5, 4))
# ax = plt.subplot(111)
#
# plt.plot(t[positive_class], X[positive_class, 1], "gs")
# plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
# plt.axis([4, 15, axes[2], axes[3]])
# plt.xlabel("$z_1$", fontsize=18)
# plt.ylabel("$z_2$", fontsize=18, rotation=0)
# plt.grid(True)


# positive_class = 2 * (t[:] - 4) > X[:, 1]

# fig = plt.figure(figsize=(6, 5))
# ax = plt.subplot(111, projection='3d')
#
# X_pos = X[positive_class]
# X_neg = X[~positive_class]
# ax.view_init(10, -70)
# ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
# ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
# ax.set_xlabel("$x_1$", fontsize=18)
# ax.set_ylabel("$x_2$", fontsize=18)
# ax.set_zlabel("$x_3$", fontsize=18)
# ax.set_xlim(axes[0:2])
# ax.set_ylim(axes[2:4])
# ax.set_zlim(axes[4:6])


# fig = plt.figure(figsize=(5, 4))
# ax = plt.subplot(111)
#
# plt.plot(t[positive_class], X[positive_class, 1], "gs")
# plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
# plt.plot([4, 15], [0, 22], "b-", linewidth=2)
# plt.axis([4, 15, axes[2], axes[3]])
# plt.xlabel("$z_1$", fontsize=18)
# plt.ylabel("$z_2$", fontsize=18, rotation=0)
# plt.grid(True)

# plt.show()
