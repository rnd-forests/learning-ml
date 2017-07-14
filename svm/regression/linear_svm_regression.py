import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR


def find_support_vectors(regressor, X, y):
    y_pred = regressor.predict(X)
    off_margin = (np.abs(y - y_pred) >= regressor.epsilon)
    return np.argwhere(off_margin)


def plot_svm_regression(regressor, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = regressor.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + regressor.epsilon, "k--")
    plt.plot(x1s, y_pred - regressor.epsilon, "k--")
    plt.scatter(X[regressor.support_], y[regressor.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)


rnd.seed(42)
m = 150
X = 2 * rnd.randn(m, 1)
y = (4 + 3 * X + rnd.randn(m, 1)).ravel()
regressors = (LinearSVR(epsilon=1.5), LinearSVR(epsilon=0.5))
regressors = (regressor.fit(X, y) for regressor in regressors)

plt.figure(figsize=(9, 4))
for i, regressor in enumerate(regressors):
    regressor.support_ = find_support_vectors(regressor, X, y)
    plt.subplot(121 + i)
    plot_svm_regression(regressor, X, y, [0, 2, 3, 11])
    plt.title(r"$\epsilon = {}$".format(regressor.epsilon), fontsize=18)
    if i == 0:
        eps_x1 = 1
        eps_y_pred = regressor.predict([[eps_x1]])
        plt.ylabel(r"$y$", fontsize=18, rotation=0)
        plt.annotate(
            '', xy=(eps_x1, eps_y_pred), xycoords='data',
            xytext=(eps_x1, eps_y_pred - regressor.epsilon),
            textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
        )
        plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)

plt.show()
