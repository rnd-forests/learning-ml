import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

"""
- Regular logistic regression is used for binary classifier.
- Softmax is a generalized version of Logistic which supports multi-class classifier
"""

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = iris["target"]

x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1)
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)
print(y_proba)
print(y_predict)

print(softmax_reg.predict_proba([[5, 2]]))
print(softmax_reg.predict([[5, 2]]))

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")

custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap, linewidth=5)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal Length", fontsize=14)
plt.ylabel("Petal Width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()
