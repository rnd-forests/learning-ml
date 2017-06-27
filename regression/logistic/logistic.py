import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

"""Implement a binary classifier with single feature
"""
iris = datasets.load_iris()
X = iris["data"][:, 3:] # extract petal width only
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)

print(log_reg)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
print(y_proba)

# Plot probabilities as a function of petal width (feature)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.show()
