import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

"""
Regularization is used to constraining the weights of the model -> reducing overfitting
Data should be scaled before using regularization technique.
"""

"""Rigde Regression (Tikhonov regularization)
- Regularization term: α∑(i := 1->n) θi^2 (the sum of the square of coefficients)
- Regularization term should be added to the cost function during training process only.
"""

rnd.seed(42)
m = 20
X = 3 * rnd.rand(m, 1)
y = 1 + 0.5 * X + rnd.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)


def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model)
                ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])

plt.figure(figsize=(8, 4))
plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
plt.show()


"""Ridge regression using closed-form formula
\hat(theta) = (X.T * X + alpha*A) ^ -1 * X.T * y

A is an identity matrix with the size of n x n (except for the top-left corner with value of 0 -> bias term)
"""
ridge_reg = Ridge(alpha=1, solver="cholesky") # using matrix factorization technique
ridge_reg.fit(X, y)
pred = ridge_reg.predict([[1.5]])
print(pred)


"""Ridge regression using Gradient Descent
"""
sgd_reg = SGDRegressor(penalty="l2", random_state=42)
sgd_reg.fit(X, y.ravel())
pred = sgd_reg.predict([[1.5]])
print(pred)

ridge_reg = Ridge(alpha=1, solver="sag", random_state=42) # using matrix factorization technique
ridge_reg.fit(X, y)
pred = ridge_reg.predict([[1.5]])
print(pred)
