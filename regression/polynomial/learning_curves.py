import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

rnd.seed(42)

m = 100
X = 6 * rnd.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + rnd.randn(m, 1)
X_new = np.linspace(-3, 3, 100).reshape(100, 1)

# plot polynomial regression with different degrees.
for style, width, degree in (("g-", 1, 300), ("b-", 2, 2), ("r-+", 2, 1)):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
            ("poly_features", poly_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg)
        ])
    polynomial_regression.fit(X, y)
    y_new = polynomial_regression.predict(X_new)
    # plt.plot(X_new, y_new, style, label=str(degree), linewidth=width)

# plt.plot(X, y, "b.", linewidth=3)
# plt.legend(loc="upper left")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([-3, 3, 0, 10])


"""Learning curves
- Determine whether the given model is overfitting or underfitting the data
- Plots of model's performance on the training set and the validation set as a function of the training set size.
- Training the model different times on different sized subsets of the training set.
"""
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

plt.subplots(nrows=2, ncols=1)

lin_reg = LinearRegression()
plt.subplot(2, 1, 1)
plot_learning_curves(lin_reg, X, y)
plt.legend(loc='best')

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression())
    ])
plt.subplot(2, 1, 2)
plot_learning_curves(polynomial_regression, X, y)
plt.axis([0, 100, 0, 3])
plt.legend(loc='best')

plt.show()

"""Note
- Resolve underfitting problem NOT by adding more training example BUT by using a more complex model
  OR by choosing better features

- Resolve overfitting problem by adding more training data until the validation error reach the training error
"""

"""Bias vs. Variance
>> Model's generalization error = Bias + Variance + Irreducible error

- Bias: wrong assumptions about the data. High-bias model -> underfitting
- Variance: the model is too sensitive to small variations in the training data. The model has many
  degrees of freedom (ex. high-degree polynomial model) -> high variance -> overfitting
- Increase model's complexity -> increase its variance and reduce its bias and vice versa.
"""
