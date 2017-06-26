import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) # randn() -> random normal

# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([0, 2, 0, 15])
# plt.show()

# Use normal equation to compute parameter vector
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 (bias term)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)

# Make the prediction on new value
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()


# Using Scikit learn
linear_reg = LinearRegression()
linear_reg.fit(X, y)
print(linear_reg.intercept_)
print(linear_reg.coef_)
prediction = linear_reg.predict(X_new)
print(prediction)
