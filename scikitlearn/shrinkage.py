import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

np.random.seed(0)
X = np.c_[.5, 1].T
y = [.5, 1]
test = np.c_[0, 2].T
print(X)
print(y)
print(test)

plt.figure()
regr = linear_model.LinearRegression()
ridge = linear_model.Ridge(alpha=.1)

for _ in range(6):
    this_X = .1 * np.random.normal(size=(2, 1)) + X
    ridge.fit(this_X, y)
    plt.plot(test, ridge.predict(test))
    plt.scatter(this_X, y, s=3)

plt.show()
