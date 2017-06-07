import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier

mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']

digit = X[36000]
# digit_image = digit.reshape(28, 28)
# plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
# plt.axis('off')
# plt.show()
# print(y[3600])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([digit]))
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict(X_test))


