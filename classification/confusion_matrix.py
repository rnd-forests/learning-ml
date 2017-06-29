import numpy as np
from sklearn.base import clone
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5)

sgd_clf = SGDClassifier(random_state=42)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion = confusion_matrix(y_train_5, y_train_pred)
print(confusion)

print('Precision:', precision_score(y_train_5, y_train_pred))
print('Recall:', recall_score(y_train_5, y_train_pred))
print('F1 Score:', f1_score(y_train_5, y_train_pred))

sgd_clf.fit(X_train, y_train)
# Get the decision function to SGDClassifier (scores corresponding to each label of all samples)
y_scores = sgd_clf.decision_function(X_train)
print(y_train[1])
print(y_scores[1])
print(np.amax(y_scores[1])) # max score will be the predicted label


def plot_precision_recall_and_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label="Precision")
    plt.plot(thresholds, recalls[:-1], 'g-', label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

print('\nPrecision-Recall curve')
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plot_precision_recall_and_threshold(precisions, recalls, thresholds)
plt.show()
