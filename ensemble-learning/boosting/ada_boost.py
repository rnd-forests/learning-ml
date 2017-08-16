"""
Boosting or Hypothesis boosting combines several weak learners into a strong learner.
In boosting, predictors are trained sequentially, each subsequent predictor tries to
correct its predecessor.

AdaBoost (Adaptive Boosting)
--------
Pay attention to the training instances that the predecessor underfitted.
As a result, we have new predictors focusing on hard cases.

In the context of an AdaBoost classifier, a base classifier (e.g Decision Tree)
is trained and used to make predictions on the trainset. Misclassified instances' weights
are then increased. Another classifier is trained using the updated weights to make predictions.
This process stops when we reach the maximum number of estimators (n_estimators in Scikit-Learn)
or a perfect predictor is found.

AdaBoost technique is very much like Gradient Descent technique except that instead of tweaking
a single predictor's parameters (to minimize the cost function), it adds predictors to the ensemble
(with updated weights of instances).

Drawback:
This sequential learning technique cannot be parallelized since each predictor can only be trained
after the previous predictor has been trained and evaluated.

Scikit-Learn uses multiclass version of AdaBoost called SAMME (Stagewise Additive Modeling using a
Multiclass Exponential loss function). If we have just two classes SAMME is equivalent to AdaBoost.
SAMME.R is a variant of SAMME which predicts class probabilities (if the estimator has predict_proba()
method) rather than the actual class.

Decision Stump is a Decision Tree with max_depth=1 (tree is composed of a single decision node and two
leaf nodes)

If AdaBoost ensemble is overfitting the training set, we can reduce the number of estimators or
regularize more on the base estimator.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap, linewidth=10)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                             n_estimators=200,
                             algorithm='SAMME.R',
                             learning_rate=0.5,
                             random_state=42)
ada_clf.fit(X_train, y_train)
# plot_decision_boundary(ada_clf, X, y)


m = len(X_train)
plt.figure(figsize=(11, 4))
for subplot, learning_rate in ((121, 1), (122, 0.5)):
    sample_weights = np.ones(m)
    for i in range(5):
        plt.subplot(subplot)
        svm_clf = SVC(kernel="rbf", C=0.05)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = svm_clf.predict(X_train)
        sample_weights[y_pred != y_train] *= (1 + learning_rate)
        plot_decision_boundary(svm_clf, X, y, alpha=0.2)
        plt.title("learning rate = {}".format(learning_rate - 1), fontsize=16)

plt.subplot(121)
plt.text(-0.7, -0.65, "1", fontsize=14)
plt.text(-0.6, -0.10, "2", fontsize=14)
plt.text(-0.5,  0.10, "3", fontsize=14)
plt.text(-0.4,  0.55, "4", fontsize=14)
plt.text(-0.3,  0.90, "5", fontsize=14)

plt.show()
