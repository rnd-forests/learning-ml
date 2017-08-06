"""
Use the same algorithm for all predictors in the ensemble
Training predictors using random subsets of the training set (sampling)

Bagging (bootstrap aggregating): using sampling with replacement on the training set
Pasting: using sampling without replacement on the training set

Bagging allows training instances to be sampled several time
for the same predictors

Ensemble can make prediction for a new instance by aggregating
the predictions of all predictors.

Generally, the ensemble has a similar bias but lower variance than
a single predictor trained on the original dataset.

Predictors can be trained in parallel and predictions can be made
in parallel too -> bagging and pasting scale very well.

Bootstrapping (Bagging) introduces a bit more diversity in the subsets
that each predictors is trained on. As a result, bagging has a slightly
higher bias than pasting; however, the ensemble's variance is reduced
(the decision boundary is less irregular)

Bagging often results in better models

We can use cross validation to evaluate bagging and pasting and choose
the one that works best.

When using bagging, some instances may not be sampled, they are called
out-of-bag instances. These instances have never used in training process,
so predictor can be evaluated using those instances without the need
for separate validation set (or cross-validation).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                            n_estimators=500,
                            max_samples=100,
                            bootstrap=True,
                            n_jobs=-1,
                            random_state=42)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Using single decision tree -> smaller accuracy score
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))

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


plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
# plt.show()

"""
To use Pasting -> set 'bootstrap' parameter to False

If the predictor (classifier or regressor) used in ensemble
can estimate class probabilities, BaggingClassifier will
automatically perform 'soft voting' instead of 'hard voting'
"""

oob_bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                                n_estimators=500,
                                bootstrap=True,
                                n_jobs=-1,
                                oob_score=True)

oob_bag_clf.fit(X_train, y_train)
print(oob_bag_clf.oob_score_)

oob_y_pred = oob_bag_clf.predict(X_test)
print(accuracy_score(y_test, oob_y_pred))

# print(oob_bag_clf.oob_decision_function_)


"""
BaggingClassifier supports sampling the features using max_features and bootstrap_features hyperparameters

Random Patches: sampling both training instances and features
Random Subspaces: sampling on features e.g (bootstrap=False, max_samples=1.0, bootstrap_features=True, max_features=0.5)

Sampling features results in more predictor diversity -> more bias but lower variance
"""

# Not so good !!!
random_subspaces_bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                                             n_estimators=500,
                                             bootstrap=False,
                                             max_samples=1.0,
                                             bootstrap_features=True,
                                             max_features=0.7)

random_subspaces_bag_clf.fit(X_train, y_train)
random_subspaces_pred = random_subspaces_bag_clf.predict(X_test)
print(accuracy_score(y_test, random_subspaces_pred))
