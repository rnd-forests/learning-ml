"""
In Random Forest, for each node, a random subset of the features is considered for splitting
We can make tree more random by using random thresholds for each feature rather than search for
best possible thresholds (minimize the cost function)

In Scikit Learn, we use ExtraTreesClassifier for extremely randomized trees ensemble

Extra-Trees is much faster than Random Forest because finding the possible threshold is the most
time-consuming task when building the tree

We can compare RandomForestClassifier and ExtraTreesClassifier using cross-validation, and tune
their hyperparameters using grid search.
"""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.30)
X_train, X_test, y_train, y_test = train_test_split(X, y)

rf_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))
