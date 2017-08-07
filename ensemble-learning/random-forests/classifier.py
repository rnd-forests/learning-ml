"""
Random forests is just an ensemble of Decision Trees (generally using bagging method)

Generally, max_samples is set to the size of training set.

RandomForestClassifier has all parameters of DecisionTreeClassifier (for building the tree)
and BaggingClassifier (for controlling the ensemble)

Random Forest searches for the best feature among a random subset of features (extra randomness)
As a result, the tree is better in term of diversity -> higher bias and lower variance -> overall better model.
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set n_jobs to -1 to train the model with all available CPU cores
rf_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))

# Building RandomForest from BaggingClassifier (just for fun)
bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter='random', max_leaf_nodes=16),
                            n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)
y_pred_bag = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_bag))

print('Comparing two set of predictions')
print(np.sum(y_pred_bag == y_pred_rf) / len(y_pred_bag))
