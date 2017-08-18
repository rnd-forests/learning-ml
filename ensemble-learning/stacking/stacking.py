"""
Stacking
--------
Stacking or Stacked Generalization is an ensemble technique which trains a model to
aggregate predictions from classifiers rather than using simple functions like hard
voting or soft voting.

The final predictor which is used to aggregate all predictions and to make the final prediction
is called 'blender' or 'meta learner'.

How to train the blender?

Hold-out set approach
---------------------
Suppose that our ensemble has only two layers, one for the list of predictors and another for the blender.
- Splitting training set into two subsets
- Using the first subset to train the predictors (the first layer)
- Making predictions from each predictor using the second subset (held-out set) - preventing biases
- Aggregating predicted values from all predictors to construct new training set (keeping the target values)
- Training the blender using the new training set

We may have multiple layers of predictors, we just need to split the original training set into
a number of subsets.
Example: if we have three layers (2 layers of predictors and 1 layer of blender), we've to split the
training set into three subsets. First subset is used to train the first layer of predictors. The second
subset is used to create training set for the second layer (used for the first layer to make predictions
and aggregate all of these predictions to construct new training set). The final subset is used to construct
training set for the blender in order to make toe final prediction.
"""

import sklearn
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from brew.base import Ensemble, EnsembleClassifier
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from brew.combination.combiner import Combiner
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions

clf1 = LogisticRegression(random_state=42)
clf2 = RandomForestClassifier(random_state=42)
clf3 = SVC(probability=True, random_state=42)

ensemble = Ensemble([clf1, clf2, clf3])
eclf = EnsembleClassifier(ensemble=ensemble, combiner='mean')

layer1 = Ensemble([clf1, clf2, clf3])
layer2 = Ensemble([sklearn.clone(clf1)])

stack = EnsembleStack(cv=3)
stack.add_layer(layer1)
stack.add_layer(layer2)

sclf = EnsembleStackClassifier(stack, combiner=Combiner('mean'))

clf_list = [clf1, clf2, clf3, eclf, sclf]
lbl_list = ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble', 'Stacking']

X, y = iris_data()
X = X[:,[0, 2]]

gs = gridspec.GridSpec(2, 3)
plt.figure(figsize=(10, 8))

itt = itertools.product([0, 1, 2], repeat=2)

for clf, lab, grd in zip(clf_list, lbl_list, itt):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)
plt.show()
