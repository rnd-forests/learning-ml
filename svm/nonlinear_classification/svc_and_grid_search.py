import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, \
    accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve

# Loads the moons dataset
X, y = make_moons(n_samples=1500, noise=0.15, random_state=42)

# Splits the original dataset into trainset and testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVC hyper-parameter space used for grid search
pram_grid = [
    {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.2, 0.5, 'auto'], 'kernel': ['rbf']}
]

# Performs grid search
clf = GridSearchCV(SVC(), param_grid=pram_grid, cv=5)
clf.fit(X_train, y_train)
y_true, y_pred = y_test, clf.predict(X_test)

# Prints classification reports and best values returned from grid search
print(classification_report(y_true, y_pred))
print("Best Parameters: {}".format(clf.best_params_))
print("Best Scores: {}".format(clf.best_score_))
print("Best estimator:")
print(clf.best_estimator_)
print()

# Uses the classifier returned from grid search to make the actual classification
polynomial_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", clf.best_estimator_)
    ])
polynomial_svm_clf.fit(X_train, y_train)
y_true, y_pred = y_test, polynomial_svm_clf.predict(X_test)

# Prints some metrics scores
print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
print("Precision: {}".format(precision_score(y_true, y_pred)))
print("Recall: {}".format(recall_score(y_true, y_pred)))
print("F1: {}".format(f1_score(y_true, y_pred)))
print()


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which="both")
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


# Train sizes: the stop points on the learning curve
def plot_learning_curve(estimator, title, X, y, cv=None,
                        n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")


plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

plt.subplot(122)
title = "Learning Curves"
plot_learning_curve(polynomial_svm_clf, title, X, y, cv=5)

plt.show()
