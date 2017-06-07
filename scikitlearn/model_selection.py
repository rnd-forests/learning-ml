import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], y_digits[:-100])
print(svc.score(X_digits[-100:], y_digits[-100:]))

# KFold cross validation
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

print(scores)

X = ["a", "a", "b", "c", "c", "c"]
k_fold = KFold(n_splits=3)
for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))

scores = [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
    for train, test in k_fold.split(X_digits)]
print(scores)

print(cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=1, scoring='precision_macro'))


# Grid search
