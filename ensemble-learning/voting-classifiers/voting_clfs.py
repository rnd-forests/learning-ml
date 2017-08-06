from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='soft'
    )

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

"""
Create a voting classifier using three very different classifiers including
LogisticRegression, RandomForestClassifier, and SVC.

In case all classifiers can estimate class probabilities (they all have
predict_proba() method), we can use 'soft' voting to predict the class with
the highest class probability (often getting better performance than 'hard' voting).

Support Vector Machine Classifiers doesn't predict class probabilities by default,
so we need to tell the classifier explicitly about that by setting 'probability' 
property to True.
"""
