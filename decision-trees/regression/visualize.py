from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor, export_graphviz

iris = load_iris()
X = iris.data
y = iris.target

tree_clf = DecisionTreeRegressor(max_depth=2)
tree_clf.fit(X, y)

export_graphviz(tree_clf,
                out_file="iris_tree.dot",
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                rounded=True,
                filled=True,
                special_characters=True)
