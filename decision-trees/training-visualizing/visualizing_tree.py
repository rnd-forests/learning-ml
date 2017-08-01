from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz

iris = load_iris()
X = iris.data
y = iris.target

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X, y)

export_graphviz(tree_clf,
                out_file="iris_tree.dot",
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                rounded=True,
                filled=True,
                special_characters=True)

# http://www.graphviz.org
# Export image from dot file: dot -Tpng iris_tree.dot -o iris_tree.png
