"""
Random Forests make it easy to measure the level of importance of each feature.

Feature importance: the wighted average (the weight is equal to the number of
training examples associated with the node), across all trees in the forest,
of how much tree nodes using the feature reduce the impurity score

We can perform feature selection based on feature importance level
"""

import matplotlib
import matplotlib.pyplot as plt
from six.moves import urllib
from sklearn.datasets import load_iris, fetch_mldata
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
random_forests_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
random_forests_clf.fit(iris["data"], iris["target"])

for name, importance in zip(iris["feature_names"], random_forests_clf.feature_importances_):
    print(name, "=", importance)

# MNIST pixel importance
try:
    mnist = fetch_mldata('MNIST original')
except urllib.error.HTTPError as ex:
    print("Could not download MNIST data from mldata.org, trying alternative...")

    from scipy.io import loadmat
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"
    response = urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    print('Done!')

rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(mnist["data"], mnist["target"])

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.hot, interpolation="nearest")
    plt.axis("off")

plot_digit(rnd_clf.feature_importances_)
cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])
plt.show()
