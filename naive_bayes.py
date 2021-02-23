from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(100, 2, centers=2, )
plt.scatter(X[:, 0], X[:, 1])
