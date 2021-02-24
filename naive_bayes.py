# Importing necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()


# Making data

X, y = make_blobs(100, 2, centers=2)
plt.scatter(X[:, 0], X[:, 1])
X[:, 0]
# model building
model = GaussianNB()
model.fit(X, )
