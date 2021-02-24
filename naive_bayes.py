# Importing necessary libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

# Generating data
X, y = make_blobs(100, 2, centers=2)
np.shape(X)
np.shape(y)
plt.scatter(X[:, 0], X[:, 1])

# model building
model = GaussianNB()
model.fit(X, y)

# model prediction

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18]*rng.rand(2000, 2)
np.shape(Xnew)
ynew = model.predict(Xnew)
ynew
