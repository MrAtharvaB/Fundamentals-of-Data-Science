# link of dataset: https://www.kaggle.com/datasets/uciml/iris

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")

X = df.iloc[:, 1:5]

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans.labels_)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KMeans Clustering on Iris Dataset")
plt.show()
