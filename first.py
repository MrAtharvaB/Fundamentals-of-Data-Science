# 1. Write a python program to develop an ML model using KNN Classifier to predict the Species information for a given iris flower using Sepal Length, Sepal Width, Petal Length & Petal Width. Use the complete iris dataset for training. Use it to predict the species of an iris flower.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn.predict(new_flower)

print("Predicted Species:", iris.target_names[prediction][0])
