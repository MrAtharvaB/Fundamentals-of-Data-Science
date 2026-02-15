import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

wine = datasets.load_wine()
X = wine.data
y = wine.target

models = []
models.append(('LR', LogisticRegression(max_iter=200)))
models.append(('DT', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('NB', GaussianNB()))

results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name} Accuracy: {cv_results.mean():.4f}")

plt.figure(figsize=(8,6))
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison on Wine Dataset')
plt.ylabel('Accuracy')
plt.xlabel('Algorithms')
plt.show()

best_index = np.argmax([np.mean(result) for result in results])
print("Best Model:", names[best_index])
