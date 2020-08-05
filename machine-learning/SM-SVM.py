from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.7, 0], [0, .7]]
N = 20  # number of samplers per class
X0 = np.random.multivariate_normal(means[0], cov, N)  # each row is a data point
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0, X1))
y = np.concatenate((np.ones(N), -np.ones(N)))

from sklearn.svm import SVC

C = 100
clf = SVC(kernel='linear', C=C)
clf.fit(X, y)
w_sklearn = clf.coef_.reshape(-1, 1)
b_sklearn = clf.intercept_[0]
print(w_sklearn.T, b_sklearn)
