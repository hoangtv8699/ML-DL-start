import numpy as np
np.random.seed(22)
# simulated samples
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # blue class data
X1 = np.random.multivariate_normal(means[1], cov, N) # red class data
X = np.concatenate((X0, X1), axis = 0) # all data
y = np.concatenate((np.ones(N), -np.ones(N)), axis = 0) # label

# solution by sklearn
from sklearn.svm import SVC
model = SVC(kernel = 'linear', C = 1e5) # just a big number
model.fit(X, y)
w = model.coef_
b = model.intercept_
print('w = ', w)
print('b = ', b)