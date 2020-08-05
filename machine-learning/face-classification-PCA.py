import numpy as np
from scipy import misc  # for loading image

np.random.seed(1)
# filename structure
path = 'unpadded/'  # path to the database
ids = range(1, 16)  # 15 persons
states = ['centerlight', 'glasses', 'happy', 'leftlight',
          'noglasses', 'normal', 'rightlight', 'sad',
          'sleepy', 'surprised', 'wink']
prefix = 'subject'
surfix = '.pgm'
# data dimension
h, w, K = 116, 98, 100  # hight, weight, new dim
D = h * w
N = len(states) * 15
# collect all data
X = np.zeros((D, N))
cnt = 0
for person_id in range(1, 16):
    for state in states:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        X[:, cnt] = misc.imread(fn).reshape(D)
        cnt += 1
# Doing PCA, note that each row is a datapoint
from sklearn.decomposition import PCA

pca = PCA(n_components=K)  # K = 100
pca.fit(X.T)
# projection matrix
U = pca.components_.T
