from __future__ import print_function
import numpy as np
import pandas as pd

# Reading user file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('recomendation-system/u.user', sep='|', names=u_cols)
n_users = users.shape[0]
print('Number of users:', n_users)
# Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('recomendation-system/ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('recomendation-system/ua.test', sep='\t', names=r_cols)
rate_train = ratings_base.as_matrix()
rate_test = ratings_test.as_matrix()
print('Number of traing rates:', rate_train.shape[0])
print('Number of test rates:', rate_test.shape[0])

# Reading items file:
i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL',
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
          'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('recomendation-system/u.item', sep='|', names=i_cols)
n_items = items.shape[0]
print('Number of items:', n_items)
