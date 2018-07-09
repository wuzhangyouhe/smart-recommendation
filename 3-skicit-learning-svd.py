import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'food_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

#Reading items file:
i_cols = ['food_id', 'food_title' ,'publish date','food release date', 'IMDb URL', 'yeast', ' salt', ' coconut oil',
 'Deluxe Nut Mix	', ' sugar', 'honey', 'pecans', 'wheat', ' organic oat flour', 'onion',
 ' garlic', 'malt', 'barley', 'soybeans', ' citric acid', 'potato', 'eggs', 'vegetable oil', 'fruit']
items = pd.read_csv('ml-100k/u.item1', sep=',', names=i_cols,
 encoding='latin-1')

# print user
print users.shape
print users.head()
print users.describe()
users.hist()
plt.show()

# print rating
print ratings.shape
print ratings.head()
print ratings.describe()
ratings.hist()
plt.show()

# print items
print items.shape
print items.head()
print items.describe()
items.hist(figsize=(16,16))
plt.show()

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header, encoding='latin-1')

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size = 0.20)

# Memory-based collaborative filtering
print "1, Memory-based collaborative filtering \n"
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
print train_data_matrix, line
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))
print user_prediction[0], '\n', item_prediction[0]

# Model-based collaborative filter
print "2, Model-based collaborative filter \n"
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print 'The sparsity level of Food amount 100k is ' +  str(sparsity*100) + '%'

# SVD(Singular value decomposition) algorithm
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print 'User-based CF RMSE: ' + str(rmse(X_pred, test_data_matrix))

def top_cosine_similarity(data, food_id, top_n=10):
    index = food_id - 1 # Food id starts from 1
    food_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(food_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Helper function to print top N similar movies
def print_similar_movies(food_data, food_id, top_indexes):
    print('\n Recommendations for {0}: \n'.format(food_data[food_data.food_id == food_id].food_title.values[0]))
    for id in top_indexes + 1:
        print(food_data[food_data.food_id == id].food_title.values[0])
        
k = 50
food_id = 10 # Grab an id from items table
top_n = 10

sliced = vt.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, food_id, top_n)
print_similar_movies(items, food_id, indexes)

## SVD and PCA relationship
#normalised_mat = ratings_mat - np.matrix(np.mean(ratings_mat, 1)).T
#cov_mat = np.cov(normalised_mat)
#evals, evecs = np.linalg.eig(cov_mat)
#
#sliced = evecs[:, :k] # representative data
#top_indexes = top_cosine_similarity(sliced, movie_id, top_n)
#print_similar_movies(items, movie_id, top_indexes)