import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt
import json

class readDataset :

    def __init__(self, userPath, ratingPath, dealPath):
        self.up = userPath
        self.rp = ratingPath
        self.dp = dealPath

    def getUsers (self):
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        users = pd.read_csv(self.up, sep='|', names = u_cols, encoding='latin-1')
        return users

    def getRatings (self):
        r_cols = ['user_id', 'deal_id', 'rating', 'unix_timestamp']
        ratings = pd.read_csv(self.rp, sep='\t', names = r_cols, encoding='latin-1')
        return ratings

    def getDeals (self):
        d_cols = ['deal_id', 'deal_title', 'publish date', 'create date', 'IMDb URL', 'SG Fathers Day', ' SG Ramadan',
                  ' SG New Year day',
                  'SG Christmas', ' SG July 14th', 'SG Mothers Day', 'SG Super Sunday', 'SG Fantastic Friday', 'HK Fathers Day', 'HK Ramadan',
                  ' HK New Years day', 'HK Christmas', 'HK July 14th', 'HK Mothers Day', ' HK Super Sunday', 'HK Fantastic Friday', 'discount 25%', 'discount 50%',
                  'discount 75%']
        deals = pd.read_csv(self.dp, sep=',', names = d_cols, encoding='latin-1')
        return deals

x = readDataset('ml-100k/u.user', 'ml-100k/u.data', 'ml-100k/u.item1')
print x.getUsers().shape, '\n' , x.getRatings().shape, '\n' , x.getDeals().shape
print x.getUsers().head(10) , '\n', x.getRatings().head(), '\n', x.getDeals().head()
print x.getUsers().describe(), '\n', x.getRatings().describe(), '\n', x.getDeals().describe()

x.getUsers().hist(figsize=(6,6))
plt.show()
x.getRatings().hist()
plt.show()
x.getDeals().hist(figsize=(12,8))
plt.show()

# class train_function :
#
#     def __init__(self, ratingData):
#         self.ratingData = ratingData
#
#     def split_dataset(self):
#
#         users = self.ratingData.user_id.unique().shape[0]
#         deals = self.ratingData.deal_id.unique().shape[0]
#
#         train_data, test_data = cv.train_test_split(x.getRatings(), test_size=0.30)
#         train_data_matrix = np.zeros((users, deals))
#         for line in train_data.itertuples():
#             train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
#         print train_data_matrix, line
#         test_data_matrix = np.zeros((users, deals))
#         for line in test_data.itertuples():
#             test_data_matrix[line[1] - 1, line[2] - 1] = line[3]
#
#         return train_data_matrix, test_data_matrix
#
#     def rmse(self, prediction, ground_truth):
#         prediction = prediction[ground_truth.nonzero()].flatten()
#         ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#         return sqrt(mean_squared_error(prediction, ground_truth))
#
# n_users = train_function(x.getRatings()).split_dataset()

users = x.getRatings().user_id.unique().shape[0]
deals = x.getRatings().deal_id.unique().shape[0]

train_data, test_data = cv.train_test_split(x.getRatings(), test_size = 0.30)
train_data_matrix = np.zeros((users, deals))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
print train_data_matrix, line
test_data_matrix = np.zeros((users, deals))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

# Memory-based collaborative filtering
# print "1, Memory-based collaborative filtering \n"
# from sklearn.metrics.pairwise import pairwise_distances
# user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
# item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
#
# def predict(ratings, similarity, type='user'):
#     if type == 'user':
#         mean_user_rating = ratings.mean(axis=1)
#         ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
#         pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
#     elif type == 'item':
#         pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
#     return pred
#
# item_prediction = predict(train_data_matrix, item_similarity, type='item')
# user_prediction = predict(train_data_matrix, user_similarity, type='user')
#
#
# print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
# print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))
# print user_prediction[0], '\n', item_prediction[0]

# Model-based collaborative filter
sparsity=round(1.0-len(x.getRatings())/float(users*deals),3)
print 'The sparsity level of Food amount 100k is ' +  str(sparsity*100) + '%'

# SVD(Singular value decomposition) algorithm
# get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 50)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print 'User-based CF RMSE: ' + str(rmse(X_pred, test_data_matrix))

def top_cosine_similarity(data, deal_id, top_n=10):
    index = deal_id - 1 # Food id starts from 1
    food_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(food_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Helper function to print top N similar movies
def print_similar_movies(food_data, deal_id, top_indexes):
    print('\n Recommendations for {0}: \n'.format(food_data[food_data.deal_id == deal_id].deal_title.values[0]))
    rcmmd = []
    for id in top_indexes[1:]:
        child_rcmmd = {}
        child_rcmmd['Deal id'] = id
        child_rcmmd['Deal name'] = food_data[food_data.deal_id == id].deal_title.values[0]
        rcmmd.append(child_rcmmd)
    print json.dumps(rcmmd, indent=4, separators=(',',':'))
        
k = 50
deal_id = 110 # Grab an id from items table
top_n = 10

sliced = vt.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, deal_id, top_n)
print_similar_movies(x.getDeals(), deal_id, indexes)

# # SVD and PCA relationship
# normalised_mat = ratings_mat - np.matrix(np.mean(ratings_mat, 1)).T
# cov_mat = np.cov(normalised_mat)
# evals, evecs = np.linalg.eig(cov_mat)
#
# sliced = evecs[:, :k] # representative data
# top_indexes = top_cosine_similarity(sliced, food_id, top_n)
# print_similar_movies(x.getDeals(), food_id, top_indexes)