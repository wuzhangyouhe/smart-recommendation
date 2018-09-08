from __future__ import print_function
import json
import boto3

print('Loading function')
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


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

def top_cosine_similarity(data, deal_id, top_n):
    index = deal_id - 1 # Deal id starts from 1
    deal_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(deal_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Helper function to print top N similar movies
def print_similar_deals(deal_data, deal_id, top_indexes):
    print('\n Recommendations for {0}: \n'.format(deal_data[deal_data.deal_id == deal_id].deal_title.values[0]))
    rcmmd = {}
    for id in top_indexes[1:]:
        rcmmd[id] = deal_data[deal_data.deal_id == id].deal_title.values[0]
    print json.dumps(rcmmd, indent=4, separators=(',',':'))

s3_client = boto3.client('s3') #low-level functional API
resource = boto3.resource('s3') #high-level object-oriented API
bucket = resource.Bucket('chooosieprototype') #subsitute this for your s3 bucket name.

def lambda_handler(event, context):
    user_data = s3_client.get_object(Bucket=bucket, Key='u.user')
    deal_data = s3_client.get_object(Bucket=bucket, Key='u.item1')
    rating_data = s3_client.get_object(Bucket=bucket, Key='u.data')

    x = readDataset(user_data, deal_data, rating_data)
    users = x.getRatings().user_id.unique().shape[0]
    deals = x.getRatings().deal_id.unique().shape[0]
    current_data = np.zeros((users, deals))
    # SVD(Singular value decomposition) algorithm
    u, s, vt = svds(current_data, k = 50)
    trigger_event = json.loads(event)
    k = 50
    deal_id = trigger_event["deal id"] # Grab an id from items table
    top_n = 11

    sliced = vt.T[:, :k] # representative data
    indexes = top_cosine_similarity(sliced, deal_id, top_n)
    result = print_similar_deals(x.getDeals(), deal_id, indexes)
    return result
