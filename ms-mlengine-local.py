from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
import json
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from array import array
import csv
import subprocess
import pprint

app = Flask(__name__)
api = Api(app)

class readDataset :

    def __init__(self, userPath, ratingPath, dealPath):
        self.up = userPath
        self.rp = ratingPath
        self.dp = dealPath

    def getUsers (self):
        u_cols = ['user_id', 'age']
        users = pd.read_csv(self.up, sep=',', names = u_cols, encoding='latin-1')
        return users

    def getRatings (self):
        r_cols = ['user_id', 'deal_id', 'rating']
        ratings = pd.read_csv(self.rp, sep=',', names = r_cols, encoding='latin-1')
        return ratings

    def getDeals (self):
        d_cols = ['deal_id', 'father_day' , 'ramadan' , 'new_year_day' , 'christmas_day' , 'july_14' , 'mother_day' , 'super_sunday' , 'fantastic_friday' , 'discount_5' , 'discount_10' , 'discount_15' , 'discount_20' , 'discount_25' , 'discount_30' , 'discount_35' , 'discount_40' , 'discount_45' , 'discount_50' , 'discount_over_50']
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
    rcmmd = []
    for id in top_indexes[1:]:
        child_rcmmd = {}
        child_rcmmd['Deal id'] = id
        rcmmd.append(child_rcmmd)
    output= json.dumps(rcmmd, indent=4, separators=(',',':'))
    print output
    return output

class Users(Resource):

    def __init__(self):
        self.u = None
    
    def post(self):
        content = request.get_json(force=True)
        with open('users.csv', mode='w') as csv_file:
            columns = ['user_id', 'age']
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            for data in content: 
                writer.writerow({'user_id': data['user_id'], 'age': data['age']})
        return '{ "message":"Users posted successful!" }'

    def get(self):
        return self.u

class Deals(Resource):
    def post(self):
        content = request.get_json(force=True)
        with open('deals.csv', mode='w') as csv_file:
            columns = ['deal_id', 'father_day' , 'ramadan' , 'new_year_day' , 'christmas_day' , 'july_14' , 'mother_day' , 'super_sunday' , 'fantastic_friday' , 'discount_5' , 'discount_10' , 'discount_15' , 'discount_20' , 'discount_25' , 'discount_30' , 'discount_35' , 'discount_40' , 'discount_45' , 'discount_50' , 'discount_over_50']
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            for data in content: 
                writer.writerow({'deal_id': data[columns[0]], 'father_day': data[columns[1]], 'ramadan': data[columns[2]], 'new_year_day': data[columns[3]], 'christmas_day': data[columns[4]], 'july_14': data[columns[5]], 'mother_day': data[columns[6]], 'super_sunday': data[columns[7]], 'fantastic_friday': data[columns[8]], 'discount_5': data[columns[9]] , 'discount_10': data[columns[10]] , 'discount_15': data[columns[11]] , 'discount_20': data[columns[12]] , 'discount_25': data[columns[13]] , 'discount_30': data[columns[14]] , 'discount_35': data[columns[15]] , 'discount_40': data[columns[16]] , 'discount_45': data[columns[17]] , 'discount_50': data[columns[18]] , 'discount_over_50': data[columns[19]]})
        return '{ "message":"Deals posted successful!" }'

class Ratings(Resource):
    def post(self):
        content = request.get_json(force=True)
        with open('rates.csv', mode='w') as csv_file:
            columns = ['user_id', 'deal_id', 'rates']
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            for data in content: 
                writer.writerow({'user_id': data['user_id'], 'deal_id': data['deal_id'],'rates': data['rates']})
        return '{ "message":"Rates posted successful!" }'
        
class rmdDeals(Resource):
    def get(self, clicked_deal_id):
        x = readDataset('users.csv', 'rates.csv', 'deals.csv')        
        users = x.getRatings().user_id.unique().shape[0]
        deals = x.getRatings().deal_id.unique().shape[0]
        current_data = np.zeros((users, deals))
        # SVD(Singular value decomposition) algorithm
        u, s, vt = svds(current_data, k = 50)
        k = 50
        deal_id = int(clicked_deal_id)
        top_n = 11

        sliced = vt.T[:, :k] # representative data
        indexes = top_cosine_similarity(sliced, deal_id, top_n)
        result = print_similar_deals(x.getDeals(), deal_id, indexes)
        return result

class resetData(Resource):
    def post(self):
        #content = request.get_json(force=True)
        subprocess.call(["sh", "reset-dataset.sh"])
        return '{ "message":"Reset successful!" }'

api.add_resource(Users, '/submitUsers') # Route_1
api.add_resource(Deals, '/submitDeals') # Route_2
api.add_resource(Ratings, '/submitRatings') # Route_3
api.add_resource(rmdDeals, '/getRmdDeals/<clicked_deal_id>') # Route_4
api.add_resource(resetData, '/resetData') # Route_5

if __name__ == '__main__':
     app.run(host='0.0.0.0',port='9080')
