from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
import json
#from flask.ext.jsonpify import jsonify
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from array import array

db_connect = create_engine('sqlite:///localEngine.db')
app = Flask(__name__)
api = Api(app)

#class readDataset :
#
#    def __init__(self, userPath, ratingPath, dealPath):
#        self.up = userPath
#        self.rp = ratingPath
#        self.dp = dealPath
#
#    def getUsers (self):
#        u_cols = ['user_id', 'name']
#        users = pd.read_csv(self.up, sep='|', names = u_cols, encoding='latin-1')
#        return users
#
#    def getRatings (self):
#        r_cols = ['user_id', 'deal_id', 'rating']
#        ratings = pd.read_csv(self.rp, sep='|', names = r_cols, encoding='latin-1')
#        return ratings
#
#    def getDeals (self):
#        d_cols = ['deal_id', 'father_day' , 'ramadan' , 'new_year_day' , 'christmas_day' , 'july_14' ,'mother_day' , 'super_sunday' , 'fantastic_friday' , 'discount_5' , 'discount_10' , 'discount_15' , 'discount_20' , 'discount_25' , 'discount_30' , 'discount_35' , 'discount_40' , 'discount_45' , 'discount_50' , 'discount_over_50']
#        deals = pd.read_csv(self.dp, sep='|', names = d_cols, encoding='latin-1')
#        return deals

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

class sqlread :
    def getUsers(self):
        conn = db_connect.connect() # connect to database
        query = conn.execute("select * from users") # This line performs query and returns json result
        return {'user_id': [i[0] for i in query.cursor.fetchall()]} # 

    def getDeals(self):
        conn = db_connect.connect() # connect to database
        query = conn.execute("select * from deals") # This line performs query and returns json result
        return {'deal_id': [i[0] for i in query.cursor.fetchall()]} # Fetches first column that is deal ID

    def getRatings(self):
        conn = db_connect.connect() # connect to database
        query = conn.execute("select * from ratings") # This line performs query and returns json result
        return {'user-id': [i[0] for i in query.cursor.fetchall()]} # Fetches first column that is rating ID 

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
    rcmmd = []
    for id in top_indexes[1:]:
        child_rcmmd = {}
        child_rcmmd['Deal id'] = id
        child_rcmmd['Deal name'] = deal_data[deal_data.deal_id == id].deal_title.values[0]
        rcmmd.append(child_rcmmd)
    output= json.dumps(rcmmd, indent=4, separators=(',',':'))
    print output
    return output

@app.route('/submitUsers', methods=['POST']) 
def postUser():
    content = request.get_json()
    user0 = array('i')
    user1 = array('l')
    columns = ['user_id', 'age']
    for data in content:                
        user0.append(data['user_id'])
    for data in content:
        user1.append(data['age'])
    print user0, user1

    return '{ "message":"Users posted successful!" }'
            

class Deals(Resource):
    def post(self):
        conn = db_connect.connect()
        query = conn.execute('''insert into deals(deal_id, father_day , ramadan , new_year_day , christmas_day , july_14 , mother_day , super_sunday , fantastic_friday , discount_5 , discount_10 , discount_15 , discount_20 , discount_25 , discount_30 , discount_35 , discount_40 , discount_45 , discount_50 , discount_over_50 ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''')
        db.commit()

class Ratings(Resource):
    def post(self):
        conn = db_connect.connect()
        query = conn.execute('''insert into ratings(user-id, deal_id, rates) VALUES(?,?,?)''')
        db.commit()
        
class rmdDeals(Resource):
    def get(self, clicked_deal_id):
        #x = readDataset(sqlread.getUsers, sqlread.getDeals, sqlread.getRatings)
        x = readDataset('ml-100k/u.user', 'ml-100k/u.data', 'ml-100k/u.item1')
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

#api.add_resource(Users, '/submitUsers') # Route_1
api.add_resource(Deals, '/submitDeals') # Route_2
api.add_resource(Ratings, '/submitRatings') # Route_3
api.add_resource(rmdDeals, '/getRmdDeals/<clicked_deal_id>') # Route_4

if __name__ == '__main__':
     app.run(port='8080')
