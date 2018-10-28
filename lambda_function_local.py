#!/usr/local/bin/python 
import json
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from BaseHTTPServer import BaseHTTPRequestHandler
import threading
import random

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
    rcmmd = []
    for id in top_indexes[1:]:
        child_rcmmd = {}
        child_rcmmd['Deal id'] = id
        child_rcmmd['Deal name'] = deal_data[deal_data.deal_id == id].deal_title.values[0]
        rcmmd.append(child_rcmmd)
    output= json.dumps(rcmmd, indent=4, separators=(',',':'))
    print output
    return output

def html_response():
    x = readDataset('ml-100k/u.user', 'ml-100k/u.data', 'ml-100k/u.item1')
    print x.getUsers().shape, '\n' , x.getRatings().shape, '\n' , x.getDeals().shape
    print x.getUsers().head(10) , '\n', x.getRatings().head(), '\n', x.getDeals().head()
    print x.getUsers().describe(), '\n', x.getRatings().describe(), '\n', x.getDeals().describe()

    users = x.getRatings().user_id.unique().shape[0]
    deals = x.getRatings().deal_id.unique().shape[0]
    current_data = np.zeros((users, deals))
    # SVD(Singular value decomposition) algorithm
    u, s, vt = svds(current_data, k = 50)
    k = 50
    deal_id = random.randint(1,200) # Grab an id from items table
    top_n = 11

    sliced = vt.T[:, :k] # representative data
    indexes = top_cosine_similarity(sliced, deal_id, top_n)
    result = print_similar_deals(x.getDeals(), deal_id, indexes)
    html='<h1 style="text-align:center;">Prototype - Demo Lambda event trigger function in smart engine</h1>'
    body="""<div style="text-align:center;">
	    <h3>Customer clicked Deal id: {0}</h3>
	    <h4>The deals recommendations for: {1}</h4>
	    <h4>Top 10 deals recommended from Smart engine</h4>
	    <table border=1 style="margin: 0 auto;">
	    <br>
	    {2}
	    </br>
	    </table>
	    </div>
	    """.format(deal_id, x.getDeals()[x.getDeals().deal_id == deal_id].deal_title.values[0],json.loads(result))
    html = html + body
    return html

class GetHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        self._set_headers()
        message = html_response()
        self.wfile.write(message.encode('utf-8'))

    def do_POST(self):
        self.do_GET()

if __name__ == '__main__':
    from BaseHTTPServer import HTTPServer

    server = HTTPServer(('localhost', 8000), GetHandler)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()
