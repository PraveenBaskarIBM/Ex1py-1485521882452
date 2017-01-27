import os
import pymongo
import json
from pymongo import MongoClient

try:
  from SimpleHTTPServer import SimpleHTTPRequestHandler as Handler
  from SocketServer import TCPServer as Server
except ImportError:
  from http.server import SimpleHTTPRequestHandler as Handler
  from http.server import HTTPServer as Server

# Read port selected by the cloud for our application
PORT = int(os.getenv('PORT', 8000))
# Change current directory to avoid exposure of control files
os.chdir('static')

# VCAP_SERVICES mapping Start

services = os.getenv('VCAP_SERVICES')
services_json = json.loads(services)
mongodb_url = services_json['compose-for-mongodb'][0]['credentials']['uri']
#connect:
client = MongoClient(mongodb_url)  
#get the default database:
db = client.get_default_database()  
print('connected to mongodb!, welcome to mongodb connection, have a fun') 

##Code Begins
# Feature Extraction with RFE
import numpy as np
import pandas as pd
import sklearn
#from pandas import read_csv
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
import cPickle as pickle

# load data
data = pd.read_csv('p_train1.csv', header=None, names=['INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.7)(A)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_1_(A.11)(B)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_2_(A.3)(D)[R]',
                                                                                                                         'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_1_(A.10)(E)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.6)(A)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.2)(C)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.7)(C)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_2_(A.3)(C)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.5)(A)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.2)(I)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.5)(C)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_2_(A.3)(B)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.8)(A)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_5_(A.9)(C)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.2)(A)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_5_(A.9)(D)[R]','Risk'])

#print data
data[np.isnan(data)] = 0

array = data.values
#print array
X = array[:,0:16]
Y = array[:,16]
print X
print Y

# feature extraction
model = LogisticRegression()
rfe = RFE(model, 16)
fit = rfe.fit(X, Y)

print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
print("Feature Estimator: %s") % fit.estimator

with open('test3_model.pkl', 'wb') as f:
    pickle.dump(fit, f)
    
## code ends

httpd = Server(("", PORT), Handler)
try:
  print("Start serving at port %i" % PORT)
  httpd.serve_forever()
except KeyboardInterrupt:
  pass
httpd.server_close()

