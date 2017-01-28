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
import pandas as pd
import numpy as np
import cPickle as pickle

data = pd.read_csv('p_test1.csv', header=None, names=['Risk','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.7)(A)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_1_(A.11)(B)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_2_(A.3)(D)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_1_(A.10)(E)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.6)(A)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.2)(C)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.7)(C)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_2_(A.3)(C)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.5)(A)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.2)(I)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.5)(C)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_2_(A.3)(B)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.8)(A)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_5_(A.9)(C)[R]','INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.2)(A)[R]',
                                                   'INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_5_(A.9)(D)[R]'])

data[np.isnan(data)] = 0
print data

print('\n')
with open('test3_model.pkl', 'rb') as f:
    classifier = pickle.load(f)
    del data['Risk']
    #print data

    predicted_set = classifier.predict(data)
    prob_predicted = classifier.predict_proba(data)

    data = pd.DataFrame(data, columns=["INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.7)(A)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_1_(A.11)(B)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_2_(A.3)(D)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_1_(A.10)(E)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.6)(A)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.2)(C)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.7)(C)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_2_(A.3)(C)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.5)(A)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.2)(I)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.5)(C)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_2_(A.3)(B)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_4_(A.8)(A)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_5_(A.9)(C)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_3_(A.2)(A)[R]",
                                       "INSTRUCTION_OBTAINED_SCORE.STAAR_v2014_Cat_5_(A.9)(D)[R]"])
    pred = pd.DataFrame(predicted_set, columns=["Expected_Risk"])
    df_prob = pd.DataFrame(prob_predicted, columns=["Risk_Prob_A", "Risk_Prob_B"])

    frame1 = [data, pred]
    df1 = pd.concat(frame1, axis=1, join_axes=[data.index])
    frame2 = [df1, df_prob]
    df2 = pd.concat(frame2, axis=1, join_axes=[data.index])
    #print df2
    df2['At_Risk'] = df2['Risk_Prob_B'].map(lambda x: 'Low' if x < 0.5 else 'Medium' if x < 0.75 else 'High')
    print(df2)
    #return df2

## code ends

##VCAP services End

httpd = Server(("", PORT), Handler)
try:
  print("Start serving at port %i" % PORT)
  httpd.serve_forever()
except KeyboardInterrupt:
  pass
httpd.server_close()

