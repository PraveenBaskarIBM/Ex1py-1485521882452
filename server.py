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
import cPickle as pickle
from sklearn.feature_selection import RFE
#from pandas import read_csv
from sklearn.linear_model import LogisticRegression

## code ends

httpd = Server(("", PORT), Handler)
try:
  print("Start serving at port %i" % PORT)
  httpd.serve_forever()
except KeyboardInterrupt:
  pass
httpd.server_close()

