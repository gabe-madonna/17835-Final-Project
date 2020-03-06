import pymongo

client = pymongo.MongoClient('mongodb://deltatrainer:legengerry@18.217.164.108:27017/admin?retryWrites=false', serverSelectionTimeoutMS=10)
db = client["admin"]["17835"]

db.insert({})
db.find_one({})

api_key = "65bULXQXhB9DD9MWtiWmuj12Y"
secret_key = "fpJFhiGd2iYQjauxCggUZYMY7mmmBRJzTJ7KQfk6pSIAYCoLSn"

access_token = "1233410547770486784-k0gSyLxHZkQTQdyOpLlpdXCASKYK9A"
access_token_secret = "THLQdZICRH5P6fXUR5pT3LSjCg0SACBBkSn6uKKWFNqDn"



