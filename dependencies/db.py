import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
mongo = MongoClient(os.getenv('MONGODB_URI'))

class MongoDB:
    def __init__(self):
        self.db = MongoClient(os.getenv('MONGODB_URI')).get_database()

