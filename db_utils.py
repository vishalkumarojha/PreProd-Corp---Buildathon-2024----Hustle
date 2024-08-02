from pymongo import MongoClient

# MongoDB connection details
MONGO_URI = 'mongodb://localhost:27017'
DATABASE_NAME = 'automl_db'
COLLECTION_NAME = 'datasets'

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

def insert_dataset(data):
    """Insert dataset into MongoDB collection."""
    result = collection.insert_many(data)
    return result.inserted_ids

def fetch_dataset():
    """Fetch dataset from MongoDB collection."""
    return list(collection.find())
