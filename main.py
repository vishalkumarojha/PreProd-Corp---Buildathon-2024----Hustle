from flask import Flask, request, jsonify
from pymongo import MongoClient
import os

# Initialize Flask app
from flask import Flask
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017"
mongo = PyMongo(app)

# MongoDB connection string
MONGO_URI = os.getenv('MONGO_URI', 'mmongodb://localhost:27017')
client = MongoClient(MONGO_URI)

# Select the database and collection
db = client['my_database']
collection = db['my_collection']

# Home route
@app.route('/')
def home():
    db.inventory.insert_one({"item": "canvas", "qty": 100, "tags": ["cotton"], "size": {"h": 28, "w": 35.5, "uom": "cm"}})
    return "Welcome to the Flask and MongoDB App!"

# Route to insert a document into the collection
@app.route('/insert', methods=['POST'])
def insert_document():
    data = request.json
    result = collection.insert_one(data)
    return jsonify({'message': 'Document inserted', 'id': str(result.inserted_id)})

# Route to retrieve documents from the collection
@app.route('/documents', methods=['GET'])
def get_documents():
    documents = list(collection.find({}, {'_id': 0}))
    return jsonify(documents)

# Route to update a document
@app.route('/update/<string:id>', methods=['PUT'])
def update_document(id):
    data = request.json
    result = collection.update_one({'_id': id}, {'$set': data})
    return jsonify({'message': 'Document updated', 'modified_count': result.modified_count})

# Route to delete a document
@app.route('/delete/<string:id>', methods=['DELETE'])
def delete_document(id):
    result = collection.delete_one({'_id': id})
    return jsonify({'message': 'Document deleted', 'deleted_count': result.deleted_count})

if __name__ == '__main__':
    app.run(debug=True)
