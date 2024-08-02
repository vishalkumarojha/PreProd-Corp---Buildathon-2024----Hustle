from flask import jsonify, request
import pandas as pd

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Reading the file into a Pandas DataFrame
    df = pd.read_csv(file)
    data_dict = df.to_dict("records")

    # Insert data into MongoDB
    collection_name = file.filename.split('.')[0]
    automl = "your_automl_value"  # Define the automl variable with your desired value
    db[automl].insert_many(data_dict)

    return jsonify({"message": "Dataset uploaded successfully"}), 200
