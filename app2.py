import pandas as pd
import numpy as np
import json

# Number of rows and columns
num_rows = 5000
num_features = 8

# Generate random data
np.random.seed(0)
data = {
    f'feature{i+1}': np.random.rand(num_rows).tolist() for i in range(num_features)
}

# Adding some categorical features
data['feature9'] = np.random.choice(['A', 'B', 'C'], size=num_rows).tolist()
data['feature10'] = np.random.choice(['X', 'Y'], size=num_rows).tolist()

# Target column (binary classification)
data['target'] = np.random.choice([0, 1], size=num_rows).tolist()

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert DataFrame to JSON for MongoDB
dataset_json = json.loads(df.to_json(orient='records'))
