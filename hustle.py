import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

Dense = layers.Dense
import argparse

# Suppress TensorFlow and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

AVAILABLE_MODELS = {
    'SVM': make_pipeline(StandardScaler(), SVC()),
    'DecisionTree': DecisionTreeClassifier(),
    'Bagging': BaggingClassifier(),
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'Stacking': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()),
            ('svm', make_pipeline(StandardScaler(), SVC()))
        ],
        final_estimator=LogisticRegression(max_iter=1000)
    )
}

def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(data, target_feature, test_size=0.2, random_state=42):
    X = data.drop(columns=[target_feature])
    y = data[target_feature]
    
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_neural_network(input_dim):
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_pred = np.argmax(model.predict_proba(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return accuracy, rmse

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models_to_train, model_output_folder):
    results = {}
    for name, model in AVAILABLE_MODELS.items():
        if name in models_to_train:
            model.fit(X_train, y_train)
            accuracy, rmse = evaluate_model(model, X_test, y_test)
            results[name] = {'accuracy': accuracy, 'rmse': rmse}
            with open(os.path.join(model_output_folder, f'{name}.pkl'), 'wb') as f:
                pickle.dump(model, f)
    
    if 'NeuralNetwork' in models_to_train:
        input_dim = X_train.shape[1]
        nn_model = build_neural_network(input_dim)
        nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        nn_model.save(os.path.join(model_output_folder, 'NeuralNetwork.keras'))
        nn_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)[1]
        nn_rmse = root_mean_squared_error(y_test, nn_model.predict(X_test).flatten())
        results['NeuralNetwork'] = {'accuracy': nn_accuracy, 'rmse': nn_rmse}
    
    return results

def main(input_file, target_feature, model_output_folder, models_to_train):
    data = load_data(input_file)
    X_train, X_test, y_train, y_test = split_data(data, target_feature)
    
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)
    
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, models_to_train, model_output_folder)
    
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}, RMSE = {metrics['rmse']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate various ML models.")
    parser.add_argument('input_file', type=str, help="Path to the input CSV file.")
    parser.add_argument('--target_feature', type=str, default='target', help="The target feature for prediction.")
    parser.add_argument('--model_output_folder', type=str, default='models', help="Folder to save the trained models.")
    parser.add_argument('--models', type=str, nargs='+', choices=list(AVAILABLE_MODELS.keys()) + ['NeuralNetwork'], 
                        default=list(AVAILABLE_MODELS.keys()) + ['NeuralNetwork'], 
                        help="List of models to train. If not provided, all models will be trained.")
    
    args = parser.parse_args()
    main(args.input_file, args.target_feature, args.model_output_folder, args.models)