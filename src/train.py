import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    df.drop(['customerID'], axis = 1, inplace = True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce').fillna(0)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

data_path = "/Users/mac/Desktop/MLOps/churn_predictor/data/data.csv"

print(f"Loading data from {data_path}...")

df = load_and_clean_data(data_path)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

params = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

print("Starting training run...")
mlflow.set_experiment("Telco_Churn_Prediction") # Groups runs together

with mlflow.start_run():
    
    # A. Log Parameters
    mlflow.log_params(params)
    
    # B. Train Model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # C. Evaluate
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # D. Log Metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # E. Log Model Artifact
    # This saves the model object so we can use it later for the API
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print("Run successful! View results in 'mlflow ui'")