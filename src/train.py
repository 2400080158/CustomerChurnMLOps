
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
import joblib

# Create project folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create a sample customer dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

columns = [
    "tenure", "monthly_charges", "total_charges",
    "support_calls", "contract_type", "payment_delay",
    "usage_score", "complaints", "age", "service_count"
]

df = pd.DataFrame(X, columns=columns)
df["churn"] = y
df.to_csv("data/customer_churn.csv", index=False)

# Prepare data
X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model and track experiment
mlflow.set_experiment("Customer-Churn-Prediction")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        model,
        name="churn_model",
        serialization_format="pickle"
    )

    joblib.dump(model, "../models/churn_model.pkl")

    print("Model training completed!")
    print("Accuracy:", round(accuracy, 4))
    print(classification_report(y_test, predictions))
    print("Model saved to models/churn_model.pkl")
