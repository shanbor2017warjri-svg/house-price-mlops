# Importing required libraries
import numpy as np
import pandas as pd
import joblib
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from pathlib import Path
import mlflow


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "house_prices.csv"

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# Clean price column
df["price"] = (
    df["price"]
    .astype(str)
    .str.replace(r"[^\d.\-]", "", regex=True)
)

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

# Check missing values after cleaning
print("Missing values:")
print(df.isnull().sum())

# Do not train on missing or invalid values
if df.isnull().any().any():
    raise ValueError(
        "Dataset contains missing or invalid values. "
        "Review the CSV before training."
    )

# Select independent variables
X = df[
    ["area_sqft", "bedrooms", "bathrooms", "house_age"]
]

# Select dependent variable
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: ₹{mae:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R² Score: {r2:.4f}")


BASE_DIR = Path(__file__).resolve().parent

# Create a SQLite tracking database
MLFLOW_DB = BASE_DIR / "mlflow.db"

# Convert Windows path into a valid SQLite URI
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

# Configure MLflow
mlflow.set_tracking_uri(TRACKING_URI)

# Create or select the experiment
mlflow.set_experiment("House Price Prediction")

print("MLflow tracking URI:", TRACKING_URI)


with mlflow.start_run(run_name="Linear Regression Baseline"):

    # Model parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("fit_intercept", model.fit_intercept)
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("random_state", 42)

    # Evaluation metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Trained model artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        name="house_price_model",
        input_example=X_train.head(3)
    )

    print("MLflow experiment logged successfully.")

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "house_price_model.joblib"

joblib.dump(model, MODEL_PATH)

print("Model saved:", MODEL_PATH)