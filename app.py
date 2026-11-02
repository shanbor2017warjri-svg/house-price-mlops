from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"

FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "house_age"
]

model = None
# Load the trained model when the API starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield
    model = None

app = FastAPI(
    title="house_price_model",
    description="This app predicts house price using a trained model",
    version="1.0.0",
    lifespan=lifespan
)

# Input validation
class HouseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area_sqft: float = Field(ge=50, le=10000)
    bedrooms: int = Field(ge=1, le=10)
    bathrooms: int = Field(ge=1, le=10)
    house_age: int = Field(ge=0, le=100)


# Prediction response
class PredictionOutput(BaseModel):
    predicted_price: float
    currency: str


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.get("/status")
def status():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_price(house: HouseInput):

    input_df = pd.DataFrame(
        [house.model_dump()],
        columns=FEATURES
    )

    prediction = model.predict(input_df)[0]

    return PredictionOutput(
        predicted_price=round(max(0.0, float(prediction)), 2),
        currency="INR"
    )


