from fastapi import FastAPI
from pydantic import BaseModel, conlist
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Logistic Regression API")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Logistic Regression API")

# Allow frontend to access backend
origins = [
    "*",  # allow all origins (for testing); later restrict to your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_model.joblib"
model = joblib.load(MODEL_PATH)

# Pydantic model for input (Pydantic v2 syntax)
class PredictRequest(BaseModel):
    features: conlist(item_type=float, min_length=30, max_length=30)

@app.get("/")
def home():
    return {"message": "Logistic Regression API is running"}

@app.post("/predict")
def predict(request: PredictRequest):
    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0].tolist()
    return {
        "prediction": int(prediction),
        "probabilities": probability
    }
