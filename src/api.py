from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from src.schemas import HouseFeatures, PredictionOutput
import uvicorn
import os

app = FastAPI(title="House Price Prediction API", description="API to predict house prices based on features.")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = 'models/house_price_model.joblib'
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Please run src/train.py first.")

model = joblib.load(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API is running!"}

@app.post("/predict", response_model=PredictionOutput)
def predict_price(features: HouseFeatures):
    try:
        # Convert input features to DataFrame
        input_df = pd.DataFrame([features.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return PredictionOutput(predicted_price=round(float(prediction), 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
