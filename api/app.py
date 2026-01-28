from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

pipeline = joblib.load('../models/sentiment_pipeline_v1.0.0.pkl')

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict

@app.post("/predict", response_model=PredictionResponse)
def predict(review: ReviewRequest):
    result = predict_sentiment(review.text, pipeline) # obviously update with own method that I define
    return result

@app.get("/health")
def health():
    return {"status": "healthy", "model_version": "v1.0.0"}
