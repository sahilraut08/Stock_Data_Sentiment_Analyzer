from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

sentiment_pipeline = pipeline("text-classification", model="yiyanghkust/finbert-tone")

@app.post("/predict/")
def predict(text: str):
    result = sentiment_pipeline(text)
    return {"sentiment": result[0]["label"], "confidence": result[0]["score"]}
