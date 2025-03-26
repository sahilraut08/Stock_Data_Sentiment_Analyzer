from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

fine_tuned_model_path = "./models/finbert_finetuned" 
sentiment_pipeline = pipeline("text-classification", model=fine_tuned_model_path)


@app.post("/predict/")
def predict(text: str):
    result = sentiment_pipeline(text)
    return {"sentiment": result[0]["label"], "confidence": result[0]["score"]}
