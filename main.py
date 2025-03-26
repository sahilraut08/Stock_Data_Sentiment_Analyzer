import os
from fastapi import FastAPI
from transformers import pipeline
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = f"https://newsapi.org/v2/everything?q=stocks&language=en&apiKey={NEWS_API_KEY}"

fine_tuned_model_path = "./models/finbert_finetuned" 
sentiment_pipeline = pipeline("text-classification", model=fine_tuned_model_path)

@app.get("/news/")
def get_financial_news():
    """Fetches latest financial news articles"""
    response = requests.get(NEWS_URL)
    if response.status_code == 200:
        news_data = response.json()
        articles = [
            {"title": article["title"], "description": article["description"], "url": article["url"]}
            for article in news_data.get("articles", [])
        ]
        return {"news": articles}
    return {"error": "Failed to fetch news"}

@app.get("/news-sentiment/")
def get_news_with_sentiment():
    """Fetches financial news and performs sentiment analysis"""
    response = requests.get(NEWS_URL)
    if response.status_code == 200:
        news_data = response.json()
        analyzed_articles = []
        
        for article in news_data.get("articles", []):
            text = article["title"]
            sentiment = sentiment_pipeline(text)[0]
            analyzed_articles.append({
                "title": text,
                "sentiment": sentiment["label"],
                "confidence": sentiment["score"],
                "url": article["url"]
            })
        
        return {"news_sentiment": analyzed_articles}
    return {"error": "Failed to fetch news"}
