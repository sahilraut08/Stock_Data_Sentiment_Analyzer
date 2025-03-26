# 📊 Stock Sentiment Analyzer

## 🔥 Overview
Stock Sentiment Analyzer is an AI-powered application that **analyzes the sentiment** of financial news using **FinBERT**, a transformer-based NLP model fine-tuned on financial data. This project integrates **FastAPI** for a robust API and supports real-time news sentiment analysis.

## 🚀 Features
✅ **Fine-tuned FinBERT model** for financial sentiment analysis  
✅ **FastAPI** for real-time inference  
✅ **Live News Integration** for realistic market insights  
✅ **Postman API testing** for quick debugging  

---

## 📂 Project Structure
```
📦 Stock_Sentiment_Analyzer
├── 📄 app.py  # FastAPI app for sentiment prediction
├── 📄 finetune.py  # Fine-tuning script for FinBERT
├── 📄 README.md  # Project documentation
```

---

## 🏗️ Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<yourusername>/Stock_Sentiment_Analyzer.git
cd Stock_Sentiment_Analyzer
```

### 2️⃣ Install Dependencies
```bash
pip install transformers fastapi uvicorn
```

### 3️⃣ Run the API
```bash
uvicorn app:app --reload
```

### 4️⃣ Test with Postman
Send a **POST** request to `http://127.0.0.1:8000/predict/` with the following JSON:
```json
{
  "text": "The stock market is crashing due to economic uncertainty."
}
```

You should receive:
```json
{
  "sentiment": "negative",
  "confidence": 0.98
}
```

---

## 📡 Connecting to Live News 📰
To make your analysis more **realistic**, integrate **Yahoo Finance API** or **Alpha Vantage** to fetch **real-time** financial news.

Example integration:
```python
import requests
url = "https://newsapi.org/v2/everything?q=stocks&apiKey=YOUR_API_KEY"
response = requests.get(url).json()
```

---

## 🏆 Future Improvements
✨ Integrate **Reinforcement Learning** for better sentiment analysis  
✨ Deploy on **AWS Lambda** for serverless inference  
✨ Add **Telegram Bot** for instant stock sentiment notifications  

---

## 🤝 Contributing
Contributions are welcome! Feel free to submit **issues** or **pull requests**.

📩 For any queries, reach out via [LinkedIn](https://www.linkedin.com/in/sahilraut8/).

---

## ⭐ Show Your Support!
If you found this useful, **give it a star ⭐** on GitHub! 🎉

