# 📰 Veracity Vigilance - Fake News Detection Web App

A full-stack web application that uses Machine Learning to detect fake news articles. Built with Flask, trained on Kaggle's Fake/True news datasets, and deployed for real-time predictions.

---

## 🔍 Features

- Detects whether a news article is **FAKE** or **REAL**
- Clean, responsive UI with glassmorphism & animations
- Trained using multiple ML models (Random Forest performs best)
- Built using Flask, Bootstrap, and scikit-learn
- Deployed on Render (or suitable hosting service)

---

## 📁 Project Structure

```
project/
├── app.py                  # Flask backend logic
├── templates/
│   └── index.html         # Frontend UI with Bootstrap
├── model/
│   ├── fake_news_model.pkl  # Trained Random Forest model
│   └── vectorizer.pkl       # TF-IDF vectorizer
├── test_model.py          # Local test script for batch predictions
├── Fake.csv               # Fake news dataset
├── True.csv               # Real news dataset
├── requirements.txt       # Python dependencies
```

---

## 🧠 Model Training

- **Data:** Kaggle Fake.csv + True.csv merged with binary labels (1 = FAKE, 0 = REAL)
- **Preprocessing:**
  - Lowercasing
  - Removing punctuation
  - Removing stopwords (NLTK)
- **Vectorization:**
  - TF-IDF with `max_features=5000`
- **Models Tried:**
  - Logistic Regression
  - SVM
  - Random Forest ✅ (Best F1 Score)
  - CNN / LSTM (Deep learning)
- **Model Saving:**
  ```python
  joblib.dump(model, "model/fake_news_model.pkl")
  joblib.dump(vectorizer, "model/vectorizer.pkl")
  ```

---

## 🌐 Web App Flow

- `/` ➜ Homepage (input form)
- `/predict` ➜ Accepts POST, returns prediction
- Flash red if FAKE, green if REAL

---

## 🚀 Deployment

1. Add `requirements.txt`:
   ```
   Flask
   scikit-learn
   joblib
   gunicorn
   newspaper3k
   ```
2. Deploy on **Render**, **Heroku**, or similar

---

## 🧪 Testing (test_model.py)
```python
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# Example usage
tests = [
    "Breaking: Chocolate cures cancer!",
    "NASA successfully lands on Mars",
]
for t in tests:
    print(t, "=>", "FAKE" if predict(t) else "REAL")
```

---

## ✨ Credits
- Aman Kumar Thakur (Developer)
- Datasets: Kaggle - Fake and Real News Dataset
- Libraries: Flask, scikit-learn, NLTK, Bootstrap

---

## 📬 Contributions & Feedback
Feel free to fork, contribute or raise issues. Star ⭐ the repo if you find it useful!
