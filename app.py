from flask import Flask, request, render_template
import joblib
import os
from newspaper import Article

app = Flask(__name__)

# Load ML model and vectorizer
model = joblib.load(os.path.join("model", "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join("model", "vectorizer.pkl"))

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()
    news_url = request.form.get("news_url", "").strip()

    if not news_text and not news_url:
        return "<div class='alert alert-warning'>Please enter text or URL of an article to check.</div>"

    # Fetch article text if URL provided
    if news_url:
        try:
            article = Article(news_url)
            article.download()
            article.parse()
            news_text = article.text
        except Exception:
            return "<div class='alert alert-danger'>Failed to fetch article from URL. Please check the URL and try again.</div>"

    try:
        vec_input = vectorizer.transform([news_text])
        result = model.predict(vec_input)[0]
        label = "FAKE" if result == 1 else "REAL"
        return f"<div class='alert alert-info'><strong>Prediction:</strong> {label}</div>"
    except Exception as e:
        return f"<div class='alert alert-danger'>Error during prediction: {str(e)}</div>"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))