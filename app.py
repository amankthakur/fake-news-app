
from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load(os.path.join("model", "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join("model", "vectorizer.pkl"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form.get("news", "")
    if not news.strip():
        return render_template("index.html", prediction_text="Please enter some news text to evaluate.")

    try:
        vec_input = vectorizer.transform([news])
        result = model.predict(vec_input)[0]
        label = "FAKE" if result == 1 else "REAL"
        return render_template("index.html", prediction_text=f"This news is likely {label}.")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
