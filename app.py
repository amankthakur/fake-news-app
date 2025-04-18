from flask import Flask, request, render_template, session
import joblib, os
from newspaper import Article

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # For session-based history

# Load your pretrained model and vectorizer
model = joblib.load(os.path.join("model", "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join("model", "vectorizer.pkl"))

@app.route("/", methods=["GET"])
def index():
    history = session.get("history", [])
    return render_template("index.html", history=history)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news_text", "").strip()
    url  = request.form.get("news_url", "").strip()

    if not text and not url:
        return "<div class='alert alert-warning'>Please enter news text or a URL.</div>"

    if url:
        try:
            art = Article(url)
            art.download(); art.parse()
            text = art.text
        except Exception:
            return "<div class='alert alert-danger'>Could not fetch article from URL.</div>"

    try:
        vec_input = vectorizer.transform([text])
        result = model.predict(vec_input)[0]
        label = "FAKE" if result == 1 else "REAL"
        out = f"<div class='alert alert-info'><strong>Prediction:</strong> {label}</div>"

        # Save to session history
        history = session.get("history", [])
        history.insert(0, {"text": text[:100] + '...', "result": label})
        session["history"] = history[:10]

        return out
    except Exception as e:
        return f"<div class='alert alert-danger'>Error during prediction: {str(e)}</div>"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
