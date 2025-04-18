from flask import Flask, request, render_template, session
import joblib, os
from newspaper import Article

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # for session-based history

# load your pretrained model + vectorizer
model      = joblib.load(os.path.join("model", "fake_news_model.pkl"))
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
        vec   = vectorizer.transform([text])
        pred  = model.predict(vec)[0]
        label = "FAKE" if pred == 1 else "REAL"
        out   = f"<div class='alert alert-info'><strong>Prediction:</strong> {label}</div>"

        # save to session history
        h = session.get("history", [])
        h.insert(0, {"text": text[:100]+"…", "result": label})
        session["history"] = h[:10]

        return out

    except Exception as e:
        return f"<div class='alert alert-danger'>Error: {e}</div>"

if __name__ == "__main__":
    # use 0.0.0.0 & PORT in production; for local just running debug is fine
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
