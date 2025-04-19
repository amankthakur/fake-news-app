from flask import Flask, request, render_template, redirect
import joblib
import os

app = Flask(__name__)

# Load the model and vectorizer
MODEL_PATH = os.path.join("model", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        # Handle direct browser visit by redirecting to homepage
        return redirect("/")

    news = request.form.get("news", "")
    if not news.strip():
        return render_template("index.html", prediction_text="Please enter some news text to evaluate.")

    try:
        vec_input = vectorizer.transform([news])
        result = model.predict(vec_input)[0]
        label = "FAKE" if result == 1 else "REAL"
        return render_template("index.html", prediction_text=f"This news is likely {label}.")
    except Exception as e:
        print("Prediction Error:", e)
        return render_template("index.html", prediction_text="An error occurred during prediction. Please try again.")

# Handle Method Not Allowed errors
@app.errorhandler(405)
def method_not_allowed(e):
    return render_template("index.html", prediction_text="Invalid access method — please use the form."), 405

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
