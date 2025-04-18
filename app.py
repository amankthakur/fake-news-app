from flask import Flask, request, render_template, session
import joblib, os, fitz, csv
from newspaper import Article
from werkzeug.utils import secure_filename
from io import StringIO

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # For session history

model = joblib.load(os.path.join("model", "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join("model", "vectorizer.pkl"))

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath):
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_csv(filepath):
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
        return "\n".join([" ".join(row) for row in rows if row])


def extract_text_from_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


@app.route("/", methods=["GET"])
def index():
    history = session.get('history', [])
    return render_template("index.html", history=history)


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()
    news_url = request.form.get("news_url", "").strip()
    result_html = ""

    if not news_text and not news_url:
        return "<div class='alert alert-warning'>Please enter text or URL of an article to check.</div>"

    if news_url:
        try:
            article = Article(news_url)
            article.download()
            article.parse()
            news_text = article.text
        except Exception:
            return "<div class='alert alert-danger'>Failed to fetch article from URL.</div>"

    try:
        vec_input = vectorizer.transform([news_text])
        result = model.predict(vec_input)[0]
        label = "FAKE" if result == 1 else "REAL"
        result_html = f"<div class='alert alert-info'><strong>Prediction:</strong> {label}</div>"

        # Store in session history
        history = session.get('history', [])
        history.insert(0, {"text": news_text[:200] + '...', "result": label})
        session['history'] = history[:10]

        return result_html
    except Exception as e:
        return f"<div class='alert alert-danger'>Error during prediction: {str(e)}</div>"


@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("file")
    if not uploaded_file or uploaded_file.filename == '':
        return "<div class='alert alert-warning'>No file selected.</div>"

    if not allowed_file(uploaded_file.filename):
        return "<div class='alert alert-warning'>Unsupported file type.</div>"

    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(filepath)

    try:
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif filename.endswith(".csv"):
            text = extract_text_from_csv(filepath)
        elif filename.endswith(".txt"):
            text = extract_text_from_txt(filepath)
        else:
            return "<div class='alert alert-danger'>Unsupported file format.</div>"

        vec_input = vectorizer.transform([text])
        result = model.predict(vec_input)[0]
        label = "FAKE" if result == 1 else "REAL"

        history = session.get('history', [])
        history.insert(0, {"text": f"File: {filename}", "result": label})
        session['history'] = history[:10]

        return f"<div class='alert alert-success'>File <strong>{filename}</strong> analyzed: <strong>{label}</strong></div>"
    except Exception as e:
        return f"<div class='alert alert-danger'>Error reading file: {str(e)}</div>"

if __name__ == "__main__":
    app.run(debug=True)