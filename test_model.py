import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join("model", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

# ── Load Artifacts ─────────────────────────────────────────────────────────────
model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ── Debug: Confirm vectorizer size ─────────────────────────────────────────────
# For scikit-learn >=1.0 use get_feature_names_out()
try:
    n_feats = len(vectorizer.get_feature_names_out())
except:
    # older versions
    n_feats = len(vectorizer.vocabulary_)
print(f"🔍 Vectorizer loaded with {n_feats} features (expect 5000)")

# ── Test Samples & True Labels ────────────────────────────────────────────────
tests = [
    # (text, true_label)
    ("Breaking: Scientists have discovered that eating chocolate every day can completely cure cancer within two weeks, according to an anonymous study leaked from a top-secret lab.", 1),
    ("The World Health Organization today released a report showing a 10% global decrease in malaria cases over the past five years, crediting expanded mosquito net distribution programs in Africa.", 0),
    ("Yesterday, NASA launched the James Webb Space Telescope successfully into orbit to study the early universe.", 0),
]

y_true = []
y_pred = []

# ── Run Through Tests ─────────────────────────────────────────────────────────
for text, true_label in tests:
    # transform
    vec = vectorizer.transform([text])
    print(f"\n🧪 Input length: {len(text)} chars → vector shape: {vec.shape}")
    
    # predict
    pred = model.predict(vec)[0]
    label = "FAKE" if pred == 1 else "REAL"
    print(f"  → Model prediction: {label} (expected {'FAKE' if true_label==1 else 'REAL'})")
    
    y_true.append(true_label)
    y_pred.append(pred)

# ── Metrics ───────────────────────────────────────────────────────────────────
print("\n📈 Overall Test Metrics:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Detailed Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["REAL","FAKE"]))
