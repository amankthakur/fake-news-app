import joblib

# Load artifacts
model      = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# Example tests
tests = [
    # Fake
    "Breaking: Scientists have discovered that eating chocolate every day can completely cure cancer within two weeks, according to an anonymous study leaked from a top-secret lab.",
    # Real
    "The World Health Organization today released a report showing a 10% global decrease in malaria cases over the past five years, crediting expanded mosquito net distribution programs in Africa.",
    # Another Real
    "Yesterday, NASA launched the James Webb Space Telescope successfully into orbit to study the early universe.",
]

for t in tests:
    label = "FAKE" if predict(t) == 1 else "REAL"
    print(f"Input: {t[:60]}…\n  → Prediction: {label}\n")
