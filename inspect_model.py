import joblib

# Load your model
model = joblib.load("model/fake_news_model.pkl")

# Print basic parameters & class distribution
print("Model parameters:", model.get_params())
print("Classes:", model.classes_)
