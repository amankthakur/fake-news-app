import joblib

# Load your vectorizer
vec = joblib.load("model/vectorizer.pkl")

# Get the number of features
try:
    n_feats = len(vec.get_feature_names_out())
except:
    n_feats = len(vec.vocabulary_)

print(f"Vectorizer has {n_feats} features.")

# Print the first 20 feature names (alphabetical)
names = sorted(vec.get_feature_names_out() if hasattr(vec, "get_feature_names_out") 
               else vec.vocabulary_.keys())
print("First 20 features:", names[:20])
