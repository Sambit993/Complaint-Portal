import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# SHIFT STORAGE TO E: DRIVE
os.environ['HF_HOME'] = 'huggingface_cache'
os.makedirs('huggingface_cache', exist_ok=True)

print("Loading dataset...")
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "complaints.csv")

try:
    # Handle different separators in CSV
    df = pd.read_csv(csv_path, sep='\t')
    if 'text' not in df.columns:
        df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

print(f"Loaded {len(df)} complaints.")

df.fillna('', inplace=True)

print("Training TF-IDF Vectorizer (Low Memory)...")
# Using a lighter approach than BERT to stay within RAM limits
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])

print("Training Category classifier...")
category_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
category_clf.fit(X, df['category'])

print("Training Priority classifier...")
priority_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
priority_clf.fit(X, df['priority'])

print("Saving models to disk...")
joblib.dump(vectorizer, os.path.join(script_dir, 'vectorizer.pkl'))
joblib.dump(category_clf, os.path.join(script_dir, 'category_model.pkl'))
joblib.dump(priority_clf, os.path.join(script_dir, 'priority_model.pkl'))

print("Training complete! Lightweight models are ready.")
