"""
CampusVoice Complaint Analyzer — Training Script
Strategy: Train ONLY on English text so the vocabulary is consistent and
          generalizes well. Non-English complaints are translated at inference
          time by app.py before being classified.
"""
import pandas as pd
import re
import joblib
import os
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Config ────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, 'complaints_augmented.csv')

# ── Label Normalization Maps ───────────────────────────────────────────────────
CATEGORY_MAP = {
    'Academic':     'Academics',
    'Billing':      'Finance',
    'Office':       'Administration',
    'HR':           'Administration',
    'Events':       'Extracurricular',
    'Cloud':        'Software',
    'Database':     'Software',
    'Access':       'Software',
    'Health':       'Facilities',
    'Examination':  'Academics',    # merge into Academics
    'Environment':  None,           # too few samples
    'Mobile':       None,
}

PRIORITY_MAP = {
    'Critical': 'High',
}


def is_english(text: str) -> bool:
    """Return True if text is primarily English (all ASCII characters)."""
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # English-only: keep ASCII
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean(csv_path: str, min_words: int = 4) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep='\t', on_bad_lines='skip')
    df.fillna('', inplace=True)

    # Normalize labels
    df['category'] = df['category'].map(lambda x: CATEGORY_MAP.get(x, x))
    df['priority']  = df['priority'].map(lambda x: PRIORITY_MAP.get(x, x))

    # Drop rows where category was mapped to None
    df = df[df['category'].notna()].reset_index(drop=True)

    # Keep ONLY English complaints for training
    df = df[df['text'].apply(is_english)].reset_index(drop=True)
    print(f"  After English-only filter: {len(df)} rows")

    # Clean text (ASCII-safe)
    df['text'] = df['text'].apply(clean_text)

    # Filter out very short entries
    df = df[df['text'].str.split().str.len() >= min_words].reset_index(drop=True)

    return df


# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading and cleaning dataset (English only)...")
df = load_and_clean(csv_path)
print(f"  Loaded {len(df)} complaints across {df['category'].nunique()} categories")
print(f"  Priority classes: {sorted(df['priority'].unique())}\n")
print(df['category'].value_counts().to_string())
print()

# ── Shared Vectorizer ─────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2),     # unigrams + bigrams
    sublinear_tf=True,      # log normalization
    min_df=2,               # drop single-occurrence noise words
    stop_words='english',
)

X = vectorizer.fit_transform(df['text'])

# ── MODEL 1: Category Classifier ─────────────────────────────────────────────
print("Training Category classifier (LinearSVC)...")
svc_cat = LinearSVC(max_iter=2000, class_weight='balanced', C=1.0)
category_clf = CalibratedClassifierCV(svc_cat, cv=3)
category_clf.fit(X, df['category'])

# ── MODEL 2: Priority Classifier ──────────────────────────────────────────────
print("Training Priority classifier (LinearSVC)...")
svc_pri = LinearSVC(max_iter=2000, class_weight='balanced', C=1.0)
priority_clf = CalibratedClassifierCV(svc_pri, cv=3)
priority_clf.fit(X, df['priority'])

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving models to disk...")
joblib.dump(vectorizer,   os.path.join(script_dir, 'vectorizer.pkl'))
joblib.dump(category_clf, os.path.join(script_dir, 'category_model.pkl'))
joblib.dump(priority_clf, os.path.join(script_dir, 'priority_model.pkl'))

print("Training complete! Models saved:")
print(f"  vectorizer.pkl     ({os.path.getsize(os.path.join(script_dir,'vectorizer.pkl'))//1024} KB)")
print(f"  category_model.pkl ({os.path.getsize(os.path.join(script_dir,'category_model.pkl'))//1024} KB)")
print(f"  priority_model.pkl ({os.path.getsize(os.path.join(script_dir,'priority_model.pkl'))//1024} KB)")
