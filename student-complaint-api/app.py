import os
import re
import torch
import joblib
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer

# NOTE: Do NOT override HF_HOME — models are cached in the default system location
# (~/.cache/huggingface/hub). Overriding it to a local empty folder breaks loading.

app = Flask(__name__)
CORS(app)

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading classification models (Low Memory)...")

vectorizer = joblib.load(os.path.join(script_dir, 'vectorizer.pkl'))
category_model = joblib.load(os.path.join(script_dir, 'category_model.pkl'))
priority_model = joblib.load(os.path.join(script_dir, 'priority_model.pkl'))

translation_models = {
    'bn': {'model': None, 'tokenizer': None, 'name': "Helsinki-NLP/opus-mt-bn-en"},
    'hi': {'model': None, 'tokenizer': None, 'name': "Helsinki-NLP/opus-mt-hi-en"}
}

complaints_db = []

def get_translation_model(lang):
    """Load Marian translation model from default HF system cache (local_files_only)."""
    if lang not in translation_models:
        return None, None

    if translation_models[lang]['model'] is None:
        model_name = translation_models[lang]['name']
        print(f"Loading {lang} translation model ({model_name}) from system cache...")
        try:
            translation_models[lang]['tokenizer'] = MarianTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            translation_models[lang]['model'] = MarianMTModel.from_pretrained(
                model_name, local_files_only=True
            ).to('cpu')
            gc.collect()
            print(f"{lang} model loaded successfully.")
        except Exception as e:
            print(f"Could not load {lang} Marian model from cache: {e}")
            translation_models[lang]['tokenizer'] = 'FAILED'
            translation_models[lang]['model'] = 'FAILED'

    tok = translation_models[lang]['tokenizer']
    mod = translation_models[lang]['model']
    if tok == 'FAILED' or mod == 'FAILED':
        return None, None
    return tok, mod


def detect_language(text):
    """Detect Bengali or Hindi by Unicode range; else assume English."""
    for char in text:
        if '\u0980' <= char <= '\u09FF':
            return "bn"
        if '\u0900' <= char <= '\u097F':
            return "hi"
    return "en"


def _fallback_translate(text, lang):
    """Fallback translation using Google Translate (no extra packages) when Marian is unavailable."""
    # Option 1: deep_translator (if installed)
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source='auto', target='en').translate(text)
        print(f"[fallback/deep_translator] {lang} -> EN: {result}")
        return result if result else text
    except ImportError:
        pass
    except Exception as e:
        print(f"deep_translator error: {e}")

    # Option 2: urllib + free Google Translate endpoint (no API key required)
    try:
        import urllib.request, urllib.parse, json as _json
        url = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl=auto&tl=en&dt=t&q={urllib.parse.quote(text)}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read())
        result = "".join(part[0] for part in data[0] if part[0])
        print(f"[fallback/urllib] {lang} -> EN: {result}")
        return result if result else text
    except Exception as fe:
        print(f"All translation fallbacks failed: {fe}")
        return text


def translate_to_english(text):
    try:
        lang = detect_language(text)
        if lang == "en":
            return text

        tokenizer, model = get_translation_model(lang)
        if tokenizer and model:
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                output = model.generate(**inputs)
                translated = tokenizer.decode(output[0], skip_special_tokens=True)
            if translated and translated.strip():
                return translated

        # Marian model not available — use Google Translate fallback
        print(f"Marian model for '{lang}' unavailable, using fallback translator.")
        return _fallback_translate(text, lang)

    except Exception as e:
        print("Translation Error:", e)
        return text

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation — English only (matches training)."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # ASCII only, matches train.py
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_authority(category):
    mapping = {
        'Academics':        'Dean of Academics',
        'Infrastructure':   'Estate Office / Maintenance Dept',
        'Hostel':           'Chief Warden',
        'Administration':   'General Administration',
        'Finance':          'Finance Office',
        'Software':         'IT Department',
        'Hardware':         'IT Department',
        'Network':          'IT Department',
        'Extracurricular':  'Student Activities Committee',
        'Placement':        'Placement Cell',
        'Library':          'Library Committee',
        'Transport':        'Transport Office',
        'Facilities':       'Facilities Management',
        'Examination':      'Examination Cell',
        'Security':         'Security Department',
        'General':          'Student Welfare Office',
    }
    return mapping.get(category, 'General Administration')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']

    # Always translate non-English to English first (model is trained on English only)
    lang = detect_language(text)
    translated = translate_to_english(text) if lang != "en" else text

    # Classify on the translated (English) text — matches training data language
    cleaned = clean_text(translated)

    text_vector = vectorizer.transform([cleaned])

    category = category_model.predict(text_vector)[0]
    priority = priority_model.predict(text_vector)[0]
    authority = get_authority(category)
    
    # Extract top keywords that contributed to this decision
    try:
        feature_names = vectorizer.get_feature_names_out()
        dense_vector = text_vector.todense().tolist()[0]
        word_scores = [(feature_names[i], score) for i, score in enumerate(dense_vector) if score > 0]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        top_words = [word for word, score in word_scores[:3]]
        if top_words:
            reason_text = f"Classified as {category} ({priority} Priority) because it detected keywords like: {', '.join(top_words)}."
        else:
            reason_text = f"Classified as {category} ({priority} Priority) based on content analysis."
    except Exception:
        reason_text = f"Classified as {category} ({priority} Priority) based on content analysis."

    complaint = {
        "original": text,
        "translated": translated,
        "translated_text": translated,
        "category": str(category),
        "priority": str(priority),
        "authority": authority,
        "reason": reason_text
    }

    complaints_db.append(complaint)
    gc.collect()
    return jsonify(complaint)

@app.route('/complaints', methods=['GET'])
def get_complaints():
    return jsonify(complaints_db)

@app.route('/')
def home():
    return "API Running Successfully (Storage on E:, Low-RAM Optimized)!"

if __name__ == '__main__':
    print("API is ready! Starting server...")
    app.run(debug=False, port=5000)