import os
import torch
import joblib
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer

os.environ['HF_HOME'] = 'huggingface_cache'

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
    if lang not in translation_models:
        return None, None
    
    if translation_models[lang]['model'] is None:
        print(f"Loading {lang} translation model on demand to E: drive...")
        model_name = translation_models[lang]['name']
        translation_models[lang]['tokenizer'] = MarianTokenizer.from_pretrained(model_name)
        translation_models[lang]['model'] = MarianMTModel.from_pretrained(model_name).to('cpu')
        gc.collect() 
    
    return translation_models[lang]['tokenizer'], translation_models[lang]['model']

def detect_language(text):
    for char in text:
        if '\u0980' <= char <= '\u09FF':
            return "bn"
        if '\u0900' <= char <= '\u097F':
            return "hi"
    return "en"

def translate_to_english(text):
    try:
        lang = detect_language(text)
        if lang == "en":
            return text

        tokenizer, model = get_translation_model(lang)
        if not tokenizer or not model:
            return text

        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            output = model.generate(**inputs)
            translated = tokenizer.decode(output[0], skip_special_tokens=True)

        return translated if translated and translated.strip() != "" else text

    except Exception as e:
        print("Translation Error:", e)
        return text

def get_authority(category):
    mapping = {
        'Infrastructure': 'Estate Office / Maintenance Dept',
        'Hostel': 'Chief Warden',
        'Academic': 'Dean of Academics',
        'Canteen': 'Canteen Committee',
        'Office / Admin': 'General Administration'
    }
    return mapping.get(category, 'General Administration')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    translated = translate_to_english(text)

    text_vector = vectorizer.transform([translated])
    
    category = category_model.predict(text_vector)[0]
    priority = priority_model.predict(text_vector)[0]
    authority = get_authority(category)

    complaint = {
        "original": text,
        "translated": translated,
        "translated_text": translated,
        "category": str(category),
        "priority": str(priority),
        "authority": authority,
        "reason": f"Classified as {category} based on content analysis."
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