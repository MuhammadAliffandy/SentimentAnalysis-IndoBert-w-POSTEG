# app.py
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import joblib
from flask import Flask, request, jsonify, render_template, abort, send_from_directory
import os
import json
from transformers import pipeline
from collections import Counter
import pandas as pd

# Config / paths (ubah jika perlu)
MODEL_PATH = "./saved_model"
LABEL_ENCODER_PATH = "label_encoder.pkl"
PRETRAINED_TOKENIZER = "indobenchmark/indobert-base-p1"

# --- Load resources (fallbacks raise clear errors) ---
try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    num_labels = len(label_encoder.classes_)
    print(f"[INFO] Label encoder loaded with {num_labels} classes.")
except FileNotFoundError:
    raise FileNotFoundError(f"{LABEL_ENCODER_PATH} not found. Kalau mau test cepat, gunakan test_app.py yang mem-mock model.")
except Exception as e:
    raise RuntimeError(f"Error loading label encoder: {e}")

try:
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_TOKENIZER)
    print("[INFO] Tokenizer loaded.")
except Exception as e:
    raise RuntimeError(f"Error loading tokenizer: {e}")

try:
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels)
    model.eval()
    print("[INFO] Model loaded.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"[INFO] Using device: {device}")

# ====== POS Tagger ======
pos_model_name = "w11wo/indonesian-roberta-base-posp-tagger"
pos_pipeline = pipeline(
    "token-classification",
    model=pos_model_name,
    tokenizer=pos_model_name,
    aggregation_strategy="simple"
)

# kata negatif tambahan (badword)
badword_file = "badword.txt"
try:
    with open(badword_file, "r", encoding="utf-8") as f:
        badword_text = f.read()
    negative_keywords = list({w.strip().lower() for w in badword_text.split(",") if w.strip()})
except Exception:
    negative_keywords = []

stop_keluhan = {"membuat", "datang", "memberi", "memberikan", "mengambil", "menjadi"}

def find_original_word(text, word):
    for w in text.split():
        if word in w.lower():
            return w
    return word

def merge_noun_phrases(tokens):
    merged = []
    skip = False
    for i in range(len(tokens)):
        if skip:
            skip = False
            continue
        word, label = tokens[i]
        if label == "NNO" and i+1 < len(tokens) and tokens[i+1][1] == "NNO":
            merged.append((word + " " + tokens[i+1][0], "NNO"))
            skip = True
        else:
            merged.append((word, label))
    return merged

# --- Flask app ---
app = Flask(__name__)

@app.route('/')
def home():
    return "Sentiment Analysis API. Use /predict endpoint or open /ui for a simple web UI."

# Simple UI route (renders templates/index.html)
@app.route('/ui')
def ui():
    # requires templates/index.html to exist
    try:
        return render_template("index.html")
    except Exception:
        abort(500, description="UI template missing. Put templates/index.html in project folder.")

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid request. Provide JSON with "text" field.'}), 400

    new_review = data['text']
    # Tokenize
    inputs = tokenizer(new_review, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
    predicted_label = label_encoder.inverse_transform([predicted_id])[0]

    return jsonify({'sentiment': predicted_label})

@app.route('/wisata_list', methods=['GET'])
def wisata_list():
    # Ambil semua file .json di folder data
    folder = os.path.join(os.path.dirname(__file__), 'data')
    wisata_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    # Nama wisata dari nama file
    wisata_names = [f.replace('.json','').replace('_',' ').title() for f in wisata_files]
    return jsonify({'wisata': wisata_names})

@app.route('/analisis_wisata', methods=['POST'])
def analisis_wisata():
    req = request.get_json(silent=True)
    if not req or 'wisata' not in req:
        return jsonify({'error': 'Missing wisata name'}), 400
    wisata_name = req['wisata']
    folder = os.path.join(os.path.dirname(__file__), 'data')
    filename = wisata_name.lower().replace(' ','_') + '.json'
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    reviews = data.get('reviews', [])
    results = []
    negative_reviews = []
    count_pos, count_neg, count_neu = 0, 0, 0
    for review in reviews:
        inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_id = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        predicted_label = label_encoder.inverse_transform([predicted_id])[0]
        if predicted_label.lower() == 'positif':
            count_pos += 1
        elif predicted_label.lower() == 'negatif':
            count_neg += 1
            negative_reviews.append(review)
        else:
            count_neu += 1
        results.append({'review': review, 'sentiment': predicted_label})
    # ====== Proses Ekstraksi Keluhan ======
    all_nno_words = []
    final_data = []
    for idx, kalimat in enumerate(negative_reviews, start=1):
        hasil_pos = pos_pipeline(kalimat)
        tokens = [(t['word'].strip().lower(), t['entity_group']) for t in hasil_pos]
        tokens = merge_noun_phrases(tokens)
        keluhan_pairs = []
        for i, (word, label) in enumerate(tokens):
            if label == "NNO":
                original_nno = find_original_word(kalimat, word)
                all_nno_words.append(original_nno)
                closest_keluhan = None
                min_distance = 999
                for j, (w2, l2) in enumerate(tokens):
                    if j != i:
                        if l2 in ["ADJ", "NEG", "VBI", "VBT", "VBP"] or w2 in negative_keywords or w2 == "tidak":
                            if w2 in stop_keluhan:
                                continue
                            distance = abs(j - i)
                            if distance <= 4 and distance < min_distance:
                                min_distance = distance
                                if w2 == "tidak":
                                    if j+1 < len(tokens):
                                        next_word = tokens[j+1][0]
                                        closest_keluhan = f"{original_nno} tidak {next_word}"
                                    else:
                                        closest_keluhan = f"{original_nno} tidak jelas"
                                elif j-1 >= 0 and tokens[j-1][0] == "tidak":
                                    closest_keluhan = f"{original_nno} tidak {w2}"
                                else:
                                    closest_keluhan = f"{original_nno} {w2}"
                if not closest_keluhan:
                    closest_keluhan = f"{original_nno} bermasalah"
                keluhan_pairs.append(closest_keluhan)
        keluhan_pairs = list(dict.fromkeys(keluhan_pairs))
        if not keluhan_pairs:
            keluhan_pairs = ["keluhan tidak terdeteksi"]
        final_data.append({
            "No": idx,
            "Ulasan Negatif": kalimat,
            "Keluhan": ", ".join(keluhan_pairs)
        })
    nno_counter = Counter(all_nno_words)
    top_5_nno = nno_counter.most_common(5)
    return jsonify({
        'wisata': wisata_name,
        'total': len(reviews),
        'positif': count_pos,
        'negatif': count_neg,
        'netral': count_neu,
        'results': results,
        'complaints': final_data,
        'top_nouns': [{'word': noun, 'count': count} for noun, count in top_5_nno]
    })

if __name__ == '__main__':
    # Jalankan Flask biasa
    app.run(host="0.0.0.0", port=5000, debug=True)
