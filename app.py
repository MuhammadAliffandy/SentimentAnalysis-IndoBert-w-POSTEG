# ============================================================================
# APP.PY - SISTEM EKSTRAKSI KELUHAN
# Fitur: Hybrid CBD + Sentiment + User Defined Algorithm (POS + Regex Clean)
# ============================================================================

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel, pipeline
from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import os
import json
import re
from collections import Counter

# ============================================================================
# 1. KONFIGURASI
# ============================================================================

# --- Path Model ---
SENTIMENT_MODEL_PATH = "./saved_model"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MODEL_NAME = "indobenchmark/indobert-base-p1"
POS_MODEL_NAME = "w11wo/indonesian-roberta-base-posp-tagger"
CBD_BERT_PATH = "indobert_finetuned_clause.pt"
CBD_CRF_PATH = "indobert_sklearn_crf_clause_model.pkl"

MAX_LEN_CBD = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- STOP WORDS KELUHAN (Sesuai Request) ---
# Kata-kata ini diabaikan jika muncul sebagai kandidat kata negatif
STOP_KELUHAN = {"membuat", "datang", "memberi", "memberikan", "mengambil", "menjadi", "bikin"}

# --- BADWORDS / NEGATIVE KEYWORDS ---
# (Disarankan hardcode disini agar tidak perlu file eksternal badword.txt saat run server)
NEGATIVE_KEYWORDS = {
    "kotor", "bau", "rusak", "mahal", "lambat", "lama", "lelet", "antri", 
    "ramai", "penuh", "sesak", "panas", "gerah", "berisik", "bising",
    "jelek", "buruk", "parah", "kecewa", "menyesal", "kapok", "kasar",
    "jutek", "sinis", "sombong", "angkuh", "curam", "licin", "gelap",
    "remang", "kumuh", "jorok", "bocor", "mati", "padam", "hilang",
    "dicuri", "copet", "pungli", "susah", "sulit", "ribet", "membingungkan",
    "kurang", "tidak", "gak", "enggak", "jangan", "bukan", "hancur", "berantakan",
    "sampah", "berserakan", "liar", "gatal", "keruh", "berlumut", "amis",
    "terbengkalai", "usang", "rapuh", "karatan", "bolong", "robek", 
    "berbayar", "bayar", "dikit", "sedikit", "kecil", "sempit", "mahalnya",
    "berlumpur", "banjir", "longsor", "retak", "pecah", "mampet", "hilang",
    "tikus", "kecoa", "nyamuk", "lalat", "semut"
}

# ============================================================================
# 2. FUNGSI PREPROCESSING (SESUAI REQUEST)
# ============================================================================
def clean_text_user(text):
    """
    Membersihkan teks sesuai spesifikasi user:
    1. Lowercase
    2. Hapus URL
    3. Hapus tanda baca/simbol (ganti spasi)
    4. Hapus spasi berlebih
    """
    if not text: return ""
    text = text.lower() # Case folding
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Hapus tanda baca/karakter aneh
    text = re.sub(r'\s+', ' ', text).strip() # Normalisasi spasi
    return text

# ============================================================================
# 3. LOAD MODEL
# ============================================================================
print("Sedang memuat model...")

# A. Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    tokenizer = None
    print(f"Error tokenizer: {e}")

# B. Model Sentimen
try:
    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        num_sent_labels = len(label_encoder.classes_)
    else:
        class DummyEnc: classes_=['negatif','netral','positif']; inverse_transform=lambda x: [self.classes_[x[0]]]
        label_encoder = DummyEnc()
        num_sent_labels = 3
    
    sentiment_model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, num_labels=num_sent_labels)
    sentiment_model.to(device)
    sentiment_model.eval()
except Exception as e:
    sentiment_model = None
    print(f"Error sentiment model: {e}")

# C. POS Tagger
try:
    pos_pipeline = pipeline(
        "token-classification", 
        model=POS_MODEL_NAME, 
        tokenizer=POS_MODEL_NAME, 
        aggregation_strategy="simple", 
        device=0 if torch.cuda.is_available() else -1
    )
except:
    pos_pipeline = None

# D. Model CBD (Clause Boundary)
class IndoBERT_FineTune(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def get_features(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        features = self.dropout(outputs.last_hidden_state)
        return features

cbd_bert_model = None
cbd_crf_model = None
try:
    # Cek library sklearn-crfsuite
    try:
        import sklearn_crfsuite
    except ImportError:
        pass

    if os.path.exists(CBD_BERT_PATH) and os.path.exists(CBD_CRF_PATH):
        cbd_bert_model = IndoBERT_FineTune(MODEL_NAME, num_labels=3)
        cbd_bert_model.load_state_dict(torch.load(CBD_BERT_PATH, map_location=device))
        cbd_bert_model.to(device)
        cbd_bert_model.eval()
        with open(CBD_CRF_PATH, 'rb') as f:
            cbd_crf_model = pickle.load(f)
        print("✓ Model Siap.")
    else:
        print("✗ Model CBD files missing.")
except Exception as e:
    print(f"✗ Gagal load CBD: {e}")

# ============================================================================
# 4. LOGIKA UTAMA
# ============================================================================

# --- A. CBD (Clause Boundary Detection) ---
def extract_bert_features_finetuned(tokens):
    encoded = tokenizer(
        tokens, is_split_into_words=True, return_tensors="pt",
        truncation=True, padding='max_length', max_length=MAX_LEN_CBD
    ).to(device)
    with torch.no_grad():
        features = cbd_bert_model.get_features(encoded['input_ids'], encoded['attention_mask'])
        features = features[0][:len(tokens)].cpu().numpy()
    return features

def detect_clauses(text):
    """Memecah paragraf menjadi klausa (sebelum preprocessing regex)"""
    if not cbd_bert_model or not cbd_crf_model:
        return re.split(r'[.,!?;]', text)

    sub_sentences = re.split(r'([.!?])', text.strip())
    all_clauses = []

    for i in range(0, len(sub_sentences), 2):
        sub_text = sub_sentences[i].strip()
        if not sub_text: continue
        tokens = tokenizer.tokenize(sub_text)
        if not tokens: continue

        try:
            features = extract_bert_features_finetuned(tokens)
            pred_tags = cbd_crf_model.predict_single(features)
        except:
            all_clauses.append(sub_text)
            continue

        current_clause = []
        for tok, tag in zip(tokens, pred_tags):
            if tok == "[UNK]": continue # Skip UNK
            
            is_subword = tok.startswith("##")
            clean_tok = tok.replace("##", "")

            if tag == "B-CLAUSE":
                if current_clause: all_clauses.append(" ".join(current_clause))
                current_clause = [clean_tok]
            elif tag == "I-CLAUSE":
                if is_subword and current_clause: current_clause[-1] += clean_tok
                else: current_clause.append(clean_tok)
            elif tag == "O":
                if current_clause:
                    all_clauses.append(" ".join(current_clause))
                    current_clause = []
        
        if current_clause: all_clauses.append(" ".join(current_clause))

    final_clauses = [c.strip() for c in all_clauses if len(c.strip()) > 3]
    return final_clauses if final_clauses else [text]

# --- B. Sentiment ---
def predict_sentiment_global(text):
    if not sentiment_model: return "netral"
    # Preprocess minimal untuk BERT sentimen (jangan hilangkan tanda baca dulu, kadang ngaruh)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred_id])[0]

# --- C. Extraction Helper Functions (User Logic) ---
def find_original_word(text, word):
    """Mencari kata asli dalam teks"""
    for w in text.split():
        if word in w.lower():
            return w
    return word

def merge_noun_phrases(tokens):
    """Gabungkan kata benda (NNO) berurutan"""
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

# --- D. EKSTRAKSI UTAMA (User Algorithm) ---
def extract_complaints_user_algo(raw_text):
    if not pos_pipeline: return []
    
    # 1. PREPROCESSING (Regex User)
    clean_text = clean_text_user(raw_text)
    if not clean_text: return []

    # 2. POS Tagging
    try:
        hasil_pos = pos_pipeline(clean_text)
    except:
        return []

    tokens = [(t['word'].strip().lower(), t['entity_group']) for t in hasil_pos]

    # 3. Gabungkan Noun Phrases
    tokens = merge_noun_phrases(tokens)

    keluhan_list = []
    
    # 4. Logika Pencarian Pasangan (Distance <= 4)
    for i, (word, label) in enumerate(tokens):
        if label == "NNO":
            # Cari bentuk asli kata (sebelum lowercase/clean) untuk display yang bagus
            # Kita cari di clean_text saja karena raw_text mungkin punya tanda baca yg bikin matching susah
            original_nno = find_original_word(clean_text, word) 
            
            closest_keluhan = None
            min_distance = 999

            for j, (w2, l2) in enumerate(tokens):
                if j != i:
                    # Cek apakah token adalah Indikator Negatif
                    is_neg_indicator = (l2 in ["ADJ", "NEG", "VBI", "VBT", "VBP"] or 
                                        w2 in NEGATIVE_KEYWORDS or 
                                        w2 == "tidak")
                    
                    if is_neg_indicator:
                        if w2 in STOP_KELUHAN: continue
                        
                        distance = abs(j - i)
                        if distance <= 4 and distance < min_distance:
                            min_distance = distance
                            
                            # Logika Khusus "Tidak"
                            if w2 == "tidak":
                                if j+1 < len(tokens):
                                    next_word = tokens[j+1][0]
                                    closest_keluhan = f"{original_nno} tidak {next_word}"
                                else:
                                    closest_keluhan = f"{original_nno} tidak jelas"
                            
                            # Logika "Tidak" di belakang kata sifat (misal: "bersih tidak") -> jarang, 
                            # tapi logika di script user: j-1 >= 0 and tokens[j-1][0] == "tidak"
                            elif j-1 >= 0 and tokens[j-1][0] == "tidak":
                                closest_keluhan = f"{original_nno} tidak {w2}"
                            
                            else:
                                closest_keluhan = f"{original_nno} {w2}"

            # Fallback
            if not closest_keluhan:
                closest_keluhan = f"{original_nno} bermasalah"
            
            keluhan_list.append(closest_keluhan)

    return list(dict.fromkeys(keluhan_list)) # Hapus duplikat

# ============================================================================
# 5. FLASK ROUTES
# ============================================================================
app = Flask(__name__)

@app.route('/')
def home(): return "API Ekstraksi Keluhan (New Algorithm + Regex Clean)."
@app.route('/ui')
def ui(): return render_template("new.html")

@app.route('/wisata_list', methods=['GET'])
def wisata_list():
    folder = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(folder): return jsonify({'wisata': []})
    wisata_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    names = [f.replace('.json','').replace('_',' ').title() for f in wisata_files]
    return jsonify({'wisata': names})

@app.route('/analisis_wisata', methods=['POST'])
def analisis_wisata():
    req = request.get_json(silent=True)
    if not req or 'wisata' not in req: return jsonify({'error': 'Missing wisata name'}), 400
    
    wisata_name = req['wisata']
    filename = wisata_name.lower().replace(' ','_') + '.json'
    filepath = os.path.join(os.path.dirname(__file__), 'data', filename)
    
    if not os.path.exists(filepath): return jsonify({'error': 'File not found'}), 404
    
    with open(filepath) as f: data = json.load(f)
    reviews = data.get('reviews', [])
    
    complaint_data = []
    count_pos, count_neg, count_neu = 0, 0, 0
    all_nno_words = []
    
    for review in reviews:
        # 1. Pecah Klausa (Menggunakan Teks Asli agar tanda baca terbaca)
        clauses = detect_clauses(review)
        
        for clause in clauses:
            # 2. Cek Sentimen
            clause_sent = predict_sentiment_global(clause)
            
            if clause_sent.lower() == 'positif': count_pos += 1
            elif clause_sent.lower() == 'negatif': count_neg += 1
            else: count_neu += 1
            
            # 3. Ekstraksi (Hanya Negatif/Netral)
            if clause_sent.lower() in ['negatif', 'netral']:
                # Ekstraksi menggunakan algoritma baru + preprocess regex
                extracted_list = extract_complaints_user_algo(clause)
                
                if extracted_list:
                    for item in extracted_list:
                        first_word = item.split()[0]
                        all_nno_words.append(first_word)

                    formatted = ", ".join(extracted_list)
                    complaint_data.append({
                        "Ulasan Negatif": clause, 
                        "Keluhan": formatted
                    })

    # Output Statistics
    nno_counter = Counter(all_nno_words)
    top_5_nno = nno_counter.most_common(5)

    for i, item in enumerate(complaint_data, 1): item['No'] = i

    return jsonify({
        'wisata': wisata_name,
        'total': len(reviews),
        'positif': count_pos,
        'negatif': count_neg,
        'netral': count_neu,
        'results': [],
        'complaints': complaint_data,
        'top_nouns': [{'word': noun, 'count': count} for noun, count in top_5_nno]
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5132, debug=True)