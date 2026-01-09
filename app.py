# ============================================================================
# APP.PY - SISTEM EKSTRAKSI KELUHAN (FINAL UPDATE)
# Fitur: Hybrid CBD + Sentiment + User Defined Algorithm (Enhanced Rules)
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
# 1. KONFIGURASI & LOADER
# ============================================================================

# --- Path Model & Data (Sesuaikan dengan struktur folder kamu) ---
SENTIMENT_MODEL_PATH = "./model/saved_model"
LABEL_ENCODER_PATH = "./model/label_encoder.pkl"
MODEL_NAME = "indobenchmark/indobert-base-p1"
POS_MODEL_NAME = "w11wo/indonesian-roberta-base-posp-tagger"
CBD_BERT_PATH = "./model/indobert_clause_detection_model.pt"
CBD_CRF_PATH = "./model/crf_clause_detection_model.pkl"

# --- CONFIG FILE TEXT ---
NEGATIVE_KEYWORDS_FILE = "./model/negative.txt"
POSITIVE_KEYWORDS_FILE = "./model/positive.txt"
STOP_KELUHAN_FILE = "./model/badword.txt"

MAX_LEN_CBD = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ============================================================================
# 2. KAMUS & LIST KATA (UPDATED - STRICT VERSION)
# ============================================================================

# --- KAMUS NORMALISASI (SLANG -> BAKU) ---
key_norm = {
    "yg": "yang", "gak": "tidak", "ga": "tidak", "g": "tidak", "nggak": "tidak",
    "kalo": "kalau", "klo": "kalau", "kl": "kalau",
    "bgt": "banget", "bg": "banget", "dgn": "dengan", "dg": "dengan",
    "krn": "karena", "karna": "karena", "tdk": "tidak", "tak": "tidak",
    "jd": "jadi", "jdi": "jadi", "bkn": "bukan", "sdh": "sudah",
    "tp": "tapi", "tpi": "tapi", "sy": "saya", "aku": "saya",
    "bgs": "bagus", "good": "bagus", "bad": "jelek",
    "d": "di", "tmpt": "tempat", "msh": "masih", "tau": "tahu"
}

# --- IGNORE WORDS (SANGAT PENTING: MENCEGAH NOISE WAKTU/SARAN) ---
IGNORE_WORDS = {
    # Kata Sambung / Tugas
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "ada", "adalah", 
    "bagi", "untuk", "cuma", "hanya", "sekedar", "agak", "sangat", 
    "tapi", "tetapi", "namun", "walaupun", "meskipun", "karena", "karna", "gara",
    "saya", "kita", "kami", "anda", "mereka", "dia", "kalian", "kamu",
    "sayangnya", "menuju", "depan", "belakang", "atas", "bawah", "samping",
    "lainnya", "katanya", "mulai", "pas", "mana", "supaya", "sebagai",
    "lah", "dong", "sih", "deh", "kok", "mah", "tuh", "pun", "kah",
    "sendiri", "gini", "gitu", "segini", "begitu", "seperti", "kayak", "serta",
    "kritik", "saran", "masukan", "halo", "kak", "min", "admin", "wkwk",
    "tolong", "mohon", "harap", "coba", "bisa", "mau", "ingin", "pingin",
    "biar", "agar", "kalau", "kalo", "jika", "jga", "juga", "memang", "emang",
    "masalah", "hal", "kondisi", "keadaan", "situasi", "soal", "terkait",
    
    # KATA WAKTU (PENYEBAB UTAMA ERROR DI SCREENSHOT SEBELUMNYA)
    "sekarang", "tadi", "nanti", "kemarin", "besok", "lusa",
    "hari", "minggu", "bulan", "tahun", "jam", "menit", "detik",
    "saat", "waktu", "pas", "ketika", "sejak", "baru", "lama", "dahulu", "dulu",
    
    # KATA SARAN (PENYEBAB ERROR "PERLU PEMBENAHAN")
    "perlu", "harus", "wajib", "mesti", "sebaiknya", "seharusnya", "akan"
}

# --- LIST NEGASI SLANG ---
SLANG_NEGATORS = {"ga", "gak", "tak", "enggak", "ora", "kagak", "nda", "ndak", "tdk"}

# --- MANUAL POSITIVE (Untuk menangkap 'kurang baik') ---
MANUAL_POSITIVE = {
    "baik", "bagus", "indah", "cantik", "nyaman", "aman", "bersih", "ramah", 
    "luas", "terawat", "rapi", "sejuk", "dingin", "puas", "enak", "sedap",
    "mantap", "keren", "oke", "layak", "memadai", "profesional", "sigap"
}

# --- FUNGSI LOADER TEXT FILES ---
def load_keywords_newline(filepath):
    keywords = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word: keywords.add(word)
        except: pass
    return keywords

def load_keywords_comma(filepath):
    keywords = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                items = content.split(',')
                for item in items:
                    word = item.strip().lower()
                    if word: keywords.add(word)
        except: pass
    else:
        return {"membuat", "datang", "memberi", "memberikan", "mengambil", "menjadi", "bikin"}
    return keywords

# Load data eksternal
NEGATIVE_KEYWORDS = load_keywords_newline(NEGATIVE_KEYWORDS_FILE)
POSITIVE_KEYWORDS = load_keywords_newline(POSITIVE_KEYWORDS_FILE)
STOP_KELUHAN = load_keywords_comma(STOP_KELUHAN_FILE)

# ============================================================================
# 3. FUNGSI PREPROCESSING
# ============================================================================
def clean_text_user(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalisasi Slang
    text = ' '.join([key_norm.get(word, word) for word in text.split()])
    return text

# ============================================================================
# 4. LOAD MODEL (INIT)
# ============================================================================
print("--- Initializing Models ---")

# A. Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except: tokenizer = None

# B. Model Sentimen
try:
    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        num_sent_labels = len(label_encoder.classes_)
    else:
        num_sent_labels = 3 # Dummy fallback
    
    sentiment_model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, num_labels=num_sent_labels)
    sentiment_model.to(device)
    sentiment_model.eval()
    print("✓ Model Sentimen Loaded")
except Exception as e:
    sentiment_model = None
    print(f"✗ Gagal load Sentimen: {e}")

# C. POS Tagger
try:
    pos_pipeline = pipeline(
        "token-classification", 
        model=POS_MODEL_NAME, 
        tokenizer=POS_MODEL_NAME, 
        aggregation_strategy="simple", 
        device=0 if torch.cuda.is_available() else -1
    )
    print("✓ POS Tagger Loaded")
except Exception as e:
    pos_pipeline = None
    print(f"✗ Gagal load POS: {e}")

# D. Model CBD (Clause Boundary)
class IndoBERT_FineTune(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def get_features(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)

cbd_bert_model = None
cbd_crf_model = None

try:
    if os.path.exists(CBD_BERT_PATH) and os.path.exists(CBD_CRF_PATH):
        import sklearn_crfsuite # Pastikan library ini terinstall
        cbd_bert_model = IndoBERT_FineTune(MODEL_NAME, num_labels=3)
        cbd_bert_model.load_state_dict(torch.load(CBD_BERT_PATH, map_location=device))
        cbd_bert_model.to(device)
        cbd_bert_model.eval()
        with open(CBD_CRF_PATH, 'rb') as f:
            cbd_crf_model = pickle.load(f)
        print("✓ Model CBD Loaded")
    else:
        print("⚠ Model CBD tidak ditemukan, menggunakan fallback regex.")
except Exception as e:
    print(f"✗ Error Loading CBD: {e}")

# ============================================================================
# 5. CORE HELPER FUNCTIONS
# ============================================================================

# --- CBD Logic ---
def extract_bert_features_cbd(tokens):
    encoded = tokenizer(
        tokens, is_split_into_words=True, return_tensors="pt",
        truncation=True, padding='max_length', max_length=MAX_LEN_CBD
    ).to(device)
    with torch.no_grad():
        features = cbd_bert_model.get_features(encoded['input_ids'], encoded['attention_mask'])
        features = features[0][:len(tokens)].cpu().numpy()
    return features

def detect_clauses(text):
    clean_text = re.sub(r'\s+', ' ', text).strip()
    if not clean_text: return []
    
    # Fallback jika model CBD tidak ada
    if not cbd_bert_model or not cbd_crf_model:
        return re.split(r'[.!?,;]\s*', clean_text)

    # Pra-pemisahan kasar (Split by punctuation marks first)
    # Ini membantu CBD bekerja lebih ringan per potongan kalimat
    rough_splits = re.split(r'([.!?;])', clean_text)
    
    final_clauses = []
    
    for segment in rough_splits:
        segment = segment.strip()
        if not segment or segment in ['.', '!', '?', ';']: continue
        
        # Tokenize
        # Kita gunakan regex tokenization sederhana agar sesuai dengan cara kerja CRF
        raw_tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", segment)
        if not raw_tokens: continue

        try:
            features = extract_bert_features_cbd(raw_tokens)
            pred_tags = cbd_crf_model.predict_single(features)
            
            current_clause = []
            for tok, tag in zip(raw_tokens, pred_tags):
                # Logika BIO Splitting
                if tag == "B-CLAUSE":
                    if current_clause: 
                        final_clauses.append(" ".join(current_clause).replace(" ,", ",").replace(" .", "."))
                    current_clause = [tok]
                else:
                    current_clause.append(tok)
            
            if current_clause:
                final_clauses.append(" ".join(current_clause).replace(" ,", ",").replace(" .", "."))
                
        except Exception:
            # Jika error saat prediksi, masukkan segment mentah
            final_clauses.append(segment)

    return [c.strip() for c in final_clauses if len(c.strip()) > 3]

# --- Sentiment Logic ---
def predict_sentiment_global(text):
    if not sentiment_model: return "netral"
    norm_text = clean_text_user(text)
    inputs = tokenizer(norm_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred_id])[0]

# --- POS & Extraction Helpers ---
def fix_broken_subwords(pos_results):
    if not pos_results: return []
    merged_tokens = []
    current_word = pos_results[0]['word']
    current_label = pos_results[0]['entity_group']
    current_end = pos_results[0]['end']

    for i in range(1, len(pos_results)):
        token = pos_results[i]
        start = token['start']
        if start == current_end: # Subword detect
            current_word += token['word'].replace('##', '')
            current_end = token['end']
        else:
            merged_tokens.append({'word': current_word, 'entity_group': current_label})
            current_word = token['word']
            current_label = token['entity_group']
            current_end = token['end']
    merged_tokens.append({'word': current_word, 'entity_group': current_label})
    return merged_tokens

def map_model_label_to_user_label(label):
    label = label.upper()
    if label in ["NOUN", "PROPN"]: return "NNO"
    if label == "ADJ": return "ADJ"
    if label == "VERB": return "VBI"
    return label

def merge_noun_phrases(tokens):
    merged = []
    i = 0
    while i < len(tokens):
        word, label = tokens[i]
        # Gabung jika NNO ketemu NNO (Rumah + Sakit -> Rumah Sakit)
        if label == "NNO" and i + 1 < len(tokens) and tokens[i+1][1] == "NNO":
            new_word = f"{word} {tokens[i+1][0]}"
            merged.append((new_word, "NNO"))
            i += 2 
        else:
            merged.append((word, label))
            i += 1
    return merged

# ============================================================================
# 6. MAIN EXTRACTION ALGORITHM (UPDATED LOGIC)
# ============================================================================
def extract_complaints_user_algo(raw_text):
    if not pos_pipeline: return []
    
    # 1. Normalisasi
    norm_text = clean_text_user(raw_text)

    # 2. POS Tagging
    try:
        hasil_pos_raw = pos_pipeline(norm_text)
    except: return []

    # 3. Fix Subwords & Label Override
    hasil_pos = fix_broken_subwords(hasil_pos_raw)
    tokens = []
    
    for t in hasil_pos:
        word = t['word'].strip().lower()
        word = re.sub(r'[^a-z0-9]', '', word) # Bersih simbol
        if not word: continue

        # Normalisasi Negasi Slang
        if word in SLANG_NEGATORS: word = "tidak"

        user_label = map_model_label_to_user_label(t['entity_group'])
        
        # === LOGIKA OVERRIDE PENTING ===
        # A. Kata Negatif di-force jadi ADJ (misal: "kotor" terdeteksi NNO -> ubah ADJ)
        if word in NEGATIVE_KEYWORDS: user_label = "ADJ"
        # B. Kata Positif khusus (untuk kasus "tidak ramah")
        if word in POSITIVE_KEYWORDS or word in MANUAL_POSITIVE: user_label = "POS_ADJ"
        # C. Negasi
        if word == "tidak": user_label = "NEG"
        # D. Kurang = Tidak
        if word == "kurang": user_label = "NEG"
        
        tokens.append((word, user_label))

    # 4. Gabung Noun Phrases (setelah label negatif diamankan jadi ADJ)
    tokens = merge_noun_phrases(tokens)

    keluhan_list = []
    
    # 5. Core Search Logic
    for i, (word, label) in enumerate(tokens):
        
        # HANYA PROSES KATA BENDA (NNO)
        if label == "NNO":
            original_nno = word
            
            # FILTER: Jangan proses kata NNO yang ada di IGNORE LIST (waktu/saran)
            if word in IGNORE_WORDS: continue

            closest_keluhan = None
            
            # === STRATEGI A: CARI KE KANAN (Noun -> Adj/Neg) ===
            # Contoh: "Toilet(NNO) kotor(ADJ)" atau "Pelayanan(NNO) kurang(NEG) ramah(POS_ADJ)"
            for j in range(i + 1, min(i + 6, len(tokens))):
                w2, l2 = tokens[j]
                if w2 in IGNORE_WORDS: continue 
                
                prev_w = tokens[j-1][0] if j > 0 else ""
                has_negator = prev_w in ["tidak", "kurang", "belum", "bukan", "jangan"]
                
                # Pola 1: Noun + Negasi + Kata Sifat (Positif/Negatif/Apapun)
                if has_negator and w2 not in IGNORE_WORDS:
                    closest_keluhan = f"{original_nno} {prev_w} {w2}"
                    break # Ketemu, stop

                # Pola 2: Noun + Kata Negatif Langsung
                is_negative_context = (w2 in NEGATIVE_KEYWORDS) or \
                                      (l2 in ["ADJ", "NEG"] and l2 != "POS_ADJ" and w2 not in MANUAL_POSITIVE)
                
                if is_negative_context:
                    if w2 in STOP_KELUHAN: continue
                    closest_keluhan = f"{original_nno} {w2}"
                    break 

            # === STRATEGI B: CARI KE KIRI (Adj/Neg <- Noun) ===
            # Contoh: "Kurang(NEG) nyaman(POS_ADJ) tempatnya(NNO)"
            if not closest_keluhan: 
                for j in range(i - 1, max(i - 6, -1), -1):
                    w2, l2 = tokens[j]
                    if w2 in IGNORE_WORDS: continue
                    
                    # Cek kata sebelum kata sifat tersebut
                    prev_modifier = tokens[j-1][0] if j > 0 else ""
                    has_negator = prev_modifier in ["tidak", "kurang", "belum", "bukan"]

                    # Pola 1: Negasi + Kata Positif + Noun (Dibalik)
                    # "Kurang nyaman tempatnya"
                    if (w2 in MANUAL_POSITIVE or l2 == "POS_ADJ") and has_negator:
                         closest_keluhan = f"{original_nno} {prev_modifier} {w2}"
                         break

                    # Pola 2: Kata Negatif + Noun (Dibalik)
                    # "Rusak jalannya"
                    is_negative_context = (w2 in NEGATIVE_KEYWORDS) or \
                                          (l2 in ["ADJ", "NEG"] and l2 != "POS_ADJ" and w2 not in MANUAL_POSITIVE)
                    
                    if is_negative_context:
                        if w2 in STOP_KELUHAN: continue
                        if has_negator:
                            closest_keluhan = f"{original_nno} {prev_modifier} {w2}"
                        else:
                            closest_keluhan = f"{original_nno} {w2}"
                        break

            if closest_keluhan:
                keluhan_list.append(closest_keluhan)

    return list(dict.fromkeys(keluhan_list)) # Hapus duplikat

# ============================================================================
# 7. FLASK ROUTES
# ============================================================================
app = Flask(__name__)

@app.route('/')
def home(): return render_template("index.html")

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
        # 1. Pecah Klausa (CBD)
        clauses = detect_clauses(review)
        
        for clause in clauses:
            # 2. Sentimen
            sent_label = predict_sentiment_global(clause)
            
            if sent_label.lower() == 'positif': count_pos += 1
            elif sent_label.lower() == 'negatif': count_neg += 1
            else: count_neu += 1
            
            # 3. Ekstraksi (Hanya pada sentimen Negatif/Netral)
            if sent_label.lower() in ['negatif', 'netral']:
                extracted_list = extract_complaints_user_algo(clause)
                
                if extracted_list:
                    # Ambil kata benda pertamanya untuk WordCloud/Top Noun
                    for item in extracted_list:
                        first_word = item.split()[0]
                        all_nno_words.append(first_word)

                    formatted_result = ", ".join(extracted_list)
                    complaint_data.append({
                        "Ulasan Negatif": clause, 
                        "Keluhan": formatted_result
                    })

    # Output Data
    nno_counter = Counter(all_nno_words)
    top_5_nno = [{'word': noun, 'count': count} for noun, count in nno_counter.most_common(5)]
    
    # Beri nomor urut
    for i, item in enumerate(complaint_data, 1): item['No'] = i

    return jsonify({
        'wisata': wisata_name,
        'total': len(reviews),
        'positif': count_pos,
        'negatif': count_neg,
        'netral': count_neu,
        'results': [],
        'complaints': complaint_data,
        'top_nouns': top_5_nno
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5132, debug=True)