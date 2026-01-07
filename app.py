# ============================================================================
# APP.PY - SISTEM EKSTRAKSI KELUHAN
# Fitur: Hybrid CBD + Sentiment + User Defined Algorithm (POS + Regex Clean + Normalization)
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

# --- Path Model & Data ---
SENTIMENT_MODEL_PATH = "./model/saved_model"
LABEL_ENCODER_PATH = "./model/label_encoder.pkl"
MODEL_NAME = "indobenchmark/indobert-base-p1"
POS_MODEL_NAME = "w11wo/indonesian-roberta-base-posp-tagger"
CBD_BERT_PATH = "./model/indobert_finetuned_clause.pt"
CBD_CRF_PATH = "./model/indobert_sklearn_crf_clause_model.pkl"

# --- CONFIG FILE TEXT ---
NEGATIVE_KEYWORDS_FILE = "./model/negative.txt" # Format: 1 kata per baris (enter)
POSITIVE_KEYWORDS_FILE = "./model/positive.txt" # Format: 1 kata per baris (enter)
STOP_KELUHAN_FILE = "./model/badword.txt"       # Format: Dipisah koma (kata1, kata2)

MAX_LEN_CBD = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODIFIER WORDS (Penentu Konteks) ---
NEGATORS = {"tidak", "kurang", "belum", "bukan", "enggak", "gak", "tak", "jangan"}

# --- KAMUS NORMALISASI (SLANG -> BAKU) ---
# Dimuat global agar bisa dipakai di cleaning
key_norm = {
    "yg": "yang", "gak": "tidak", "ga": "tidak", "g": "tidak", "nggak": "tidak",
    "kalo": "kalau", "klo": "kalau", "kl": "kalau",
    "bgt": "banget", "bg": "banget",
    "dgn": "dengan", "dg": "dengan",
    "krn": "karena", "karna": "karena",
    "tdk": "tidak", "tak": "tidak",
    "jd": "jadi", "jdi": "jadi",
    "bkn": "bukan",
    "sdh": "sudah", "udah": "sudah", "dh": "sudah",
    "blm": "belum",
    "tp": "tapi", "tpi": "tapi",
    "sy": "saya", "aku": "saya", "gue": "saya", "gw": "saya",
    "km": "kamu", "lu": "kamu", "loe": "kamu",
    "bgs": "bagus", "good": "bagus",
    "bad": "jelek", "worse": "buruk",
    "jg": "juga", "utk": "untuk",
    "bnyak": "banyak", "bnyk": "banyak",
    "d": "di", "org": "orang", "dlm": "dalam",
    "aja": "saja", "ae": "saja",
    "bs": "bisa", "bisa": "bisa",
    "sm": "sama", "dr": "dari",
    "tmp": "tempat", "tmpt": "tempat",
    "jln": "jalan", "skrg": "sekarang",
    "tau": "tahu", "msh": "masih",
    "sngt": "sangat", "thx": "terima kasih", "makasih": "terima kasih"
}

# --- FUNGSI LOADER FILES ---
def load_keywords_newline(filepath):
    keywords = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word: keywords.add(word)
            print(f"✓ Berhasil memuat {len(keywords)} kata dari {filepath}")
        except Exception as e:
            print(f"✗ Gagal membaca file {filepath}: {e}")
    else:
        print(f"⚠ File {filepath} tidak ditemukan! Menggunakan set kosong.")
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
            print(f"✓ Berhasil memuat {len(keywords)} stop words dari {filepath}")
        except Exception as e:
            print(f"✗ Gagal membaca file {filepath}: {e}")
    else:
        return {"membuat", "datang", "memberi", "memberikan", "mengambil", "menjadi", "bikin"}
    return keywords

# --- LOAD VARIABLES ---
NEGATIVE_KEYWORDS = load_keywords_newline(NEGATIVE_KEYWORDS_FILE)
POSITIVE_KEYWORDS = load_keywords_newline(POSITIVE_KEYWORDS_FILE)
STOP_KELUHAN = load_keywords_comma(STOP_KELUHAN_FILE)


# ============================================================================
# 2. FUNGSI PREPROCESSING & NORMALISASI
# ============================================================================
def clean_text_user(text):
    if not text: return ""
    text = str(text).lower() # Case folding
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Hapus tanda baca/karakter aneh
    text = re.sub(r'\s+', ' ', text).strip() # Normalisasi spasi

    # --- STEP TAMBAHAN: NORMALISASI ---
    # Memecah kalimat jadi kata, cek di kamus slang, lalu gabung lagi
    text = ' '.join([key_norm.get(word, word) for word in text.split()])

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
        print("✓ Model CBD Siap.")
    else:
        print("✗ Model CBD files missing.")
except Exception as e:
    print(f"✗ Gagal load CBD: {e}")

# ============================================================================
# 4. HELPER FUNCTIONS & LOGIC
# ============================================================================

# --- A. CBD Helpers ---
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
            if tok == "[UNK]": continue
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
    # Normalisasi dulu sebelum prediksi sentimen
    norm_text = clean_text_user(text)
    inputs = tokenizer(norm_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred_id])[0]

# --- C. POS & Extraction Helpers ---

def fix_broken_subwords(pos_results):
    if not pos_results: return []
    merged_tokens = []
    
    current_word = pos_results[0]['word']
    current_label = pos_results[0]['entity_group']
    current_end = pos_results[0]['end']

    for i in range(1, len(pos_results)):
        token = pos_results[i]
        word = token['word']
        label = token['entity_group']
        start = token['start']
        end = token['end']

        if start == current_end:
            clean_subword = word.replace('##', '') 
            current_word += clean_subword
            current_end = end
        else:
            merged_tokens.append({'word': current_word, 'entity_group': current_label})
            current_word = word
            current_label = label
            current_end = end

    merged_tokens.append({'word': current_word, 'entity_group': current_label})
    return merged_tokens

def merge_noun_phrases(tokens):
    merged = []
    i = 0
    while i < len(tokens):
        word, label = tokens[i]
        if label == "NNO" and i + 1 < len(tokens) and tokens[i+1][1] == "NNO":
            new_word = f"{word} {tokens[i+1][0]}"
            merged.append((new_word, "NNO"))
            i += 2 
        else:
            merged.append((word, label))
            i += 1
    return merged

def map_model_label_to_user_label(label):
    label = label.upper()
    if label in ["NOUN", "PROPN"]: return "NNO"
    if label == "ADJ": return "ADJ"
    if label == "VERB": return "VBI"
    return label

def find_original_word(full_text, word):
    try:
        # Karena kita pakai normalisasi, full_text di sini bisa jadi beda (masih raw)
        # Tapi kita coba cari dulu di raw text.
        match = re.search(re.escape(word), full_text, re.IGNORECASE)
        return match.group(0) if match else word
    except:
        return word

# ==========================================
# 5. CORE LOGIC (STRICT FILTER & SLANG HANDLING)
# ==========================================

# DAFTAR KATA "SAMPAH"
# ==========================================
# FINAL TUNING: ANTI-NOISE & STRICT PHRASING
# ==========================================

# 1. PERLUAS DAFTAR KATA SAMPAH (IGNORE WORDS)
# Tambahkan kata-kata waktu, tempat, dan saran yang sering muncul di screenshot
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
    
    # KATA WAKTU (PENYEBAB UTAMA ERROR DI SCREENSHOT 3)
    "sekarang", "tadi", "nanti", "kemarin", "besok", "lusa",
    "hari", "minggu", "bulan", "tahun", "jam", "menit", "detik",
    "saat", "waktu", "pas", "ketika", "sejak", "baru", "lama", "dahulu", "dulu",
    
    # KATA SARAN (PENYEBAB ERROR "PERLU PEMBENAHAN")
    "perlu", "harus", "wajib", "mesti", "sebaiknya", "seharusnya", "akan"
}

# 2. LIST NEGASI SLANG
SLANG_NEGATORS = {"ga", "gak", "tak", "enggak", "ora", "kagak", "nda", "ndak", "tdk"}

# 3. SET KATA POSITIF (Manual define jika positive.txt belum sempurna)
# Ini penting agar "kurang baik" tertangkap, tapi "baik" sendirian dibuang
MANUAL_POSITIVE = {
    "baik", "bagus", "indah", "cantik", "nyaman", "aman", "bersih", "ramah", 
    "luas", "terawat", "rapi", "sejuk", "dingin", "puas", "enak", "sedap",
    "mantap", "keren", "oke", "layak", "memadai", "profesional", "sigap"
}

def extract_complaints_user_algo(raw_text):
    if not pos_pipeline: return []
    
    # 1. NORMALISASI TEKS
    norm_text = clean_text_user(raw_text)

    # 2. POS Tagging
    try:
        hasil_pos_raw = pos_pipeline(norm_text)
    except Exception:
        return []

    # 3. Fix Broken Subwords
    hasil_pos = fix_broken_subwords(hasil_pos_raw)

    tokens = []
    for t in hasil_pos:
        word = t['word'].strip().lower()
        word = re.sub(r'[^a-z0-9]', '', word) 
        
        if not word: continue

        # Normalisasi Negasi
        if word in SLANG_NEGATORS: word = "tidak"

        user_label = map_model_label_to_user_label(t['entity_group'])
        
        # Override Label
        if word in NEGATIVE_KEYWORDS: user_label = "ADJ"
        if word in POSITIVE_KEYWORDS or word in MANUAL_POSITIVE: user_label = "POS_ADJ" # Label khusus Positif
        if word == "tidak": user_label = "NEG"
        if word == "kurang": user_label = "NEG" # Perlakukan 'kurang' sama dengan 'tidak'
        
        tokens.append((word, user_label))

    # 4. Gabungkan Noun Phrases
    tokens = merge_noun_phrases(tokens)

    keluhan_list = []
    
    # 5. LOGIKA PENCARIAN
    for i, (word, label) in enumerate(tokens):
        
        # HANYA PROSES KATA BENDA (NNO)
        if label == "NNO":
            original_nno = word 
            
            # Skip jika NNO adalah kata yang di-ignore (waktu/tempat/saran)
            if word in IGNORE_WORDS: continue

            closest_keluhan = None
            
            # === STRATEGI A: CARI KE KANAN (Noun + ... + Adj) ===
            # Contoh: "Pelayanan kurang baik", "Toilet kotor"
            for j in range(i + 1, min(i + 6, len(tokens))):
                w2, l2 = tokens[j]
                
                if w2 in IGNORE_WORDS: continue 
                
                prev_w = tokens[j-1][0] if j > 0 else ""
                
                # Cek Negator (tidak/kurang/belum)
                has_negator = prev_w in ["tidak", "kurang", "belum", "bukan", "jangan"]
                
                # KASUS 1: Noun + Negator + Kata Apapun (Kecuali Ignore)
                # Contoh: "Tiket tidak sepadan", "Air tidak ada"
                if has_negator:
                    if w2 not in IGNORE_WORDS:
                        # Tangkap paket lengkap: "Pelayanan kurang baik"
                        closest_keluhan = f"{original_nno} {prev_w} {w2}"
                        break

                # KASUS 2: Noun + Kata Negatif Langsung
                # Contoh: "Toilet kotor", "Akses susah"
                # Syarat: Label ADJ/NEG ATAU ada di list negatif, DAN BUKAN kata positif
                is_negative_context = (w2 in NEGATIVE_KEYWORDS) or (l2 in ["ADJ", "NEG"] and l2 != "POS_ADJ" and w2 not in MANUAL_POSITIVE)
                
                if is_negative_context:
                    if w2 in STOP_KELUHAN: continue
                    closest_keluhan = f"{original_nno} {w2}"
                    break 

            # === STRATEGI B: CARI KE KIRI (Adj + ... + Noun) ===
            # Contoh: "Kurang nyaman suasananya"
            if not closest_keluhan: 
                for j in range(i - 1, max(i - 6, -1), -1):
                    w2, l2 = tokens[j]
                    
                    if w2 in IGNORE_WORDS: continue
                    
                    # Cek Modifier di sebelah kiri kata sifatnya
                    # Contoh: [kurang] [nyaman] [tempat] -> prev_modifier adalah 'kurang'
                    prev_modifier = tokens[j-1][0] if j > 0 else ""
                    has_negator = prev_modifier in ["tidak", "kurang", "belum", "bukan"]

                    # KASUS 1: Negator + Kata Positif + Noun (DIBALIK)
                    # Contoh: "Kurang nyaman tempatnya" -> "tempatnya kurang nyaman"
                    if (w2 in MANUAL_POSITIVE or l2 == "POS_ADJ") and has_negator:
                         closest_keluhan = f"{original_nno} {prev_modifier} {w2}"
                         break

                    # KASUS 2: Kata Negatif + Noun (DIBALIK)
                    # Contoh: "Rusak jalannya" -> "jalannya rusak"
                    is_negative_context = (w2 in NEGATIVE_KEYWORDS) or (l2 in ["ADJ", "NEG"] and l2 != "POS_ADJ" and w2 not in MANUAL_POSITIVE)
                    
                    if is_negative_context:
                        if w2 in STOP_KELUHAN: continue
                        
                        if has_negator:
                            closest_keluhan = f"{original_nno} {prev_modifier} {w2}"
                        else:
                            closest_keluhan = f"{original_nno} {w2}"
                        break

            if closest_keluhan:
                keluhan_list.append(closest_keluhan)

    return list(dict.fromkeys(keluhan_list))


# ============================================================================
# 6. FLASK ROUTES
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
        # 1. Pecah Klausa
        clauses = detect_clauses(review)
        
        for clause in clauses:
            # 2. Cek Sentimen (gunakan raw, didalam fungsi sentimen ada normalisasi)
            clause_sent = predict_sentiment_global(clause)
            
            if clause_sent.lower() == 'positif': count_pos += 1
            elif clause_sent.lower() == 'negatif': count_neg += 1
            else: count_neu += 1
            
            # 3. Ekstraksi (Hanya Negatif/Netral)
            if clause_sent.lower() in ['negatif', 'netral']:
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