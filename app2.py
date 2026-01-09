# ============================================================================
# APP.PY - SISTEM EKSTRAKSI KELUHAN WISATA
# Fitur: Hybrid CBD + Sentiment IndoBERT + OpenAI GPT Extraction
# ============================================================================

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel
from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import os
import json
import re
import openai  # Library OpenAI
from collections import Counter

# ============================================================================
# 1. KONFIGURASI GLOBAL
# ============================================================================

# --- KONFIGURASI OPENAI (WAJIB DIISI) ---
OPENAI_API_KEY = ""   # ⚠️ TEMPEL API KEY KAMU DI SINI
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- KONFIGURASI PATH MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "saved_model")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
CBD_BERT_PATH = os.path.join(MODEL_DIR, "indobert_finetuned_clause.pt")
CBD_CRF_PATH = os.path.join(MODEL_DIR, "indobert_sklearn_crf_clause_model.pkl")

# --- KONFIGURASI MODEL NAME ---
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN_CBD = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {device}")

# --- KAMUS NORMALISASI (SLANG -> BAKU) ---
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

# ============================================================================
# 2. HELPER FUNCTIONS (PREPROCESSING)
# ============================================================================
def clean_text_user(text):
    if not text: return ""
    text = str(text).lower() # Case folding
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    # Hapus karakter aneh tapi biarkan tanda baca titik/koma untuk CBD
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip() # Normalisasi spasi

    # Normalisasi Slang
    text = ' '.join([key_norm.get(word, word) for word in text.split()])
    return text

# ============================================================================
# 3. DEFINISI KELAS & MODEL LOADER
# ============================================================================

# --- A. Definisi Class IndoBERT untuk CBD (Wajib ada agar pickle bisa load) ---
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

# --- B. Load Models ---
print("Sedang memuat model...")

tokenizer = None
sentiment_model = None
label_encoder = None
cbd_bert_model = None
cbd_crf_model = None

try:
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Sentiment Model & Label Encoder
    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        num_sent_labels = len(label_encoder.classes_)
        
        sentiment_model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, num_labels=num_sent_labels)
        sentiment_model.to(device)
        sentiment_model.eval()
        print("✓ Model Sentimen Siap.")
    else:
        print("⚠ Label Encoder tidak ditemukan.")

    # 3. CBD Models (BERT + CRF)
    # Cek import sklearn_crfsuite (diperlukan untuk load pickle CRF)
    try:
        import sklearn_crfsuite
    except ImportError:
        print("⚠ Library sklearn_crfsuite belum terinstall.")

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
    print(f"Critical Error Loading Models: {e}")


# ============================================================================
# 4. CORE LOGIC FUNCTIONS
# ============================================================================

# --- A. Clause Boundary Detection (CBD) ---
def extract_bert_features_finetuned(tokens):
    """Mendapatkan fitur embedding dari IndoBERT Fine-Tuned"""
    encoded = tokenizer(
        tokens, is_split_into_words=True, return_tensors="pt",
        truncation=True, padding='max_length', max_length=MAX_LEN_CBD
    ).to(device)
    with torch.no_grad():
        features = cbd_bert_model.get_features(encoded['input_ids'], encoded['attention_mask'])
        # Ambil hanya bagian token asli (bukan padding)
        features = features[0][:len(tokens)].cpu().numpy()
    return features

def detect_clauses(text):
    """Memecah paragraf menjadi klausa-klausa pendek menggunakan IndoBERT+CRF"""
    if not cbd_bert_model or not cbd_crf_model:
        # Fallback jika model CBD rusak: Split berdasarkan tanda baca saja
        return re.split(r'[.,!?;]', text)

    # Split kasar dulu berdasarkan tanda baca utama agar tidak terlalu panjang buat BERT
    sub_sentences = re.split(r'([.!?])', text.strip())
    all_clauses = []

    for sub_text in sub_sentences:
        sub_text = sub_text.strip()
        if not sub_text or len(sub_text) < 2: continue # Skip jika kosong atau cuma tanda baca
        
        tokens = tokenizer.tokenize(sub_text)
        if not tokens: continue

        try:
            # Prediksi BIO Tags
            features = extract_bert_features_finetuned(tokens)
            pred_tags = cbd_crf_model.predict_single(features)
        except:
            all_clauses.append(sub_text)
            continue

        # Reconstruct Clauses based on Tags
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

    # Filter klausa yang terlalu pendek
    final_clauses = [c.strip() for c in all_clauses if len(c.strip()) > 3]
    return final_clauses if final_clauses else [text]

# --- B. Sentiment Analysis ---
def predict_sentiment_global(text):
    if not sentiment_model: return "netral"
    # Normalisasi teks sebelum masuk model
    norm_text = clean_text_user(text)
    inputs = tokenizer(norm_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred_id])[0]

# --- C. Extraction using OpenAI GPT (THE NEW LOGIC) ---
# ============================================================================
# PERBAIKAN: EKSTRAKSI KELUHAN (STRICT / KETAT)
# ============================================================================
def extract_complaints_with_gpt(text):
    """
    Menggunakan OpenAI untuk mengekstrak keluhan spesifik.
    HANYA mengambil frasa yang benar-benar ada di teks (Extractive).
    """
    if not text or len(text.strip()) < 3: return []

    # Kita pakai teks yang sudah dibersihkan (slang -> baku) oleh fungsi clean_text_user
    clean_input = clean_text_user(text)

    prompt = f"""
    Tugas: Ekstrak inti keluhan dari teks ulasan wisata berikut.
    
    Teks: "{clean_input}"
    
    Aturan Ketat:
    1. Ambil HANYA frasa inti yang mengandung keluhan (Aspek + Kondisi).
    2. GUNAKAN KATA-KATA YANG ADA DI DALAM TEKS. Jangan mengarang kata benda yang tidak disebutkan.
    3. Hapus kata-kata sambung atau basa-basi (seperti: "sayangnya", "tapi", "padahal", "jujur", "menurut saya").
    4. Contoh:
       - Input: "sayangnya airnya kotor banget" -> Output: ["airnya kotor"]
       - Input: "pelayanan lambat dan tiket mahal" -> Output: ["pelayanan lambat", "tiket mahal"]
       - Input: "toiletnya bau pesing" -> Output: ["toiletnya bau"]
    5. Jika teks berisi pujian atau netral (tidak ada keluhan), kembalikan array kosong [].
    6. Output HARUS JSON Array of Strings.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "Kamu adalah ekstraktor teks keluhan yang presisi."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # Wajib 0 agar tidak kreatif/mengarang
            response_format={"type": "json_object"} # Memaksa output JSON valid
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Penanganan format JSON return dari OpenAI
        # Kadang dia membalas { "keluhan": ["..."] } atau langsung ["..."]
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, list):
                    return val
            return []
        return []

    except Exception as e:
        print(f"Error OpenAI: {e}")
        return []

# ============================================================================
# 5. FLASK ROUTES
# ============================================================================
app = Flask(__name__)

@app.route('/')
def home(): 
    return render_template("index.html")

@app.route('/wisata_list', methods=['GET'])
def wisata_list():
    folder = os.path.join(BASE_DIR, 'data')
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
    filepath = os.path.join(BASE_DIR, 'data', filename)
    
    if not os.path.exists(filepath): return jsonify({'error': 'File not found'}), 404
    
    with open(filepath) as f: data = json.load(f)
    reviews = data.get('reviews', [])
    
    complaint_data = []
    count_pos, count_neg, count_neu = 0, 0, 0
    all_nno_words = []
    
    print(f"Menganalisis {len(reviews)} ulasan untuk {wisata_name}...")

    for review in reviews:
        # 1. Pecah kalimat panjang menjadi klausa (IndoBERT CBD)
        clauses = detect_clauses(review)
        
        for clause in clauses:
            # 2. Cek Sentimen (IndoBERT Sentiment)
            clause_sent = predict_sentiment_global(clause)
            
            if clause_sent.lower() == 'positif': count_pos += 1
            elif clause_sent.lower() == 'negatif': count_neg += 1
            else: count_neu += 1
            
            # 3. Ekstraksi Keluhan (OpenAI GPT)
            # Hanya proses jika sentimen Negatif atau Netral
            if clause_sent.lower() in ['negatif', 'netral']:
                extracted_list = extract_complaints_with_gpt(clause)
                
                if extracted_list:
                    for item in extracted_list:
                        # Statistik Kata Benda (Ambil kata pertama)
                        parts = item.split()
                        if parts:
                            first_word = parts[0].lower()
                            if first_word not in ["yang", "di", "dan", "tapi", "karena"]:
                                all_nno_words.append(first_word)

                    formatted = ", ".join(extracted_list)
                    
                    # === PERBAIKAN DI SINI (PENTING) ===
                    complaint_data.append({
                        "Ulasan Negatif": clause,  # <--- Nama key harus persis ini agar dibaca HTML
                        "Keluhan": formatted
                    })

    # Statistik Kata Terbanyak
    nno_counter = Counter(all_nno_words)
    top_5_nno = nno_counter.most_common(5)

    # Beri nomor urut
    for i, item in enumerate(complaint_data, 1): item['No'] = i

    return jsonify({
        'wisata': wisata_name,
        'total': len(reviews),
        'positif': count_pos,
        'negatif': count_neg,
        'netral': count_neu,
        'results': [], 
        'complaints': complaint_data, # Data dikirim dengan key "Ulasan Negatif"
        'top_nouns': [{'word': noun, 'count': count} for noun, count in top_5_nno]
    })

if __name__ == '__main__':
    print("🚀 Server Flask Berjalan...")
    app.run(host="0.0.0.0", port=5132, debug=True)