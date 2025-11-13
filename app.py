# ============================================================================
# SENTIMENT ANALYSIS API - FLASK APPLICATION
# Tujuan: REST API untuk analisis sentimen ulasan wisata menggunakan IndoBERT
# Fitur: Prediksi sentimen, ekstraksi keluhan, NLP dengan POS tagging
# ============================================================================

# ============================================================================
# IMPORT LIBRARY
# ============================================================================
# PyTorch: Deep learning framework untuk inference model neural network
import torch

# Transformers: Library Hugging Face untuk BERT dan pipeline NLP
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import pipeline

# Joblib: Untuk load/save model scikit-learn (label encoder)
import joblib

# Flask: Web framework untuk membuat REST API
from flask import Flask, request, jsonify, render_template, abort, send_from_directory

# Standar library
import os          # Operasi file dan folder
import json        # Parse JSON
from collections import Counter  # Hitung frekuensi kata
import pandas as pd  # Data processing (opsional, untuk export/analisis)


# ============================================================================
# KONFIGURASI PATH MODEL DAN RESOURCES
# ============================================================================
# Path ke model IndoBERT yang sudah disimpan (training output)
MODEL_PATH = "./saved_model"

# Path ke label encoder yang menyimpan mapping: angka -> label sentimen
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Model tokenizer pre-trained IndoBERT dari Hugging Face
PRETRAINED_TOKENIZER = "indobenchmark/indobert-base-p1"


# ============================================================================
# LOAD LABEL ENCODER (Mengubah prediksi numerik menjadi label: Positif/Negatif/Netral)
# ============================================================================
try:
    # Load label encoder yang di-save saat training
    # Label encoder menyimpan: [0: 'negatif', 1: 'netral', 2: 'positif'] (contoh)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    # Hitung jumlah label (kelas) untuk konfigurasi model
    num_labels = len(label_encoder.classes_)
    print(f"[INFO] Label encoder loaded with {num_labels} classes.")
except FileNotFoundError:
    raise FileNotFoundError(
        f"{LABEL_ENCODER_PATH} not found. "
        "Kalau mau test cepat, gunakan test_app.py yang mem-mock model."
    )
except Exception as e:
    raise RuntimeError(f"Error loading label encoder: {e}")


# ============================================================================
# LOAD TOKENIZER (Mengubah teks menjadi token/ID angka yang dipahami model)
# ============================================================================
try:
    # Load tokenizer IndoBERT dari Hugging Face
    # Tokenizer akan memecah teks -> token -> convert ke ID
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_TOKENIZER)
    print("[INFO] Tokenizer loaded.")
except Exception as e:
    raise RuntimeError(f"Error loading tokenizer: {e}")


# ============================================================================
# LOAD MODEL BERT (Model neural network untuk sentiment classification)
# ============================================================================
try:
    # Load model BertForSequenceClassification dari folder saved_model
    # Model ini sudah di-fine-tune untuk sentiment analysis bahasa Indonesia
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels)
    
    # Set model ke evaluation mode (matikan dropout, BatchNorm, dll)
    model.eval()
    print("[INFO] Model loaded.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


# ============================================================================
# DEVICE SETUP (GPU atau CPU)
# ============================================================================
# Deteksi apakah CUDA (GPU NVIDIA) tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pindahkan model ke device (GPU lebih cepat untuk inference)
model.to(device)
print(f"[INFO] Using device: {device}")


# ============================================================================
# LOAD POS TAGGER (Part-of-Speech Tagging untuk ekstraksi keluhan)
# ============================================================================
# POS tagging: menentukan jenis kata (Noun, Adjective, Verb, dll)
# Model: Indonesian RoBERTa yang sudah pre-trained untuk POS tagging
pos_model_name = "w11wo/indonesian-roberta-base-posp-tagger"

# Pipeline: Hugging Face pipeline untuk token-level classification
# aggregation_strategy="simple": gabungkan subtoken ke word level
pos_pipeline = pipeline(
    "token-classification",
    model=pos_model_name,
    tokenizer=pos_model_name,
    aggregation_strategy="simple"
)


# ============================================================================
# LOAD BADWORD LIST (Kata-kata negatif tambahan untuk ekstraksi keluhan)
# ============================================================================
badword_file = "badword.txt"
try:
    # Load file badword.txt (format: kata1, kata2, kata3, ...)
    with open(badword_file, "r", encoding="utf-8") as f:
        badword_text = f.read()
    
    # Ubah menjadi set (lowercase) untuk deduplicasi dan pencarian cepat
    negative_keywords = list({w.strip().lower() for w in badword_text.split(",") if w.strip()})
except Exception:
    # Jika file tidak ada, gunakan list kosong (tidak fatal)
    negative_keywords = []


# ============================================================================
# STOP WORDS UNTUK EKSTRAKSI KELUHAN
# ============================================================================
# Stop words: kata yang diabaikan karena terlalu umum atau tidak signifikan
# Contoh: "membuat", "datang" tidak dianggap keluhan yang spesifik
stop_keluhan = {"membuat", "datang", "memberi", "memberikan", "mengambil", "menjadi"}


# ============================================================================
# FUNGSI HELPER: CARI KATA ASLI DI KALIMAT ORIGINAL
# ============================================================================
def find_original_word(text, word):
    """
    Cari kata asli di teks (karena tokenizer mungkin mengubah case/format).
    
    Args:
        text (str): Kalimat asli dari review
        word (str): Kata yang sudah diproses (lowercase, mungkin substring)
    
    Returns:
        str: Kata asli dengan format original (capitalization, dll)
    
    Contoh: 
        text = "Pantainya Sangat Indah"
        word = "pantainya" -> return "Pantainya"
    """
    # Loop setiap kata di kalimat original
    for w in text.split():
        # Jika kata cocok (case-insensitive), return kata original
        if word in w.lower():
            return w
    # Jika tidak ketemu, return kata yang diberikan (fallback)
    return word


# ============================================================================
# FUNGSI HELPER: MERGE NOUN PHRASES (NNO NNO -> NNO tunggal)
# ============================================================================
def merge_noun_phrases(tokens):
    """
    Gabungkan dua noun bersebelahan menjadi satu phrase.
    
    Args:
        tokens (list): List of (word, POS_label) tuples
    
    Returns:
        list: List of (word, POS_label) tuples dengan noun phrases yang sudah digabung
    
    Contoh:
        Input:  [("pantai", "NNO"), ("pasir", "NNO"), ("indah", "ADJ")]
        Output: [("pantai pasir", "NNO"), ("indah", "ADJ")]
    """
    merged = []
    skip = False  # Flag untuk skip token yang sudah digabung
    
    for i in range(len(tokens)):
        # Skip jika sudah digabung ke elemen sebelumnya
        if skip:
            skip = False
            continue
        
        word, label = tokens[i]
        
        # Jika noun dan noun berikutnya, gabungkan
        if label == "NNO" and i+1 < len(tokens) and tokens[i+1][1] == "NNO":
            # Gabung dengan spasi: "pantai" + "pasir" -> "pantai pasir"
            merged.append((word + " " + tokens[i+1][0], "NNO"))
            skip = True  # Skip token berikutnya karena sudah digabung
        else:
            # Token tidak memenuhi kondisi, tambah ke hasil
            merged.append((word, label))
    
    return merged


# ============================================================================
# INISIALISASI FLASK APP
# ============================================================================
app = Flask(__name__)


# ============================================================================
# ROUTE 1: HOME / INFORMASI API
# ============================================================================
@app.route('/')
def home():
    """
    Endpoint root untuk informasi API.
    
    Returns:
        str: Pesan sambutan dan panduan endpoint
    """
    return "Sentiment Analysis API. Use /predict endpoint or open /ui for a simple web UI."


# ============================================================================
# ROUTE 2: UI (User Interface - Halaman Web)
# ============================================================================
@app.route('/ui')
def ui():
    """
    Render halaman web interaktif untuk testing sentiment analysis.
    
    Returns:
        HTML: Dari templates/index.html
    
    Requires:
        File templates/index.html harus ada di project folder
    """
    try:
        # Render template HTML (templates/index.html)
        return render_template("index.html")
    except Exception:
        # Jika template tidak ada, return error 500
        abort(500, description="UI template missing. Put templates/index.html in project folder.")


# ============================================================================
# ROUTE 3: PREDICT SENTIMENT (Single Review)
# ============================================================================
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    Endpoint untuk prediksi sentimen single review.
    
    Request JSON:
        {
            "text": "Pantainya sangat indah dan bersih!"
        }
    
    Response JSON:
        {
            "sentiment": "positif"
        }
    
    Returns:
        JSON: {"sentiment": label} atau error 400 jika request invalid
    """
    # Parse JSON dari request body
    data = request.get_json(silent=True)
    
    # Validasi: pastikan field "text" ada
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid request. Provide JSON with "text" field.'}), 400

    new_review = data['text']
    
    # ====== TOKENIZATION ======
    # Ubah teks string -> token ID + attention mask
    inputs = tokenizer(
        new_review,
        return_tensors="pt",      # Return PyTorch tensor
        padding=True,             # Padding ke max_length
        truncation=True,          # Potong jika > max_length
        max_length=128            # Max token length (BERT standar 512, tapi 128 cukup)
    )
    
    # Pindahkan input tensor ke device (GPU atau CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ====== INFERENCE (Prediksi) ======
    # Jalankan model tanpa menghitung gradien (eval mode)
    with torch.no_grad():
        # outputs.logits: shape (batch_size, num_labels) = (1, 3) untuk 3 sentimen
        outputs = model(**inputs)

    # Ambil class dengan logit tertinggi
    predicted_id = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    # Convert ID -> label string (contoh: 2 -> "positif")
    predicted_label = label_encoder.inverse_transform([predicted_id])[0]

    return jsonify({'sentiment': predicted_label})


# ============================================================================
# ROUTE 4: DAFTAR WISATA (List semua file ulasan)
# ============================================================================
@app.route('/wisata_list', methods=['GET'])
def wisata_list():
    """
    Endpoint untuk mendapatkan daftar semua wisata yang tersedia.
    
    Response JSON:
        {
            "wisata": ["Pantai Papuma Jember", "Jember Mini Zoo", ...]
        }
    
    Returns:
        JSON: {"wisata": [list nama wisata]}
    """
    # Folder "data" tempat menyimpan file JSON ulasan
    folder = os.path.join(os.path.dirname(__file__), 'data')
    
    # Ambil semua file .json
    wisata_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    
    # Convert nama file -> nama wisata
    # Contoh: "pantai_papuma_jember.json" -> "Pantai Papuma Jember"
    wisata_names = [f.replace('.json','').replace('_',' ').title() for f in wisata_files]
    
    return jsonify({'wisata': wisata_names})


# ============================================================================
# ROUTE 5: ANALISIS WISATA (Full sentiment analysis + complaint extraction)
# ============================================================================
@app.route('/analisis_wisata', methods=['POST'])
def analisis_wisata():
    """
    Endpoint untuk analisis lengkap satu wisata:
    - Predict sentiment setiap ulasan
    - Hitung statistik sentimen
    - Extract keluhan dari review negatif
    
    Request JSON:
        {
            "wisata": "Pantai Papuma Jember"
        }
    
    Response JSON:
        {
            "wisata": "...",
            "total": 150,
            "positif": 120,
            "negatif": 20,
            "netral": 10,
            "results": [{"review": "...", "sentiment": "positif"}, ...],
            "complaints": [{"No": 1, "Ulasan Negatif": "...", "Keluhan": "..."}, ...],
            "top_nouns": [{"word": "pantai", "count": 5}, ...]
        }
    """
    # Parse request JSON
    req = request.get_json(silent=True)
    
    # Validasi: pastikan "wisata" field ada
    if not req or 'wisata' not in req:
        return jsonify({'error': 'Missing wisata name'}), 400
    
    wisata_name = req['wisata']
    
    # ====== BACA FILE DATA ======
    # Folder data
    folder = os.path.join(os.path.dirname(__file__), 'data')
    
    # Konvert nama wisata -> nama file
    # Contoh: "Pantai Papuma Jember" -> "pantai_papuma_jember.json"
    filename = wisata_name.lower().replace(' ','_') + '.json'
    filepath = os.path.join(folder, filename)
    
    # Check apakah file ada
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Buka dan parse JSON
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract list ulasan dari JSON
    reviews = data.get('reviews', [])
    
    # ====== INISIALISASI COUNTER ======
    results = []              # Simpan semua hasil prediksi
    negative_reviews = []     # Simpan hanya ulasan negatif (untuk ekstraksi keluhan)
    count_pos, count_neg, count_neu = 0, 0, 0  # Counter sentimen
    
    # ====== LOOP SETIAP REVIEW: PREDICT SENTIMENT ======
    for review in reviews:
        # Tokenization
        inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Prediksi
        predicted_id = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        predicted_label = label_encoder.inverse_transform([predicted_id])[0]
        
        # Hitung berdasarkan sentimen
        if predicted_label.lower() == 'positif':
            count_pos += 1
        elif predicted_label.lower() == 'negatif':
            count_neg += 1
            # Simpan ulasan negatif untuk ekstraksi keluhan nanti
            negative_reviews.append(review)
        else:
            count_neu += 1
        
        # Simpan hasil
        results.append({'review': review, 'sentiment': predicted_label})
    
    # ====== EKSTRAKSI KELUHAN DARI REVIEW NEGATIF ======
    # Hanya proses ulasan negatif (optimasi waktu)
    
    all_nno_words = []        # Kumpul semua noun (untuk top-5 nanti)
    final_data = []           # Hasil akhir keluhan dengan nomor urut
    
    # Loop setiap ulasan negatif
    for idx, kalimat in enumerate(negative_reviews, start=1):
        # ====== POS TAGGING ======
        # Jalankan POS tagger pada kalimat
        hasil_pos = pos_pipeline(kalimat)
        
        # Extract (word, POS_label) dari output tagger
        # Contoh: [("pantai", "NNO"), ("sangat", "ADV"), ("jelek", "ADJ"), ...]
        tokens = [(t['word'].strip().lower(), t['entity_group']) for t in hasil_pos]
        
        # Merge noun phrases: "pantai" + "pasir" -> "pantai pasir"
        tokens = merge_noun_phrases(tokens)
        
        # ====== EKSTRAKSI KELUHAN (Pairing Object + Modifier) ======
        keluhan_pairs = []  # Keluhan untuk kalimat ini
        
        # Loop setiap token untuk cari noun
        for i, (word, label) in enumerate(tokens):
            # Jika token adalah noun (NNO)
            if label == "NNO":
                # Cari kata asli di kalimat (bukan lowercase hasil tokenizer)
                original_nno = find_original_word(kalimat, word)
                
                # Tambahkan ke daftar semua noun (untuk top-5 nanti)
                all_nno_words.append(original_nno)
                
                # ====== CARI MODIFIER TERDEKAT (ADJ, VB, NEG, dll) ======
                closest_keluhan = None
                min_distance = 999  # Jarak minimal ke modifier
                
                # Loop token lain untuk cari modifier
                for j, (w2, l2) in enumerate(tokens):
                    if j != i:  # Jangan compare dengan diri sendiri
                        # Check apakah token adalah modifier
                        if l2 in ["ADJ", "NEG", "VBI", "VBT", "VBP"] or w2 in negative_keywords or w2 == "tidak":
                            # Skip kata umum yang tidak spesifik
                            if w2 in stop_keluhan:
                                continue
                            
                            # Hitung jarak antara noun dan modifier
                            distance = abs(j - i)
                            
                            # Jika jarak <= 4 kata dan lebih dekat dari yang sebelumnya
                            if distance <= 4 and distance < min_distance:
                                min_distance = distance
                                
                                # ====== BUILD KELUHAN PHRASE ======
                                # Special case: "tidak" -> "tidak X"
                                if w2 == "tidak":
                                    if j+1 < len(tokens):
                                        next_word = tokens[j+1][0]
                                        closest_keluhan = f"{original_nno} tidak {next_word}"
                                    else:
                                        closest_keluhan = f"{original_nno} tidak jelas"
                                # Case: "tidak X" -> "tidak X"
                                elif j-1 >= 0 and tokens[j-1][0] == "tidak":
                                    closest_keluhan = f"{original_nno} tidak {w2}"
                                # Case normal: "noun modifier"
                                else:
                                    closest_keluhan = f"{original_nno} {w2}"
                
                # Jika tidak ada modifier ditemukan, default "bermasalah"
                if not closest_keluhan:
                    closest_keluhan = f"{original_nno} bermasalah"
                
                keluhan_pairs.append(closest_keluhan)
        
        # Hapus duplikat keluhan (dict.fromkeys untuk preserve order)
        keluhan_pairs = list(dict.fromkeys(keluhan_pairs))
        
        # Jika tidak ada keluhan terdeteksi
        if not keluhan_pairs:
            keluhan_pairs = ["keluhan tidak terdeteksi"]
        
        # Simpan hasil keluhan untuk kalimat ini
        final_data.append({
            "No": idx,
            "Ulasan Negatif": kalimat,
            "Keluhan": ", ".join(keluhan_pairs)
        })
    
    # ====== TOP-5 NOUN (Objek paling sering dikomplain) ======
    # Hitung frekuensi setiap noun
    nno_counter = Counter(all_nno_words)
    
    # Ambil 5 noun dengan frekuensi tertinggi
    top_5_nno = nno_counter.most_common(5)

    # ====== RETURN RESPONSE ======
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


# ============================================================================
# MAIN: JALANKAN FLASK APP
# ============================================================================
if __name__ == '__main__':
    # Jalankan Flask development server
    # host="0.0.0.0": accessible dari network (bukan hanya localhost)
    # port=5000: port standar Flask
    # debug=True: auto-reload saat ada perubahan kode, detailed error messages
    app.run(host="0.0.0.0", port=5000, debug=True)
