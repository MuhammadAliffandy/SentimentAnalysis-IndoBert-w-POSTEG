import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pickle
import os
import re
import numpy as np

# ==============================
# KONFIGURASI
# ==============================
MODEL_NAME = "indobenchmark/indobert-base-p1"
BERT_MODEL_PATH = "../indobert_finetuned_clause.pt"
CRF_MODEL_PATH = "../indobert_sklearn_crf_clause_model.pkl"
TEST_CSV = "Dataset-test.csv"
MAX_LEN = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# CLASS MODEL
# ==============================
class IndoBERT_FineTune(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def get_features(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)

# ==============================
# LOAD MODELS
# ==============================
print("="*60)
print("SMART CLAUSE EXTRACTION (FIXED LOGIC)")
print("="*60)

if not os.path.exists(BERT_MODEL_PATH) or not os.path.exists(CRF_MODEL_PATH):
    print(f"❌ Model tidak ditemukan di path: {BERT_MODEL_PATH}")
    # exit(1) # Commented for debugging flexibility

print(f"✓ Loading Models ke {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = IndoBERT_FineTune(MODEL_NAME, num_labels=3)
try:
    bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
    bert_model = bert_model.to(device)
    bert_model.eval() # PENTING: Matikan dropout saat inferensi
    print("✓ BERT Loaded.")
except Exception as e:
    print(f"❌ Gagal load BERT: {e}")

try:
    with open(CRF_MODEL_PATH, 'rb') as f:
        crf_model = pickle.load(f)
    print("✓ CRF Loaded.")
except Exception as e:
    print(f"❌ Gagal load CRF: {e}")

# ==============================
# FUNGSI EXTRACTION (ALIGNED)
# ==============================
def extract_bert_features_aligned(tokens, tokenizer, model, device):
    # Gabungkan token jadi string untuk tokenizer, lalu tokenize ulang
    # Ini trik agar kita dapat offset mapping yang akurat
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        full_features = model.get_features(
            input_ids=encoding['input_ids'], 
            attention_mask=encoding['attention_mask']
        )

    word_ids = encoding.word_ids(batch_index=0)
    aligned_features = []
    
    # Ambil fitur dari token PERTAMA dari setiap kata asli
    single_seq_features = full_features[0] 
    previous_word_idx = None
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None: continue
        if word_idx != previous_word_idx:
            feat = single_seq_features[idx].cpu().numpy()
            aligned_features.append(feat)
            previous_word_idx = word_idx

    # Potong token jika truncation terjadi
    limit = len(aligned_features)
    return np.array(aligned_features), tokens[:limit]

# ==============================
# LOGIKA DETEKSI CERDAS (REVISED)
# ==============================
def detect_clauses_smart(text):
    if not isinstance(text, str) or not text.strip():
        return []
    
    # 1. Normalisasi Teks
    # Ubah enter jadi spasi, hapus spasi ganda
    clean_text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Tokenisasi Awal (Mempertahankan tanda baca sebagai token terpisah)
    # Regex ini memisahkan kata dan tanda baca
    raw_tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", clean_text)
    
    if not raw_tokens: return []

    # 3. Ekstrak Fitur & Prediksi
    features, valid_tokens = extract_bert_features_aligned(
        raw_tokens, tokenizer, bert_model, device
    )
    
    if len(valid_tokens) == 0: return []
    
    pred_tags = crf_model.predict_single(features)

    # 4. REKONSTRUKSI CERDAS
    clauses = []
    current_clause = []
    
    # Kata penghubung yang TIDAK boleh memisahkan klausa (enumerasi)
    # Contoh: "mandi DAN renang", "murah, enak"
    NO_SPLIT_PREV = [",", "dan", "serta", "atau", "&", "plus", "dengan", "buat", "untuk"]
    
    # Kata penghubung yang WAJIB memisahkan klausa (kontradiksi/sebab-akibat)
    # Contoh: "bagus TAPI mahal", "sepi KARENA hujan"
    FORCE_SPLIT_CURR = ["tapi", "tetapi", "namun", "sedangkan", "melainkan", "padahal", "walau", "meski", "karena", "soalnya", "sehingga", "pas", "waktu", "saat"]

    for i, (tok, tag) in enumerate(zip(valid_tokens, pred_tags)):
        tok_lower = tok.lower()
        prev_tok_lower = valid_tokens[i-1].lower() if i > 0 else ""
        
        should_split = False

        # --- LOGIKA SPLIT ---
        
        # Rule 1: Model mendeteksi B-CLAUSE (Awal Baru)
        if tag == "B-CLAUSE":
            # PENGECUALIAN: Jika kata sebelumnya adalah enumerasi ("dan", ","), abaikan B-CLAUSE
            if prev_tok_lower in NO_SPLIT_PREV:
                should_split = False
            else:
                should_split = True
        
        # Rule 2: Paksa pisah jika ketemu kata sambung kuat (TAPI, KARENA, dll)
        # Ini mengatasi jika model CRF gagal memprediksi B-CLAUSE
        elif tok_lower in FORCE_SPLIT_CURR:
            should_split = True
            
        # Rule 3: Tanda baca akhir kalimat (. ! ?) selalu pisah
        elif tok_lower in ['.', '!', '?']:
            # Append tanda baca dulu ke klausa saat ini, baru nanti di loop berikutnya jadi baru
            current_clause.append(tok)
            if current_clause:
                clauses.append(" ".join(current_clause).replace(" ,", ",").replace(" .", "."))
            current_clause = []
            continue # Lanjut ke token berikutnya (klausa baru mulai kosong)

        # --- EKSEKUSI ---
        
        if should_split and current_clause:
            # Simpan klausa sebelumnya
            clauses.append(" ".join(current_clause).replace(" ,", ",").replace(" .", "."))
            current_clause = [tok] # Mulai klausa baru dengan token ini
        else:
            # Gabung ke klausa yang sedang berjalan
            current_clause.append(tok)
            
    # Simpan sisa klausa terakhir
    if current_clause:
        clauses.append(" ".join(current_clause).replace(" ,", ",").replace(" .", "."))

    # Filter klausa yang terlalu pendek/sampah (misal cuma ".")
    final_clauses = [c for c in clauses if len(c) > 2 or re.search(r'\w', c)]
    
    return final_clauses

# ==============================
# MAIN TEST
# ==============================
if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"TESTING PADA DATASET: {TEST_CSV}")
    print("="*60)

    if os.path.exists(TEST_CSV):
        try:
            df = pd.read_csv(TEST_CSV, encoding='utf-8')
        except:
            df = pd.read_csv(TEST_CSV, encoding='latin1')
            
        # Cari kolom teks secara otomatis
        text_col = next((col for col in df.columns if col.lower() in ['komentar', 'ulasan', 'text', 'content', 'review']), df.columns[0])
        print(f"✓ Kolom Teks: {text_col}")

        # Ambil sampel acak atau 5 teratas
        subset = df.head(10)
        
        for idx, row in subset.iterrows():
            original = str(row[text_col])
            print(f"\n[Baris {idx}] Input: \"{original[:100]}...\"")
            
            results = detect_clauses_smart(original)
            
            for i, res in enumerate(results, 1):
                print(f"   ✅ Klausa {i}: {res}")
    else:
        # Dummy Test jika CSV tidak ada
        print("⚠️ File CSV tidak ditemukan. Menggunakan kalimat contoh.")
        dummy_texts = [
            "Ga wort it orang 6 dewasa kenak 56 anak kecil di itung juga sangat buruk.",
            "Tempatnya bagus tapi toiletnya kotor, bau pesing.",
            "Makanannya enak, murah, dan pelayanannya ramah banget.",
            "Padahal ga mau masuk ke lokasi wisatanya malah tetep disuruh bayar."
        ]
        
        for txt in dummy_texts:
            print(f"\nInput: \"{txt}\"")
            results = detect_clauses_smart(txt)
            for i, res in enumerate(results, 1):
                print(f"   ✅ Klausa {i}: {res}")