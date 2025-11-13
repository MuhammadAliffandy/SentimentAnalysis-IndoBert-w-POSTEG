# ============================================================================
# CLAUSE BOUNDARY DETECTION (CBD)
# Tujuan: Deteksi batas-batas klausa dalam kalimat bahasa Indonesia
# Menggunakan: Fine-tuned IndoBERT + CRF (Conditional Random Field)
# ============================================================================

# ============================================================================
# IMPORT LIBRARY
# ============================================================================
# Pandas: Data processing (baca/tulis CSV)
import pandas as pd

# PyTorch: Deep learning framework
import torch
import torch.nn as nn

# Hugging Face Transformers: BERT dan tokenizer
from transformers import AutoTokenizer, AutoModel

# Pickle: Load/save model Python object
import pickle

# OS: Operasi file dan folder
import os


# ============================================================================
# KONFIGURASI GLOBAL
# ============================================================================
# Nama model BERT pre-trained dari Hugging Face
MODEL_NAME = "indobenchmark/indobert-base-p1"

# Path file model BERT yang sudah di-fine-tune
BERT_MODEL_PATH = "indobert_finetuned_clause.pt"

# Path file CRF model (scikit-learn CRF)
CRF_MODEL_PATH = "indobert_sklearn_crf_clause_model.pkl"

# Path file CSV test dataset
TEST_CSV = "Dataset-test.csv"

# Max length untuk tokenization BERT
MAX_LEN = 128

# Device: CUDA (GPU) atau CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# FINE-TUNED BERT MODEL CLASS
# ============================================================================
class IndoBERT_FineTune(nn.Module):
    """
    Custom BERT model untuk Clause Boundary Detection.
    
    Struktur:
    - BERT encoder (pre-trained dari Hugging Face)
    - Dropout layer (regularisasi)
    - Linear classifier (output layer)
    """
    
    def __init__(self, model_name, num_labels):
        """
        Constructor class.
        
        Args:
            model_name (str): Nama model BERT pre-trained
            num_labels (int): Jumlah label output (3: B-CLAUSE, I-CLAUSE, O)
        """
        super().__init__()
        
        # Load BERT pre-trained dari Hugging Face
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Dropout layer: dropout 30% untuk regularisasi
        self.dropout = nn.Dropout(0.3)
        
        # Linear layer: transformasi BERT hidden state -> num_labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def get_features(self, input_ids, attention_mask):
        """
        Ekstrak BERT features tanpa classifier.
        
        Args:
            input_ids: Tensor token IDs dari BERT tokenizer
            attention_mask: Tensor attention mask (1 untuk token real, 0 untuk padding)
        
        Returns:
            Tensor: BERT features setelah dropout (shape: [batch_size, seq_len, hidden_size])
        """
        # Forward pass ke BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # Ambil last hidden state (token-level embeddings)
        # Shape: [batch_size, seq_len, hidden_size]
        features = self.dropout(outputs.last_hidden_state)
        
        return features


# ============================================================================
# LOAD MODELS (BERT + CRF)
# ============================================================================
print("="*60)
print("CLAUSE BOUNDARY DETECTION - Fine-tuned IndoBERT + CRF")
print("="*60)

# ================================================================
# CHECK: Apakah file model BERT ada
# ================================================================
if not os.path.exists(BERT_MODEL_PATH):
    # Jika file tidak ada, tampilkan error dan exit
    print(f"\n❌ Fine-tuned BERT model tidak ditemukan: {BERT_MODEL_PATH}")
    print("Jalankan train_test_clause.py terlebih dahulu!")
    exit(1)

# ================================================================
# CHECK: Apakah file model CRF ada
# ================================================================
if not os.path.exists(CRF_MODEL_PATH):
    # Jika file tidak ada, tampilkan error dan exit
    print(f"\n❌ CRF model tidak ditemukan: {CRF_MODEL_PATH}")
    print("Jalankan train_test_clause.py terlebih dahulu!")
    exit(1)

# ================================================================
# LOAD TOKENIZER
# ================================================================
print(f"\n✓ Loading fine-tuned BERT dari {BERT_MODEL_PATH}...")
# Load tokenizer dari model pre-trained
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ================================================================
# LOAD BERT MODEL
# ================================================================
# Instantiate custom BERT model class dengan num_labels=3 (B, I, O)
bert_model = IndoBERT_FineTune(MODEL_NAME, num_labels=3)

# Load state dict dari file yang sudah di-train
bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))

# Pindahkan model ke device (GPU atau CPU)
bert_model = bert_model.to(device)

# Set model ke evaluation mode (disable dropout, batch norm, dll)
bert_model.eval()

# ================================================================
# LOAD CRF MODEL
# ================================================================
print(f"✓ Loading CRF model dari {CRF_MODEL_PATH}...")
# Load CRF model dari pickle file
with open(CRF_MODEL_PATH, 'rb') as f:
    crf_model = pickle.load(f)

# ================================================================
# KONFIRMASI: Semua model berhasil dimuat
# ================================================================
print("✓ Semua model berhasil dimuat!")


# ============================================================================
# FUNGSI: EKSTRAKSI BERT FEATURES
# ============================================================================
def extract_bert_features_finetuned(tokens):
    """
    Ekstraksi BERT features dari fine-tuned model untuk token list.
    
    Args:
        tokens (list): List token (sudah di-tokenize dari text)
    
    Returns:
        numpy.ndarray: BERT features dengan shape (len(tokens), hidden_size)
    """
    # Tokenize dengan BERT tokenizer
    # is_split_into_words=True: token sudah di-split (tidak perlu split lagi)
    # return_tensors="pt": return PyTorch tensor
    # truncation=True: potong jika > max_length
    # padding='max_length': pad ke max_length
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    ).to(device)
    
    # Forward pass ke BERT model (tanpa gradient)
    with torch.no_grad():
        # Ekstrak features dari fine-tuned BERT
        features = bert_model.get_features(encoded['input_ids'], encoded['attention_mask'])
        
        # Ambil batch pertama, truncate ke jumlah token asli, convert ke numpy
        # Shape: (len(tokens), hidden_size)
        features = features[0][:len(tokens)].cpu().numpy()
    
    return features


# ============================================================================
# FUNGSI: DETEKSI KLAUSA
# ============================================================================
def detect_clauses(text):
    """
    Deteksi boundary klausa dari teks.
    
    Args:
        text (str): Kalimat input
    
    Returns:
        list: List klausa yang terdeteksi
    """
    import re
    
    # Split text by punctuation marks (., !, ?)
    # Contoh: "Rumah ini indah. Tempat tinggal yang nyaman!" 
    #         -> ['Rumah ini indah', '.', 'Tempat tinggal yang nyaman', '!', '']
    sub_sentences = re.split(r'([.!?])', text.strip())
    
    # List untuk menyimpan semua klausa yang terdeteksi
    all_clauses = []
    
    # ================================================================
    # LOOP: Process setiap sub-sentence (teks antar punctuation)
    # ================================================================
    # range(0, len, 2): ambil index 0, 2, 4, ... (teks, bukan punctuation)
    for i in range(0, len(sub_sentences), 2):
        # Ambil sub-sentence dan strip whitespace
        sub_text = sub_sentences[i].strip()
        
        # Skip jika kosong
        if not sub_text:
            continue
        
        # ================================================================
        # TOKENIZATION: Tokenize sub-sentence
        # ================================================================
        # Tokenize dengan BERT tokenizer
        tokens = tokenizer.tokenize(sub_text)
        
        # Skip jika tidak ada token
        if not tokens:
            continue
        
        # ================================================================
        # EKSTRAKSI FEATURES: Ambil BERT features dari fine-tuned model
        # ================================================================
        features = extract_bert_features_finetuned(tokens)
        
        # ================================================================
        # PREDIKSI: CRF predict tags (B-CLAUSE, I-CLAUSE, O)
        # ================================================================
        # CRF model predict BIO tags untuk setiap token
        pred_tags = crf_model.predict_single(features)
        
        # ================================================================
        # PARSE BIO TAGS: Konversi BIO tags -> klausa
        # ================================================================
        # Current clause accumulator
        current_clause = []
        
        # Loop setiap token dan tag
        for tok, tag in zip(tokens, pred_tags):
            # B-CLAUSE: Beginning of new clause
            if tag == "B-CLAUSE":
                # Jika ada clause sebelumnya, simpan ke all_clauses
                if current_clause:
                    all_clauses.append(" ".join(current_clause))
                # Mulai clause baru
                current_clause = [tok]
            
            # I-CLAUSE: Inside clause (continuation)
            elif tag == "I-CLAUSE":
                # Tambah token ke current clause
                current_clause.append(tok)
            
            # O: Outside (tidak ada clause)
            elif tag == "O":
                # Jika ada clause sebelumnya, simpan
                if current_clause:
                    all_clauses.append(" ".join(current_clause))
                    # Reset current clause
                    current_clause = []
        
        # ================================================================
        # CLEANUP: Simpan sisa clause jika ada
        # ================================================================
        if current_clause:
            all_clauses.append(" ".join(current_clause))
    
    # Return list semua klausa yang terdeteksi
    return all_clauses


# ============================================================================
# FUNGSI: EKSTRAK DAN SIMPAN KLAUSA KE CSV
# ============================================================================
def extract_and_save_clauses(csv_path=TEST_CSV, output_csv="output_extracted_clauses.csv"):
    """
    Ekstrak klausa dari CSV test dan simpan ke file CSV output.
    
    Args:
        csv_path (str): Path file input CSV
        output_csv (str): Path file output CSV (default: output_extracted_clauses.csv)
    """
    print("\n" + "="*60)
    print(f"EKSTRAKSI KLAUSA - {csv_path}")
    print("="*60)
    
    # ================================================================
    # CHECK: Apakah file CSV ada
    # ================================================================
    if not os.path.exists(csv_path):
        print(f"\n❌ File tidak ditemukan: {csv_path}")
        return
    
    # ================================================================
    # LOAD CSV: Baca dengan berbagai encoding
    # ================================================================
    try:
        # Try latin1 encoding dulu
        df = pd.read_csv(csv_path, encoding='latin1')
    except:
        try:
            # Fallback ke UTF-8
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            # Jika semua gagal, print error dan return
            print(f"❌ Error membaca CSV: {e}")
            return
    
    # Konfirmasi: dataset berhasil dimuat
    print(f"✓ Dataset loaded: {len(df)} baris")
    
    # ================================================================
    # DETECT COLUMN: Cari kolom teks (komentar, comment, text, dll)
    # ================================================================
    text_col = None
    # Loop semua kolom dan cari yang namanya cocok
    for col in df.columns:
        if col.lower() in ['komentar', 'comment', 'text', 'ulasan', 'review']:
            text_col = col
            break
    
    # Jika tidak ketemu, gunakan kolom pertama
    if text_col is None:
        text_col = df.columns[0]
    
    print(f"✓ Menggunakan kolom: {text_col}\n")
    
    # ================================================================
    # PROCESS: Loop semua baris dan ekstrak klausa
    # ================================================================
    # List untuk menyimpan hasil (setiap klausa 1 baris)
    results = []
    # Counter total klausa
    total_clauses = 0
    
    print("Processing...")
    # Loop setiap baris
    for idx in range(len(df)):
        # Ambil text dari kolom terpilih
        text = str(df.iloc[idx][text_col]).strip()
        
        # Skip jika text kosong atau NaN
        if not text or text.lower() == 'nan':
            continue
        
        # ================================================================
        # DETEKSI: Jalankan fungsi detect_clauses
        # ================================================================
        clauses = detect_clauses(text)
        # Update total counter
        total_clauses += len(clauses)
        
        # ================================================================
        # SIMPAN: Setiap klausa sebagai 1 baris di results
        # ================================================================
        for clause in clauses:
            results.append({'klausa': clause})
        
        # Progress indicator setiap 10 baris
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(df)} selesai...")
    
    # ================================================================
    # SAVE: Konversi ke DataFrame dan simpan ke CSV
    # ================================================================
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # ================================================================
    # OUTPUT: Tampilkan ringkasan dan preview
    # ================================================================
    print(f"\n✅ Ekstraksi selesai!")
    print(f"📊 Total: {total_clauses} klausa")
    print(f"✓ Hasil disimpan ke: {output_csv}")
    
    # Preview: 10 baris pertama
    print(f"\n{'='*60}")
    print("PREVIEW HASIL (10 baris pertama):")
    print(f"{'='*60}\n")
    print(result_df.head(10).to_string(index=False))


# ============================================================================
# FUNGSI: TEST MODE (INTERACTIVE)
# ============================================================================
def test_dataset(csv_path=TEST_CSV, n_samples=None):
    """
    Test deteksi klausa dengan Dataset CSV (mode interactive).
    Hasil ditampilkan di terminal, tidak disimpan.
    
    Args:
        csv_path (str): Path file CSV test
        n_samples (int): Jumlah baris yang diproses (None = semua)
    """
    print("\n" + "="*60)
    print(f"TEST DETEKSI KLAUSA - {csv_path}")
    print("="*60)
    
    # ================================================================
    # CHECK: Apakah file CSV ada
    # ================================================================
    if not os.path.exists(csv_path):
        print(f"\n❌ File tidak ditemukan: {csv_path}")
        return
    
    # ================================================================
    # LOAD CSV: Baca dengan berbagai encoding
    # ================================================================
    try:
        df = pd.read_csv(csv_path, encoding='latin1')
    except:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            print(f"❌ Error membaca CSV: {e}")
            return
    
    print(f"✓ Dataset loaded: {len(df)} baris")
    
    # ================================================================
    # DETECT COLUMN: Cari kolom teks
    # ================================================================
    text_col = None
    for col in df.columns:
        if col.lower() in ['komentar', 'comment', 'text', 'ulasan', 'review']:
            text_col = col
            break
    
    if text_col is None:
        text_col = df.columns[0]
    
    print(f"✓ Menggunakan kolom: {text_col}\n")
    
    # ================================================================
    # DETERMINE: Jumlah sample yang diproses
    # ================================================================
    # Jika n_samples None, proses semua baris
    if n_samples is None:
        n_samples = len(df)
    
    # ================================================================
    # PROCESS: Loop baris dan tampilkan hasil di terminal
    # ================================================================
    # Counter total klausa
    total_clauses = 0
    
    # Loop setiap baris (max n_samples)
    for idx in range(min(n_samples, len(df))):
        # Ambil text dari kolom
        text = str(df.iloc[idx][text_col]).strip()
        
        # Skip jika kosong
        if not text or text.lower() == 'nan':
            continue
        
        # ================================================================
        # DISPLAY: Tampilkan nomor baris dan text
        # ================================================================
        print(f"\n[Baris {idx+1}]")
        print(f"📝 Teks: {text}")
        
        # ================================================================
        # DETEKSI: Jalankan detect_clauses
        # ================================================================
        clauses = detect_clauses(text)
        total_clauses += len(clauses)
        
        # ================================================================
        # OUTPUT: Tampilkan klausa yang terdeteksi
        # ================================================================
        print(f"\n✅ Klausa terdeteksi ({len(clauses)}):")
        if clauses:
            # List setiap klausa dengan nomor
            for i, clause in enumerate(clauses, 1):
                print(f"   {i}. {clause}")
        else:
            # Jika tidak ada klausa
            print("   (Tidak ada klausa terdeteksi)")
        
        # Separator antar result
        print("\n" + "-" * 60)
    
    # ================================================================
    # SUMMARY: Tampilkan statistik total
    # ================================================================
    print(f"\n📊 Total: {total_clauses} klausa terdeteksi dari {min(n_samples, len(df))} baris")


# ============================================================================
# MAIN: USER INTERACTION
# ============================================================================
if __name__ == "__main__":
    # ================================================================
    # MENU 1: Pilih mode (ekstrak ke CSV atau test interactive)
    # ================================================================
    print("\n" + "="*60)
    print("OPSI:")
    print("1. Ekstrak semua klausa dan simpan ke CSV")
    print("2. Test interactive (tampilkan di terminal)")
    print("="*60)
    
    # Minta input user
    choice = input("\nPilihan (1/2): ").strip()
    
    # ================================================================
    # CHOICE 1: Ekstrak dan simpan ke CSV
    # ================================================================
    if choice == "1":
        # Minta nama file output
        output_file = input("Nama file output (default: output_extracted_clauses.csv): ").strip()
        # Jika kosong, gunakan default
        if not output_file:
            output_file = "output_extracted_clauses.csv"
        # Jalankan ekstraksi
        extract_and_save_clauses(output_csv=output_file)
    
    # ================================================================
    # CHOICE 2: Test mode (interactive)
    # ================================================================
    elif choice == "2":
        # Sub-menu: pilih test option
        print("\nOPSI TEST:")
        print("1. Test 10 baris pertama")
        print("2. Test semua data")
        print("3. Test dengan jumlah custom")
        
        # Minta input sub-choice
        test_choice = input("\nPilihan (1/2/3): ").strip()
        
        # ================================================================
        # SUB-CHOICE 1: Test 10 baris
        # ================================================================
        if test_choice == "1":
            test_dataset(n_samples=10)
        
        # ================================================================
        # SUB-CHOICE 2: Test semua baris
        # ================================================================
        elif test_choice == "2":
            test_dataset()
        
        # ================================================================
        # SUB-CHOICE 3: Test custom jumlah baris
        # ================================================================
        elif test_choice == "3":
            try:
                # Minta input jumlah baris
                n = int(input("Jumlah baris: ").strip())
                # Jalankan test dengan n sample
                test_dataset(n_samples=n)
            except ValueError:
                # Jika input bukan angka
                print("❌ Input tidak valid!")
        else:
            # Jika pilihan invalid
            print("❌ Pilihan tidak valid!")
    
    # ================================================================
    # INVALID CHOICE: Pilihan tidak sesuai
    # ================================================================
    else:
        print("❌ Pilihan tidak valid!")
