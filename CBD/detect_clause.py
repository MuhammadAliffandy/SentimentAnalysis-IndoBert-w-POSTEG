import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pickle
import os

# ==============================
# KONFIGURASI
# ==============================
MODEL_NAME = "indobenchmark/indobert-base-p1"
BERT_MODEL_PATH = "indobert_finetuned_clause.pt"
CRF_MODEL_PATH = "indobert_sklearn_crf_clause_model.pkl"
TEST_CSV = "Dataset-test.csv"
MAX_LEN = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# FINE-TUNED BERT MODEL CLASS
# ==============================
class IndoBERT_FineTune(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def get_features(self, input_ids, attention_mask):
        """Ekstrak BERT features tanpa classifier"""
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        features = self.dropout(outputs.last_hidden_state)
        return features

# ==============================
# LOAD MODELS
# ==============================
print("="*60)
print("CLAUSE BOUNDARY DETECTION - Fine-tuned IndoBERT + CRF")
print("="*60)

if not os.path.exists(BERT_MODEL_PATH):
    print(f"\n❌ Fine-tuned BERT model tidak ditemukan: {BERT_MODEL_PATH}")
    print("Jalankan train_test_clause.py terlebih dahulu!")
    exit(1)

if not os.path.exists(CRF_MODEL_PATH):
    print(f"\n❌ CRF model tidak ditemukan: {CRF_MODEL_PATH}")
    print("Jalankan train_test_clause.py terlebih dahulu!")
    exit(1)

print(f"\n✓ Loading fine-tuned BERT dari {BERT_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = IndoBERT_FineTune(MODEL_NAME, num_labels=3)
bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
bert_model = bert_model.to(device)
bert_model.eval()

print(f"✓ Loading CRF model dari {CRF_MODEL_PATH}...")
with open(CRF_MODEL_PATH, 'rb') as f:
    crf_model = pickle.load(f)

print("✓ Semua model berhasil dimuat!")

# ==============================
# FUNGSI EKSTRAKSI BERT FEATURES
# ==============================
def extract_bert_features_finetuned(tokens):
    """Ekstrak BERT features dari fine-tuned model"""
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    ).to(device)
    
    with torch.no_grad():
        features = bert_model.get_features(encoded['input_ids'], encoded['attention_mask'])
        features = features[0][:len(tokens)].cpu().numpy()
    
    return features

# ==============================
# FUNGSI DETEKSI KLAUSA
# ==============================
def detect_clauses(text):
    """Deteksi boundary klausa dari teks"""
    import re
    
    # Split by punctuation marks (., !, ?)
    sub_sentences = re.split(r'([.!?])', text.strip())
    
    all_clauses = []
    
    # Process setiap sub-sentence
    for i in range(0, len(sub_sentences), 2):
        sub_text = sub_sentences[i].strip()
        
        if not sub_text:
            continue
        
        tokens = tokenizer.tokenize(sub_text)
        
        if not tokens:
            continue
        
        # Ekstraksi features dari fine-tuned BERT
        features = extract_bert_features_finetuned(tokens)
        
        # Prediksi dengan CRF
        pred_tags = crf_model.predict_single(features)
        
        # Parse BIO tags menjadi klausa
        current_clause = []
        
        for tok, tag in zip(tokens, pred_tags):
            if tag == "B-CLAUSE":
                if current_clause:
                    all_clauses.append(" ".join(current_clause))
                current_clause = [tok]
            elif tag == "I-CLAUSE":
                current_clause.append(tok)
            elif tag == "O":
                if current_clause:
                    all_clauses.append(" ".join(current_clause))
                    current_clause = []
        
        if current_clause:
            all_clauses.append(" ".join(current_clause))
    
    return all_clauses

# ==============================
# TEST DENGAN DATASET-TEST.CSV DAN SIMPAN KE CSV
# ==============================
def extract_and_save_clauses(csv_path=TEST_CSV, output_csv="output_extracted_clauses.csv"):
    """Ekstrak klausa dari Dataset-test.csv dan simpan ke CSV baru"""
    print("\n" + "="*60)
    print(f"EKSTRAKSI KLAUSA - {csv_path}")
    print("="*60)
    
    if not os.path.exists(csv_path):
        print(f"\n❌ File tidak ditemukan: {csv_path}")
        return
    
    # Baca CSV dengan berbagai encoding
    try:
        df = pd.read_csv(csv_path, encoding='latin1')
    except:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            print(f"❌ Error membaca CSV: {e}")
            return
    
    print(f"✓ Dataset loaded: {len(df)} baris")
    
    # Cari kolom komentar
    text_col = None
    for col in df.columns:
        if col.lower() in ['komentar', 'comment', 'text', 'ulasan', 'review']:
            text_col = col
            break
    
    if text_col is None:
        text_col = df.columns[0]
    
    print(f"✓ Menggunakan kolom: {text_col}\n")
    
    # Proses semua baris dan kumpulkan hasil
    results = []
    total_clauses = 0
    
    print("Processing...")
    for idx in range(len(df)):
        text = str(df.iloc[idx][text_col]).strip()
        
        if not text or text.lower() == 'nan':
            continue
        
        # Deteksi klausa
        clauses = detect_clauses(text)
        total_clauses += len(clauses)
        
        # Simpan hanya klausa saja
        for clause in clauses:
            results.append({'klausa': clause})
        
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(df)} selesai...")
    
    # Simpan ke CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"\n✅ Ekstraksi selesai!")
    print(f"📊 Total: {total_clauses} klausa")
    print(f"✓ Hasil disimpan ke: {output_csv}")
    
    # Tampilkan preview
    print(f"\n{'='*60}")
    print("PREVIEW HASIL (10 baris pertama):")
    print(f"{'='*60}\n")
    print(result_df.head(10).to_string(index=False))

def test_dataset(csv_path=TEST_CSV, n_samples=None):
    """Test deteksi dengan Dataset-test.csv (mode interactive)"""
    print("\n" + "="*60)
    print(f"TEST DETEKSI KLAUSA - {csv_path}")
    print("="*60)
    
    if not os.path.exists(csv_path):
        print(f"\n❌ File tidak ditemukan: {csv_path}")
        return
    
    # Baca CSV dengan berbagai encoding
    try:
        df = pd.read_csv(csv_path, encoding='latin1')
    except:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            print(f"❌ Error membaca CSV: {e}")
            return
    
    print(f"✓ Dataset loaded: {len(df)} baris")
    
    # Cari kolom komentar
    text_col = None
    for col in df.columns:
        if col.lower() in ['komentar', 'comment', 'text', 'ulasan', 'review']:
            text_col = col
            break
    
    if text_col is None:
        text_col = df.columns[0]
    
    print(f"✓ Menggunakan kolom: {text_col}\n")
    
    # Tentukan jumlah sample
    if n_samples is None:
        n_samples = len(df)
    
    # Proses baris
    total_clauses = 0
    for idx in range(min(n_samples, len(df))):
        text = str(df.iloc[idx][text_col]).strip()
        
        if not text or text.lower() == 'nan':
            continue
        
        print(f"\n[Baris {idx+1}]")
        print(f"📝 Teks: {text}")
        
        # Deteksi klausa
        clauses = detect_clauses(text)
        total_clauses += len(clauses)
        
        print(f"\n✅ Klausa terdeteksi ({len(clauses)}):")
        if clauses:
            for i, clause in enumerate(clauses, 1):
                print(f"   {i}. {clause}")
        else:
            print("   (Tidak ada klausa terdeteksi)")
        
        print("\n" + "-" * 60)
    
    print(f"\n📊 Total: {total_clauses} klausa terdeteksi dari {min(n_samples, len(df))} baris")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("OPSI:")
    print("1. Ekstrak semua klausa dan simpan ke CSV")
    print("2. Test interactive (tampilkan di terminal)")
    print("="*60)
    
    choice = input("\nPilihan (1/2): ").strip()
    
    if choice == "1":
        output_file = input("Nama file output (default: output_extracted_clauses.csv): ").strip()
        if not output_file:
            output_file = "output_extracted_clauses.csv"
        extract_and_save_clauses(output_csv=output_file)
    elif choice == "2":
        print("\nOPSI TEST:")
        print("1. Test 10 baris pertama")
        print("2. Test semua data")
        print("3. Test dengan jumlah custom")
        
        test_choice = input("\nPilihan (1/2/3): ").strip()
        
        if test_choice == "1":
            test_dataset(n_samples=10)
        elif test_choice == "2":
            test_dataset()
        elif test_choice == "3":
            try:
                n = int(input("Jumlah baris: ").strip())
                test_dataset(n_samples=n)
            except ValueError:
                print("❌ Input tidak valid!")
        else:
            print("❌ Pilihan tidak valid!")
    else:
        print("❌ Pilihan tidak valid!")
