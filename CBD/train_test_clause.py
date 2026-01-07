# ============================================================================
# CLAUSE BOUNDARY DETECTION (CBD) - TRAINING & TESTING (WITH METRICS)
# Tujuan: Train model untuk deteksi batas klausa + Evaluasi Akurasi
# ============================================================================

# ============================================================================
# IMPORT LIBRARY
# ============================================================================
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics as crf_metrics # IMPORT UNTUK METRIK
import pickle
import numpy as np



# ============================================================================
# KONFIGURASI GLOBAL
# ============================================================================
MODEL_NAME = "indobenchmark/indobert-base-p1"
DATASET_PATH = "./dataset_bio_8k_chaos.csv" # Pastikan path file benar
MAX_LEN = 128
EPOCHS = 10           # Epoch dikurangi sedikit agar demo lebih cepat (bisa dinaikkan)
BATCH_SIZE = 16      # Batch size dinaikkan sedikit
LR = 2e-5
BERT_MODEL_SAVE_PATH = "../cbd_new/indobert_finetuned_clause.pt"
CRF_MODEL_SAVE_PATH = "../cbd_new/indobert_sklearn_crf_clause_model.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*60}")
print(f"TRAINING CLAUSE DETECTION - IndoBERT + CRF + METRICS")
print(f"{'='*60}")
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}\n")

# ============================================================================
# LOAD & PREPROCESS DATA
# ============================================================================
try:
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    print(f"✓ Dataset loaded: {len(df)} tokens")
except FileNotFoundError:
    print(f"ERROR: File {DATASET_PATH} tidak ditemukan.")
    exit()

sentences_data = []
for sent_id, group in df.groupby('sentence_id'):
    kalimat = group['kalimat'].iloc[0]
    tokens = group['token'].tolist()
    tags = group['bio_tag'].tolist()
    sentences_data.append((kalimat, tokens, tags))

print(f"✓ Total kalimat: {len(sentences_data)}")

# ============================================================================
# CLASS DEFINITIONS (Dataset & Model)
# ============================================================================
class ClauseDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = [d for d in data if len(d[1]) > 0]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = {"O": 0, "B-CLAUSE": 1, "I-CLAUSE": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        kalimat, tokens, tags = self.data[idx]
        encodings = self.tokenizer(
            tokens, truncation=True, padding='max_length',
            max_length=self.max_len, is_split_into_words=True, return_tensors='pt'
        )
        label_ids = [self.tag2id.get(tag, 0) for tag in tags]
        label_ids = label_ids[:self.max_len] + [0] * (self.max_len - len(label_ids))
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

class IndoBERT_FineTune(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.last_hidden_state))

        if labels is not None:
            B, L, C = logits.size()
            loss_tok = self.ce_loss(logits.view(B*L, C), labels.view(B*L))
            mask = attention_mask.view(B*L).float()
            loss = (loss_tok * mask).sum() / mask.sum().clamp_min(1.0)
            return loss
        else:
            return logits

    def get_features(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)

# ============================================================================
# STAGE 1: FINE-TUNE INDOBERT
# ============================================================================
print(f"\n{'='*60}")
print("STAGE 1: FINE-TUNING IndoBERT")
print(f"{'='*60}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_data, test_data = train_test_split(sentences_data, test_size=0.2, random_state=42)

train_dataset = ClauseDataset(train_data, tokenizer, MAX_LEN)
test_dataset = ClauseDataset(test_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = IndoBERT_FineTune(MODEL_NAME, num_labels=3).to(device)
optimizer = AdamW(model.parameters(), lr=LR)

print(f"✓ Split - Train: {len(train_data)} | Test: {len(test_data)}")
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['labels'].to(device)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), BERT_MODEL_SAVE_PATH)

print(f"\n✓ Fine-tuning selesai! Best loss: {best_loss:.4f}")

# ============================================================================
# STAGE 2: EKSTRAKSI FEATURES (DIPERBAIKI DENGAN SUBWORD ALIGNMENT)
# ============================================================================
print(f"\n{'='*60}")
print("STAGE 2: EKSTRAKSI FEATURES (FIXED ALIGNMENT)")
print(f"{'='*60}\n")

model.eval()

def extract_bert_features_aligned(tokens, tokenizer, model, device):
    """
    Ekstrak fitur dengan menangani Subword Tokenization IndoBERT.
    Kita mengambil embedding dari token pertama setiap kata asli.
    """
    # 1. Tokenize dengan mapping word_ids
    # is_split_into_words=True mengasumsikan input sudah list kata
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    # 2. Forward pass ke BERT
    with torch.no_grad():
        outputs = model.bert(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        # Ambil last_hidden_state: (batch, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

    # 3. Align Subwords ke Words Asli
    # word_ids() memberikan ID kata asli untuk setiap sub-token
    # Contoh: [None, 0, 1, 1, 1, 2, None] -> None=CLS/SEP, 0=Kata1, 1=Kata2(pecah 3), 2=Kata3
    word_ids = encoding.word_ids(batch_index=0)

    aligned_features = []

    # Kita hanya ambil embedding dari sub-token PERTAMA untuk setiap kata asli
    # Ini metode standar (First Subword Pooling)
    previous_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        # Skip special tokens (None)
        if word_idx is None:
            continue

        # Jika ini adalah sub-token pertama dari sebuah kata baru
        if word_idx != previous_word_idx:
            # Ambil feature vector di posisi ini
            feat = last_hidden_state[0, idx, :].cpu().numpy()
            aligned_features.append(feat)
            previous_word_idx = word_idx

    # Pastikan panjang feature sama dengan panjang token input
    # (Kadang truncation memotong token terakhir, kita potong list token asli jika perlu)
    if len(aligned_features) < len(tokens):
        # Jika BERT memotong karena max_len, kita sesuaikan
        return np.array(aligned_features), tokens[:len(aligned_features)]
    else:
        return np.array(aligned_features), tokens

# ================================================================
# EKSEKUSI ULANG EKSTRAKSI
# ================================================================
print("✓ Ekstraksi features Train & Test dengan Alignment yang Benar...")

# Train Data
X_train = []
y_train_fixed = [] # Kita buat y baru jaga-jaga kalau ada truncation
for _, toks, tags in train_data:
    feat, valid_toks = extract_bert_features_aligned(toks, tokenizer, model, device)
    X_train.append(feat)
    # Potong tags sesuai valid_toks (jika kena truncate max_len)
    y_train_fixed.append(tags[:len(valid_toks)])

# Test Data
X_test = []
y_test_fixed = []
for _, toks, tags in test_data:
    feat, valid_toks = extract_bert_features_aligned(toks, tokenizer, model, device)
    X_test.append(feat)
    y_test_fixed.append(tags[:len(valid_toks)])

print(f"✓ Selesai. Sample feature shape: {X_train[0].shape}")

# ============================================================================
# STAGE 3: TRAINING CRF (ULANG)
# ============================================================================
print(f"\n{'='*60}")
print("STAGE 3: RE-TRAINING CRF")
print(f"{'='*60}\n")

crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train_fixed)
print(f"✓ CRF Model Re-trained")

# 2. SAVE MODEL CRF (BAGIAN INI YANG DITAMBAHKAN)
with open(CRF_MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(crf, f)

print(f"✓ CRF Model saved to: {CRF_MODEL_SAVE_PATH}")

# ============================================================================
# STAGE 4: EVALUASI ULANG
# ============================================================================
print(f"\n{'='*60}")
print("STAGE 4: EVALUASI METRIK PERFORMA (REVISI)")
print(f"{'='*60}\n")

y_pred = crf.predict(X_test)
labels = list(crf.classes_) # Otomatis deteksi label yang ada

# Tampilkan Report
print(crf_metrics.flat_classification_report(
    y_test_fixed, y_pred, labels=labels, digits=4
))

# Cek apakah Label O sudah terdeteksi
print("📊 Cek Transisi 'O' (Pemisah):")
from collections import Counter
trans = Counter(crf.transition_features_)
print(f"O -> B-CLAUSE score: {trans[('O', 'B-CLAUSE')]:.4f} (Harusnya tinggi)")
print(f"I-CLAUSE -> O score: {trans[('I-CLAUSE', 'O')]:.4f} (Harusnya tinggi)")

# ============================================================================
# TEST DENGAN KALIMAT BARU (FIXED)
# ============================================================================
print(f"\n{'='*60}")
print("TEST LIVE DENGAN KALIMAT BARU")
print(f"{'='*60}\n")

import re # Import regex untuk memisahkan tanda baca dengan rapi

def extract_clauses(tokens, tags):
    clauses = []
    cur = []
    for tok, tag in zip(tokens, tags):
        if tag == 'B-CLAUSE':
            if cur: clauses.append(' '.join(cur))
            cur = [tok]
        elif tag == 'I-CLAUSE':
            cur.append(tok)
        else: # O
            if cur: clauses.append(' '.join(cur))
            cur = []
    if cur: clauses.append(' '.join(cur))
    return clauses

test_sentences = [
    "pantai indah tetapi ombak besar",
    "hotel nyaman namun harga mahal dan lokasi jauh",
    "makanan lezat, pelayanan cepat, toilet bersih"
]

for test_text in test_sentences:
    print(f"Kalimat: {test_text}")

    # 1. SPLIT KALIMAT JADI KATA (Bukan Subword)
    # Kita pakai Regex sederhana agar koma terpisah dari kata (misal: "lezat," -> "lezat", ",")
    # Ini penting karena model CBD sangat bergantung pada tanda baca.
    raw_tokens = re.findall(r'\w+|[^\w\s]', test_text)

    # 2. EKSTRAKSI FEATURES (Gunakan fungsi BARU dengan Alignment)
    # Perhatikan argumennya sekarang butuh: tokenizer, model, device
    features, valid_tokens = extract_bert_features_aligned(raw_tokens, tokenizer, model, device)

    # 3. PREDIKSI
    pred_tags = crf.predict_single(features)

    # 4. TAMPILKAN HASIL
    clauses = extract_clauses(valid_tokens, pred_tags)

    # Print per token untuk debug
    # print(list(zip(valid_tokens, pred_tags)))

    for i, c in enumerate(clauses, 1):
        print(f"   ► Klausa {i}: {c}")
    print("-" * 40)

print(f"\n{'='*60}")
print("✅ SELESAI!")
print(f"{'='*60}")