# ============================================================================
# CLAUSE BOUNDARY DETECTION (CBD) - TRAINING & TESTING
# Tujuan: Train model untuk deteksi batas klausa menggunakan:
# 1. Fine-tuned IndoBERT (ekstraksi features)
# 2. CRF (Conditional Random Field) untuk sequence tagging
# ============================================================================

# ============================================================================
# IMPORT LIBRARY
# ============================================================================
# Pandas: Data processing (baca CSV, manipulasi data)
import pandas as pd

# PyTorch: Deep learning framework
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

# Hugging Face Transformers: BERT dan tokenizer
from transformers import AutoTokenizer, AutoModel

# Scikit-learn: Train-test split dan CRF model
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF

# Utilities
import pickle      # Untuk save/load model
import numpy as np  # Numpy array operations


# ============================================================================
# KONFIGURASI GLOBAL
# ============================================================================
# Nama model BERT pre-trained dari Hugging Face
MODEL_NAME = "indobenchmark/indobert-base-p1"

# Path file dataset CSV (format: sentence_id, kalimat, token, bio_tag)
DATASET_PATH = "./clause_dataset.csv"

# Max length untuk BERT tokenization
MAX_LEN = 128

# Jumlah epoch untuk fine-tuning BERT
EPOCHS = 5

# Batch size untuk training
BATCH_SIZE = 8

# Learning rate untuk AdamW optimizer
LR = 2e-5

# Path untuk menyimpan fine-tuned BERT model
BERT_MODEL_SAVE_PATH = "indobert_finetuned_clause.pt"

# Path untuk menyimpan CRF model
CRF_MODEL_SAVE_PATH = "indobert_sklearn_crf_clause_model.pkl"

# Device: CUDA (GPU) atau CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# PRINT HEADER
# ================================================================
print(f"\n{'='*60}")
print(f"TRAINING CLAUSE DETECTION - IndoBERT (fine-tuned) + CRF")
print(f"{'='*60}")
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}\n")


# ============================================================================
# LOAD DATA DARI CSV
# ============================================================================
# Baca file CSV dengan encoding UTF-8
df = pd.read_csv(DATASET_PATH, encoding='utf-8')
print(f"✓ Dataset loaded: {len(df)} tokens")

# ================================================================
# ORGANIZE DATA: Group by sentence_id
# ================================================================
# List untuk menyimpan data per kalimat
sentences_data = []

# Group DataFrame by sentence_id
for sent_id, group in df.groupby('sentence_id'):
    # Ambil kalimat asli (sama untuk semua token dalam group)
    kalimat = group['kalimat'].iloc[0]
    
    # Ambil list token
    tokens = group['token'].tolist()
    
    # Ambil list BIO tag (B-CLAUSE, I-CLAUSE, O)
    tags = group['bio_tag'].tolist()
    
    # Simpan tuple (kalimat, tokens, tags)
    sentences_data.append((kalimat, tokens, tags))

print(f"✓ Total kalimat: {len(sentences_data)}")


# ============================================================================
# DATASET CLASS UNTUK FINE-TUNING BERT
# ============================================================================
class ClauseDataset(Dataset):
    """
    Custom PyTorch Dataset class untuk Clause Boundary Detection.
    
    Fungsi:
    - Load dan tokenize data
    - Convert BIO tags -> numeric labels
    - Return batch tensors untuk training
    """
    
    def __init__(self, data, tokenizer, max_len=128):
        """
        Constructor.
        
        Args:
            data (list): List of (kalimat, tokens, tags) tuples
            tokenizer: BERT tokenizer
            max_len (int): Max sequence length
        """
        # Filter data: hapus kalimat kosong (len(tokens) > 0)
        self.data = [d for d in data if len(d[1]) > 0]
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Mapping BIO tags ke numeric ID
        # O (Outside): 0, B-CLAUSE (Begin): 1, I-CLAUSE (Inside): 2
        self.tag2id = {"O": 0, "B-CLAUSE": 1, "I-CLAUSE": 2}

    def __len__(self):
        """Return jumlah data (samples)"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return satu sample dari dataset.
        
        Returns:
            dict dengan keys: input_ids, attention_mask, labels
        """
        # Ambil data ke-idx
        kalimat, tokens, tags = self.data[idx]
        
        # ================================================================
        # TOKENIZE: BERT tokenize tokens
        # ================================================================
        encodings = self.tokenizer(
            tokens,
            truncation=True,           # Potong jika > max_length
            padding='max_length',      # Pad ke max_length
            max_length=self.max_len,
            is_split_into_words=True,  # Token sudah split (no split lagi)
            return_tensors='pt'        # Return PyTorch tensor
        )
        
        # ================================================================
        # CONVERT TAGS: BIO tags -> numeric labels
        # ================================================================
        # Map setiap tag ke numeric ID (O->0, B-CLAUSE->1, I-CLAUSE->2)
        label_ids = [self.tag2id.get(tag, 0) for tag in tags]
        
        # Pad label ke max_len (padding dengan label 0 = "O")
        label_ids = label_ids[:self.max_len] + [0] * (self.max_len - len(label_ids))

        # ================================================================
        # RETURN: Batch dict
        # ================================================================
        return {
            'input_ids': encodings['input_ids'].squeeze(0),        # Shape: (max_len,)
            'attention_mask': encodings['attention_mask'].squeeze(0),  # Shape: (max_len,)
            'labels': torch.tensor(label_ids, dtype=torch.long)    # Shape: (max_len,)
        }


# ============================================================================
# FINE-TUNE MODEL CLASS
# ============================================================================
class IndoBERT_FineTune(nn.Module):
    """
    Custom BERT model untuk Clause Boundary Detection.
    
    Struktur:
    - BERT encoder (pre-trained)
    - Dropout (regularisasi)
    - Linear classifier (output BIO tags)
    
    Methods:
    - forward(): Forward pass dengan loss computation
    - get_features(): Extract BERT features (untuk CRF nanti)
    """
    
    def __init__(self, model_name, num_labels):
        """
        Constructor.
        
        Args:
            model_name (str): Nama model BERT pre-trained
            num_labels (int): Jumlah label output (3: O, B-CLAUSE, I-CLAUSE)
        """
        super().__init__()
        
        # Load BERT pre-trained model dari Hugging Face
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Dropout layer: 30% dropout untuk regularisasi
        self.dropout = nn.Dropout(0.3)
        
        # Linear classifier: BERT hidden state -> num_labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Cross-entropy loss: reduction='none' (per-token loss, bukan mean)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass untuk training/inference.
        
        Args:
            input_ids: Token IDs tensor
            attention_mask: Attention mask tensor
            labels: Target labels (opsional, hanya untuk training)
        
        Returns:
            - Jika labels provided: loss scalar
            - Jika labels is None: logits tensor (untuk inference)
        """
        # ================================================================
        # BERT FORWARD: Extract token-level embeddings
        # ================================================================
        # Forward ke BERT encoder
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # Ambil last hidden state: (batch_size, seq_len, hidden_size)
        # Lalu apply dropout dan classifier
        logits = self.classifier(self.dropout(outputs.last_hidden_state))
        
        # ================================================================
        # LOSS COMPUTATION (hanya jika labels provided)
        # ================================================================
        if labels is not None:
            # Get shapes
            B, L, C = logits.size()  # B=batch, L=seq_len, C=num_classes
            
            # Compute per-token cross-entropy loss
            # Reshape logits: (B*L, C), reshape labels: (B*L,)
            loss_tok = self.ce_loss(logits.view(B*L, C), labels.view(B*L))
            
            # Mask: hanya hitung loss untuk non-padding tokens
            # attention_mask=1 untuk real tokens, 0 untuk padding
            mask = attention_mask.view(B*L).float()
            
            # Weighted loss: hanya non-padding tokens berkontribusi
            loss = (loss_tok * mask).sum() / mask.sum().clamp_min(1.0)
            
            return loss
        else:
            # Jika no labels, return logits (untuk inference)
            return logits

    def get_features(self, input_ids, attention_mask):
        """
        Ekstrak BERT features (tanpa classifier layer).
        
        Digunakan untuk: Extract features untuk training CRF
        
        Args:
            input_ids: Token IDs tensor
            attention_mask: Attention mask tensor
        
        Returns:
            Tensor: BERT features shape (batch_size, seq_len, hidden_size)
        """
        # Forward ke BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # Apply dropout ke last hidden state
        features = self.dropout(outputs.last_hidden_state)
        
        return features


# ============================================================================
# STAGE 1: FINE-TUNE INDOBERT
# ============================================================================
print(f"\n{'='*60}")
print("STAGE 1: FINE-TUNING IndoBERT")
print(f"{'='*60}\n")

# ================================================================
# INITIALIZE: Tokenizer dan Model
# ================================================================
# Load tokenizer dari pre-trained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ================================================================
# SPLIT DATA: Train-test split (80-20)
# ================================================================
train_data, test_data = train_test_split(sentences_data, test_size=0.2, random_state=42)

# ================================================================
# CREATE DATASETS: PyTorch Dataset objects
# ================================================================
train_dataset = ClauseDataset(train_data, tokenizer, MAX_LEN)
test_dataset = ClauseDataset(test_data, tokenizer, MAX_LEN)

# ================================================================
# CREATE DATALOADERS: For batch training
# ================================================================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ================================================================
# INITIALIZE MODEL: Move to device
# ================================================================
model = IndoBERT_FineTune(MODEL_NAME, num_labels=3).to(device)

# ================================================================
# OPTIMIZER: AdamW
# ================================================================
optimizer = AdamW(model.parameters(), lr=LR)

# ================================================================
# PRINT STATS
# ================================================================
print(f"✓ Split - Train: {len(train_data)} | Test: {len(test_data)}")
print("Starting fine-tuning...\n")

# ================================================================
# TRAINING LOOP
# ================================================================
best_loss = float('inf')  # Track best loss

# Loop setiap epoch
for epoch in range(EPOCHS):
    # Set model ke training mode
    model.train()
    total_loss = 0
    
    # Loop setiap batch
    for batch in train_loader:
        # ================================================================
        # ZERO GRAD: Reset gradients
        # ================================================================
        optimizer.zero_grad()
        
        # ================================================================
        # PREPARE BATCH: Move to device
        # ================================================================
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # ================================================================
        # FORWARD: Compute loss
        # ================================================================
        loss = model(input_ids, attention_mask, labels)
        
        # ================================================================
        # BACKWARD: Backpropagation
        # ================================================================
        loss.backward()
        
        # ================================================================
        # UPDATE: Gradient descent step
        # ================================================================
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()

    # ================================================================
    # END OF EPOCH: Print progress
    # ================================================================
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # ================================================================
    # SAVE CHECKPOINT: Jika loss better, save model
    # ================================================================
    if avg_loss < best_loss:
        best_loss = avg_loss
        # Save state dict (model weights)
        torch.save(model.state_dict(), BERT_MODEL_SAVE_PATH)

# ================================================================
# TRAINING COMPLETE: Print summary
# ================================================================
print(f"\n✓ Fine-tuning selesai! Best loss: {best_loss:.4f}")
print(f"✓ Model saved: {BERT_MODEL_SAVE_PATH}")


# ============================================================================
# STAGE 2: EKSTRAK FEATURES DARI FINE-TUNED MODEL
# ============================================================================
print(f"\n{'='*60}")
print("STAGE 2: EKSTRAKSI FEATURES DARI FINE-TUNED BERT")
print(f"{'='*60}\n")

# ================================================================
# SET MODEL TO EVAL MODE
# ================================================================
# Set model ke evaluation mode (disable dropout, batch norm, dll)
model.eval()

# ================================================================
# DEFINE FEATURE EXTRACTION FUNCTION
# ================================================================
def extract_bert_features_finetuned(tokens):
    """
    Ekstrak BERT features dari fine-tuned model untuk token list.
    
    Args:
        tokens (list): List token (sudah di-tokenize)
    
    Returns:
        numpy.ndarray: BERT features (shape: len(tokens), hidden_size)
    """
    # ================================================================
    # TOKENIZE: BERT tokenization
    # ================================================================
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,      # Token sudah split
        return_tensors="pt",            # Return PyTorch tensor
        truncation=True,                # Potong jika > max_length
        padding='max_length',           # Pad ke max_length
        max_length=MAX_LEN
    ).to(device)
    
    # ================================================================
    # FORWARD: Extract features tanpa gradient
    # ================================================================
    with torch.no_grad():
        # Get features dari model (shape: 1, seq_len, hidden_size)
        features = model.get_features(encoded['input_ids'], encoded['attention_mask'])
        
        # Ambil batch pertama, truncate ke jumlah token asli, convert ke numpy
        features = features[0][:len(tokens)].cpu().numpy()
    
    return features

# ================================================================
# EXTRACT FEATURES: Train data
# ================================================================
print("✓ Ekstraksi BERT features dari fine-tuned model...")
X_train, y_train = [], []

# Loop setiap kalimat di train data
for kalimat, tokens, tags in train_data:
    # Extract features
    features = extract_bert_features_finetuned(tokens)
    X_train.append(features)
    y_train.append(tags)

# ================================================================
# EXTRACT FEATURES: Test data
# ================================================================
X_test, y_test = [], []

# Loop setiap kalimat di test data
for kalimat, tokens, tags in test_data:
    # Extract features
    features = extract_bert_features_finetuned(tokens)
    X_test.append(features)
    y_test.append(tags)

# ================================================================
# PRINT STATS
# ================================================================
print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")


# ============================================================================
# STAGE 3: TRAINING CRF
# ============================================================================
print(f"\n{'='*60}")
print("STAGE 3: TRAINING CRF")
print(f"{'='*60}\n")

# ================================================================
# INITIALIZE CRF MODEL
# ================================================================
# Create CRF model dengan hyperparameters
crf = CRF(
    algorithm='lbfgs',      # Algorithm: L-BFGS (optimization algorithm)
    c1=0.1,                 # Coefficient untuk L1 penalty (feature regularization)
    c2=0.1,                 # Coefficient untuk L2 penalty (feature regularization)
    max_iterations=100,     # Max iterations untuk convergence
    verbose=True            # Print progress
)

# ================================================================
# TRAIN CRF
# ================================================================
print("Training CRF model...")
# Fit CRF dengan train features (X_train) dan labels (y_train)
crf.fit(X_train, y_train)

# ================================================================
# SAVE CRF MODEL
# ================================================================
# Save model ke pickle file
with open(CRF_MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(crf, f)

print(f"\n✓ CRF Model saved: {CRF_MODEL_SAVE_PATH}")


# ============================================================================
# TESTING: Evaluasi hasil training
# ============================================================================
print(f"\n{'='*60}")
print("TESTING HASIL TRAINING")
print(f"{'='*60}\n")

# ================================================================
# TEST ON TEST SET: Show predictions vs ground truth
# ================================================================
print("📊 Test pada data test set:")

# Loop 3 sample pertama (atau kurang jika test_data < 3)
for idx in range(min(3, len(test_data))):
    # Ambil data
    kalimat, tokens, true_tags = test_data[idx]
    features = X_test[idx]
    
    # ================================================================
    # PREDICT: CRF predict tags untuk sample ini
    # ================================================================
    print(f"\n[Test {idx+1}] Kalimat: {kalimat}")
    pred_tags = crf.predict_single(features)
    
    # ================================================================
    # DISPLAY: Print token, true tag, pred tag side-by-side
    # ================================================================
    print(f"{'Token':<15} {'True':<12} {'Pred':<12}")
    print("-" * 40)
    
    # Loop setiap token
    for tok, true_tag, pred_tag in zip(tokens, true_tags, pred_tags):
        # Check jika prediksi benar
        match = "✓" if true_tag == pred_tag else "✗"
        print(f"{tok:<15} {true_tag:<12} {pred_tag:<12} {match}")


# ============================================================================
# TEST DENGAN KALIMAT BARU (Custom sentences)
# ============================================================================
print(f"\n{'='*60}")
print("📝 Test kalimat baru:")
print(f"{'='*60}\n")

# ================================================================
# HELPER FUNCTION: Extract clauses dari BIO tags
# ================================================================
def extract_clauses(tokens, tags):
    """
    Convert BIO tags -> klausa text.
    
    Args:
        tokens (list): List token
        tags (list): List BIO tags
    
    Returns:
        list: List klausa text
    """
    clauses = []
    cur = []  # Current clause accumulator
    
    # Loop setiap token dan tag
    for tok, tag in zip(tokens, tags):
        # B-CLAUSE: Beginning of new clause
        if tag == 'B-CLAUSE':
            # Jika ada clause sebelumnya, simpan
            if cur:
                clauses.append(' '.join(cur))
                cur = []
            # Mulai clause baru
            cur.append(tok)
        # I-CLAUSE: Inside clause (continuation)
        elif tag == 'I-CLAUSE':
            # Tambah token ke current clause
            cur.append(tok)
        # O: Outside (tidak ada clause)
        else:
            # Jika ada clause sebelumnya, simpan
            if cur:
                clauses.append(' '.join(cur))
                cur = []
    
    # Simpan sisa clause jika ada
    if cur:
        clauses.append(' '.join(cur))
    
    return clauses

# ================================================================
# TEST SENTENCES: Custom kalimat untuk testing
# ================================================================
test_sentences = [
    "pantai indah tetapi ombak besar",
    "hotel nyaman namun harga mahal",
    "makanan lezat dan pelayanan cepat"
]

# Loop setiap test sentence
for test_text in test_sentences:
    print(f"\nKalimat: {test_text}")
    
    # ================================================================
    # TOKENIZE: Tokenize kalimat
    # ================================================================
    test_tokens = tokenizer.tokenize(test_text)
    
    # ================================================================
    # EXTRACT FEATURES: Extract dari fine-tuned BERT
    # ================================================================
    features = extract_bert_features_finetuned(test_tokens)
    
    # ================================================================
    # PREDICT: CRF predict tags
    # ================================================================
    pred_tags = crf.predict_single(features)
    
    # ================================================================
    # DISPLAY: Print token-tag pairs
    # ================================================================
    print(f"{'Token':<15} {'BIO Tag':<12}")
    print("-" * 30)
    for tok, tag in zip(test_tokens, pred_tags):
        print(f"{tok:<15} {tag:<12}")
    
    # ================================================================
    # EXTRACT CLAUSES: Convert BIO tags -> klausa
    # ================================================================
    clauses = extract_clauses(test_tokens, pred_tags)
    print("Klausa hasil prediksi:")
    for i, c in enumerate(clauses, 1):
        print(f"   {i}. {c}")

# ================================================================
# FINAL MESSAGE
# ================================================================
print(f"\n{'='*60}")
print("✅ SELESAI!")
print(f"{'='*60}")
