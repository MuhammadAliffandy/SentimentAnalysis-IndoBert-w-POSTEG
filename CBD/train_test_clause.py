import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
import pickle
import numpy as np

# ==============================
# KONFIGURASI
# ==============================
MODEL_NAME = "indobenchmark/indobert-base-p1"
DATASET_PATH = "./clause_dataset.csv"
MAX_LEN = 128
EPOCHS = 5
BATCH_SIZE = 8
LR = 2e-5

BERT_MODEL_SAVE_PATH = "indobert_finetuned_clause.pt"
CRF_MODEL_SAVE_PATH = "indobert_sklearn_crf_clause_model.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*60}")
print(f"TRAINING CLAUSE DETECTION - IndoBERT (fine-tuned) + CRF")
print(f"{'='*60}")
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}\n")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATASET_PATH, encoding='utf-8')
print(f"✓ Dataset loaded: {len(df)} tokens")

sentences_data = []
for sent_id, group in df.groupby('sentence_id'):
    kalimat = group['kalimat'].iloc[0]
    tokens = group['token'].tolist()
    tags = group['bio_tag'].tolist()
    sentences_data.append((kalimat, tokens, tags))

print(f"✓ Total kalimat: {len(sentences_data)}")

# ==============================
# DATASET CLASS UNTUK FINE-TUNING
# ==============================
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
            tokens,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            is_split_into_words=True,
            return_tensors='pt'
        )
        label_ids = [self.tag2id.get(tag, 0) for tag in tags]
        label_ids = label_ids[:self.max_len] + [0] * (self.max_len - len(label_ids))

        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

# ==============================
# FINE-TUNE MODEL CLASS
# ==============================
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
        """Ekstrak BERT features tanpa classifier"""
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        features = self.dropout(outputs.last_hidden_state)
        return features

# ==============================
# STAGE 1: FINE-TUNE INDOBERT
# ==============================
print(f"\n{'='*60}")
print("STAGE 1: FINE-TUNING IndoBERT")
print(f"{'='*60}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_data, test_data = train_test_split(sentences_data, test_size=0.2, random_state=42)

train_dataset = ClauseDataset(train_data, tokenizer, MAX_LEN)
test_dataset = ClauseDataset(test_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = IndoBERT_FineTune(MODEL_NAME, num_labels=3).to(device)
optimizer = AdamW(model.parameters(), lr=LR)

print(f"✓ Split - Train: {len(train_data)} | Test: {len(test_data)}")
print("Starting fine-tuning...\n")

best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), BERT_MODEL_SAVE_PATH)

print(f"\n✓ Fine-tuning selesai! Best loss: {best_loss:.4f}")
print(f"✓ Model saved: {BERT_MODEL_SAVE_PATH}")

# ==============================
# STAGE 2: EKSTRAK FEATURES DARI FINE-TUNED MODEL
# ==============================
print(f"\n{'='*60}")
print("STAGE 2: EKSTRAKSI FEATURES DARI FINE-TUNED BERT")
print(f"{'='*60}\n")

model.eval()

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
        features = model.get_features(encoded['input_ids'], encoded['attention_mask'])
        features = features[0][:len(tokens)].cpu().numpy()
    
    return features

print("✓ Ekstraksi BERT features dari fine-tuned model...")
X_train, y_train = [], []
for kalimat, tokens, tags in train_data:
    features = extract_bert_features_finetuned(tokens)
    X_train.append(features)
    y_train.append(tags)

X_test, y_test = [], []
for kalimat, tokens, tags in test_data:
    features = extract_bert_features_finetuned(tokens)
    X_test.append(features)
    y_test.append(tags)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")

# ==============================
# STAGE 3: TRAINING CRF
# ==============================
print(f"\n{'='*60}")
print("STAGE 3: TRAINING CRF")
print(f"{'='*60}\n")

crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    verbose=True
)

print("Training CRF model...")
crf.fit(X_train, y_train)

# Save model
with open(CRF_MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(crf, f)

print(f"\n✓ CRF Model saved: {CRF_MODEL_SAVE_PATH}")

# ==============================
# TESTING
# ==============================
print(f"\n{'='*60}")
print("TESTING HASIL TRAINING")
print(f"{'='*60}\n")

print("📊 Test pada data test set:")

for idx in range(min(3, len(test_data))):
    kalimat, tokens, true_tags = test_data[idx]
    features = X_test[idx]
    
    print(f"\n[Test {idx+1}] Kalimat: {kalimat}")
    pred_tags = crf.predict_single(features)
    
    print(f"{'Token':<15} {'True':<12} {'Pred':<12}")
    print("-" * 40)
    for tok, true_tag, pred_tag in zip(tokens, true_tags, pred_tags):
        match = "✓" if true_tag == pred_tag else "✗"
        print(f"{tok:<15} {true_tag:<12} {pred_tag:<12} {match}")

# ==============================
# Test kalimat baru
# ==============================
print(f"\n{'='*60}")
print("📝 Test kalimat baru:")
print(f"{'='*60}\n")

def extract_clauses(tokens, tags):
    clauses = []
    cur = []
    for tok, tag in zip(tokens, tags):
        if tag == 'B-CLAUSE':
            if cur:
                clauses.append(' '.join(cur))
                cur = []
            cur.append(tok)
        elif tag == 'I-CLAUSE':
            cur.append(tok)
        else:
            if cur:
                clauses.append(' '.join(cur))
                cur = []
    if cur:
        clauses.append(' '.join(cur))
    return clauses

test_sentences = [
    "pantai indah tetapi ombak besar",
    "hotel nyaman namun harga mahal",
    "makanan lezat dan pelayanan cepat"
]

for test_text in test_sentences:
    print(f"\nKalimat: {test_text}")
    test_tokens = tokenizer.tokenize(test_text)
    features = extract_bert_features_finetuned(test_tokens)
    pred_tags = crf.predict_single(features)
    
    print(f"{'Token':<15} {'BIO Tag':<12}")
    print("-" * 30)
    for tok, tag in zip(test_tokens, pred_tags):
        print(f"{tok:<15} {tag:<12}")
    
    clauses = extract_clauses(test_tokens, pred_tags)
    print("Klausa hasil prediksi:")
    for i, c in enumerate(clauses, 1):
        print(f"   {i}. {c}")

print(f"\n{'='*60}")
print("✅ SELESAI!")
print(f"{'='*60}")
