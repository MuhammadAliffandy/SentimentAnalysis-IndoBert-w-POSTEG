import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import torch

# ========== STEP 1: Load Dataset ==========
df = pd.read_csv("./code/gabungan.csv")
assert 'Ulasan' in df.columns and 'Label' in df.columns
df = df.dropna(subset=['Ulasan', 'Label'])

# ===== Text Cleaning =====
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # hapus URL
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # hapus simbol
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Ulasan'] = df['Ulasan'].apply(clean_text)

# ===== Label Encoding =====
label_encoder = LabelEncoder()
df['LabelEncoded'] = label_encoder.fit_transform(df['Label'])

# ===== Balance Dataset =====
print("\nDistribusi awal label:")
print(df['LabelEncoded'].value_counts())

df_majority = df[df['LabelEncoded'] == df['LabelEncoded'].mode()[0]]
df_minority = df[df['LabelEncoded'] != df['LabelEncoded'].mode()[0]]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df = pd.concat([df_majority, df_minority_upsampled])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDistribusi setelah balancing:")
print(df['LabelEncoded'].value_counts())

# ===== Stratified Split =====
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['LabelEncoded'],
    random_state=42
)

# ========== STEP 2: IndoBERT Setup ==========
model_name = "indobenchmark/indobert-base-p1"
tokenizer = BertTokenizer.from_pretrained(model_name)

train_dataset = Dataset.from_pandas(train_df[['Ulasan', 'LabelEncoded']].rename(columns={'Ulasan': 'text', 'LabelEncoded': 'label'}))
test_dataset = Dataset.from_pandas(test_df[['Ulasan', 'LabelEncoded']].rename(columns={'Ulasan': 'text', 'LabelEncoded': 'label'}))

# Tokenisasi
def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load Model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA tersedia! Menggunakan GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA tidak tersedia. Menggunakan CPU.")
model.to(device)

# ========== STEP 3: Training Args ==========
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,           # naikkan epoch agar stabil
    learning_rate=2e-5,           # learning rate stabil untuk BERT
    weight_decay=0.01,
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=50
)

# Evaluasi Metrics
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"]
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train IndoBERT
trainer.train()

# ========== STEP 4: Evaluasi ==========
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))



# ========== STEP 3: Prediksi Review Baru ==========
new_reviews = [
    "Makanan dingin dan tidak enak.",
    "Tempatnya kotor dan bau, tidak nyaman sama sekali.",
    "Pelayanan sangat buruk, lama dan tidak ramah.",
    "Harga mahal tapi kualitas sangat jelek.",
    "Kamar hotel berisik dan AC tidak dingin.",
    "Toilet rusak dan airnya tidak mengalir.",
    "Staf tidak membantu dan cuek dengan tamu.",
    "Tempat tidur keras dan tidak bersih.",
    "Kebersihan sangat kurang, banyak debu di mana-mana.",
    "Wi-Fi lambat dan sering putus.",
    "Pesanan makanan datang sangat terlambat.",
    "Meja makan berminyak dan tidak dibersihkan.",
    "Check-in sangat lama dan membingungkan.",
    "Lampu kamar redup dan tidak nyaman untuk membaca.",
    "Kolam renang kotor dan airnya keruh.",
    "Parkiran sempit dan tidak teratur.",
    "Pelayan sering salah mengantar pesanan.",
    "Menu tidak sesuai dengan yang tertulis di aplikasi.",
    "Kamar mandi berbau tidak sedap.",
    "Kondisi hotel tidak seperti yang ada di foto.",
    "Handuk bau apek dan tidak diganti.",
    "Air panas tidak tersedia padahal sudah diminta.",
    "Sarapan sedikit dan rasanya biasa saja.",
    "Tidak ada air minum di kamar.",
    "Kunci kamar sering error dan tidak bisa dibuka.",
    "Jalan menuju lokasi rusak dan sulit diakses.",
    "Lift rusak dan harus naik tangga.",
    "AC bocor dan membuat lantai basah.",
    "Tidak ada resepsionis saat malam.",
    "Harga tidak sesuai dengan fasilitas yang diberikan."
]


# Tokenisasi dan pindahkan tensor ke device
inputs = tokenizer(new_reviews, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}  # ✅ Fix: semua input ke device

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Prediksi label
predicted_ids = torch.argmax(outputs.logits, dim=1).cpu().numpy()  # pindahkan ke CPU
predicted_labels = label_encoder.inverse_transform(predicted_ids)

# Tampilkan hasil
print("\n=== Prediksi Sentimen Ulasan Baru ===")
for review, label in zip(new_reviews, predicted_labels):
    print(f"\"{review}\" => {label}")

# ========== STEP 4: LDA pada Ulasan Negatif ==========
negative_reviews = [review for review, label in zip(new_reviews, predicted_labels) if label == "Negatif"]

# 

import pandas as pd
from collections import Counter
from transformers import pipeline

# ====== Mapping POS ke Deskripsi ======
pos_label_description = {
    "NNO": "Noun (Umum)",
    "NNP": "Noun (Nama Diri)",
    "NUM": "Numeralia",
    "PRN": "Pronoun (Kata Ganti)",
    "PRR": "Possessive Pronoun",
    "PRK": "Demonstrative Pronoun",
    "ART": "Article / Determiner",
    "VBI": "Verb (Intransitif)",
    "VBT": "Verb (Transitif)",
    "VBP": "Verb Passive",
    "VBL": "Verb Linking",
    "VBE": "Verb Existential",
    "PPO": "Preposisi",
    "ADV": "Adverbia",
    "ADK": "Konjungsi Adverbial",
    "ADJ": "Adjektiva",
    "CSN": "Konjungsi Subordinatif",
    "NEG": "Negasi",
    "CCN": "Konjungsi Koordinatif",
    "INT": "Interjeksi",
    "SC": "Subordinating Conjunction",
    "SYM": "Simbol",
    "Z": "Other",
    "X": "Unknown"
}

# Load model POS tagging
model_name = "w11wo/indonesian-roberta-base-posp-tagger"
pos_pipeline = pipeline("token-classification", model=model_name, tokenizer=model_name, aggregation_strategy="simple")

# Daftar review negatif (hasil prediksi sebelumnya)
kalimat_array = negative_reviews  # asumsi sudah tersedia

all_nno_words = []
final_data = []

stop_keluhan = {"membuat", "datang", "memberi", "memberikan", "mengambil", "menjadi"}

# kata yang sering salah label tapi negatif
badword_file = "badword.txt"
with open(badword_file, "r", encoding="utf-8") as f:
    badword_text = f.read()
negative_keywords = list({w.strip().lower() for w in badword_text.split(",") if w.strip()})

def find_original_word(text, word):
    for w in text.split():
        if word in w.lower():
            return w
    return word

for idx, kalimat in enumerate(kalimat_array, start=1):
    hasil_pos = pos_pipeline(kalimat)
    tokens = [(t['word'].strip().lower(), t['entity_group']) for t in hasil_pos]

    keluhan_pairs = []

    for i, (word, label) in enumerate(tokens):
        if label == "NNO":
            original_nno = find_original_word(kalimat, word)
            all_nno_words.append(original_nno)

            closest_keluhan = None
            min_distance = 999

            for j, (w2, l2) in enumerate(tokens):
                if j != i:
                    if l2 in ["ADJ", "NEG", "VBI", "VBT"] or w2 in negative_keywords or w2 == "tidak":
                        if w2 not in stop_keluhan:
                            distance = abs(j - i)
                            if distance <= 3 and distance < min_distance:
                                min_distance = distance
                                if w2 == "tidak":
                                    # cari kata sesudah "tidak"
                                    if j+1 < len(tokens):
                                        next_word = tokens[j+1][0]
                                        closest_keluhan = f"{original_nno} tidak {next_word}"
                                    else:
                                        closest_keluhan = f"{original_nno} tidak jelas"
                                elif j-1 >= 0 and tokens[j-1][0] == "tidak":
                                    closest_keluhan = f"{original_nno} tidak {w2}"
                                else:
                                    closest_keluhan = f"{original_nno} {w2}"

            if not closest_keluhan:
                closest_keluhan = f"{original_nno} bermasalah"

            keluhan_pairs.append(closest_keluhan)

    keluhan_pairs = list(dict.fromkeys(keluhan_pairs))
    if not keluhan_pairs:
        keluhan_pairs = ["keluhan tidak terdeteksi"]

    final_data.append({
        "No": idx,
        "Ulasan Negatif": kalimat,
        "Keluhan": ", ".join(keluhan_pairs)
    })

nno_counter = Counter(all_nno_words)
top_5_nno = nno_counter.most_common(5)

print("\n=== 5 Kata Benda (NNO) Paling Sering Muncul ===")
for kata, jumlah in top_5_nno:
    print(f"{kata} → {jumlah} kali")

df_output = pd.DataFrame(final_data)
print("\n=== Daftar Keluhan Negatif (Final Clean) ===")
print(df_output.to_string(index=False))






