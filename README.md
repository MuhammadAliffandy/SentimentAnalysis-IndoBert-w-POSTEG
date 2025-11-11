# Analisis Sentimen dan Ekstraksi Keluhan

Proyek ini melakukan analisis sentimen menggunakan IndoBERT dan ekstraksi keluhan dari review negatif menggunakan POS Tagging.

## Struktur Proyek

```
website-resifa/
├── main.py                    # Script utama
├── config.py                  # Konfigurasi
├── data_preprocessing.py      # Modul preprocessing data
├── sentiment_analysis.py      # Modul analisis sentimen
├── complaint_extraction.py    # Modul ekstraksi keluhan
├── requirements.txt           # Dependencies
├── Dataset.csv               # Dataset training
├── badword.txt               # Kata-kata negatif
└── README.md                 # Dokumentasi ini
```

## Instalasi

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Stanza model (opsional, untuk POS tagging alternatif):**
   ```bash
   python main.py --mode setup-stanza
   ```

## Cara Penggunaan

### Mode Interactive (Mudah)
Jalankan script utama tanpa argumen:
```bash
python main.py
```
Kemudian pilih mode yang diinginkan dari menu.

### Mode Command Line

1. **Full Pipeline (Training + Prediksi + Ekstraksi):**
   ```bash
   python main.py --mode full
   ```

2. **Training Model Saja:**
   ```bash
   python main.py --mode train
   ```

3. **Prediksi Saja (model sudah ada):**
   ```bash
   python main.py --mode predict
   ```

4. **Ekstraksi Keluhan Saja:**
   ```bash
   python main.py --mode extract
   ```

5. **Testing dengan Sample Data:**
   ```bash
   python main.py --mode full --sample
   ```

### Opsi Tambahan

- `--use-stanza`: Gunakan Stanza POS Tagger instead of Transformers
- `--no-save`: Jangan simpan model setelah training
- `--sample`: Gunakan sample data untuk testing

## File yang Diperlukan

1. **Dataset.csv**: File dataset dengan kolom 'Komentar' dan 'Label'
2. **badword.txt**: File berisi kata-kata negatif dipisahkan koma

## Output

- Model tersimpan di folder `./results/` dan `./saved_model/`
- Hasil ekstraksi keluhan disimpan dalam file CSV
- Log training tersimpan di folder `./logs/`

## Fitur Utama

### 1. Data Preprocessing
- Pembersihan teks (case folding, removal URL, punctuation)
- Label encoding
- Dataset balancing
- Train-test split dengan stratifikasi

### 2. Sentiment Analysis
- Menggunakan IndoBERT (indobenchmark/indobert-base-p1)
- Training dengan Transformers Trainer
- Evaluasi dengan metrics (accuracy, F1-score)
- Support GPU/CUDA jika tersedia

### 3. Complaint Extraction
- POS Tagging untuk identifikasi kata benda dan kata sifat
- Algoritma pencarian pasangan keluhan terdekat
- Support 2 POS Tagger: Transformers dan Stanza
- Filtering kata stop dan normalisasi

## Contoh Penggunaan Individual Module

### Data Preprocessing
```python
from data_preprocessing import preprocess_data

train_df, test_df, label_encoder = preprocess_data()
```

### Sentiment Analysis
```python
from sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.setup_model(label_encoder)
predictions = analyzer.predict_texts(["Review negatif"])
```

### Complaint Extraction
```python
from complaint_extraction import ComplaintExtractor

extractor = ComplaintExtractor()
df_result, top_nouns = extractor.generate_report(negative_reviews)
```

## Troubleshooting

1. **Error loading dataset:** Pastikan file `Dataset.csv` ada dan memiliki kolom 'Komentar' dan 'Label'

2. **CUDA/GPU issues:** Script akan otomatis fallback ke CPU jika GPU tidak tersedia

3. **Stanza model tidak ditemukan:** Jalankan `python main.py --mode setup-stanza` terlebih dahulu

4. **Memory issues:** Kurangi batch size di `config.py` dalam `TRAINING_ARGS`

## Konfigurasi

Edit file `config.py` untuk mengubah:
- Model yang digunakan
- Parameter training
- Path file dataset
- Parameter ekstraksi keluhan

## Dependencies

- pandas
- scikit-learn
- datasets
- transformers
- torch
- matplotlib
- numpy
- nltk
- stanza (opsional)
- spacy (opsional)

## Catatan

- Project ini dioptimalkan untuk bahasa Indonesia
- Membutuhkan koneksi internet untuk download model pertama kali
- Proses training bisa memakan waktu lama tergantung hardware
- Hasil terbaik dengan dataset yang cukup besar dan balanced
