# 1. INSTALL LIBRARY (Jika di Colab, uncomment baris ini)

import openai
import pandas as pd
import json
from tqdm import tqdm
import time

# ==========================================
# KONFIGURASI API KEY
# ==========================================
# ⚠️ Tempel API KEY kamu di sini
""  

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# PROMPT ENGINEERING
# ==========================================
# Kita menyuruh GPT bertindak sebagai Labeler Data
SYSTEM_PROMPT = """
Kamu adalah asisten pembuat dataset NLP untuk tugas Clause Detection (Sentiment Aspect Extraction).
Tugasmu adalah membuat kalimat review restoran/wisata dalam Bahasa Indonesia yang alami, bervariasi, dan masuk akal.

Aturan BIO Tagging:
1. Kalimat terdiri dari satu atau lebih klausa (Subjek + Aspek/Sifat).
2. Tagging menggunakan skema BIO:
   - B-CLAUSE: Kata pertama dari sebuah klausa (Biasanya Subjek atau Aspek).
   - I-CLAUSE: Kata kelanjutan dalam klausa tersebut (Kata sifat, intensifier, preposisi dalam frasa).
   - O: Kata pemisah klausa (tanda baca, kata hubung seperti 'tapi', 'dan', 'karena', 'sedangkan').

Contoh Output JSON yang diharapkan:
[
  {
    "kalimat": "Pemandangan indah tapi akses jalan rusak.",
    "tokens": [
      {"word": "Pemandangan", "tag": "B-CLAUSE"},
      {"word": "indah", "tag": "I-CLAUSE"},
      {"word": "tapi", "tag": "O"},
      {"word": "akses", "tag": "B-CLAUSE"},
      {"word": "jalan", "tag": "I-CLAUSE"},
      {"word": "rusak", "tag": "I-CLAUSE"},
      {"word": ".", "tag": "O"}
    ]
  }
]

Buatlah variasi kalimat:
- Kalimat tunggal ("Makanannya enak banget.")
- Kalimat majemuk bertentangan ("Tempat bagus tapi mahal.")
- Kalimat enumerasi ("Pelayanan ramah, tempat bersih, harga oke.")
- Kalimat sebab akibat ("Saya kecewa karena toilet kotor.")
"""

# ==========================================
# FUNGSI GENERATOR
# ==========================================
def generate_batch(batch_size=10):
    """Meminta GPT membuat N kalimat sekaligus dalam format JSON"""
    
    user_content = f"Buatkan {batch_size} kalimat review variatif (positif/negatif/campuran) beserta tag BIO-nya dalam format JSON list."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Bisa ganti "gpt-4" jika ingin lebih pintar tapi lebih mahal
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.8, # Agar kreatif dan tidak repetitif
            response_format={"type": "json_object"} # Memaksa output JSON valid
        )
        
        content = response.choices[0].message.content
        # Parsing JSON string ke Python Dict
        data = json.loads(content)
        
        # GPT kadang membungkus list dalam key seperti "reviews" atau "data"
        # Kita coba ambil list-nya
        if isinstance(data, dict):
            for key in data.keys():
                if isinstance(data[key], list):
                    return data[key]
            return [] # Jika struktur aneh
        elif isinstance(data, list):
            return data
            
    except Exception as e:
        print(f"Error generating batch: {e}")
        return []

# ==========================================
# LOOP UTAMA (TARGET 500 DATA)
# ==========================================
all_data_rows = []
TOTAL_TARGET = 2500
BATCH_SIZE = 10 # Request 10 kalimat per panggilan API
start_id = 1 # ID mulai dari 3000

print(f"🚀 Memulai generate {TOTAL_TARGET} data dengan OpenAI...")

pbar = tqdm(total=TOTAL_TARGET)

while len(all_data_rows) < TOTAL_TARGET: # Logic loop diperbaiki, hitung berdasarkan row unik kalimat
    
    # Generate batch
    batch_results = generate_batch(BATCH_SIZE)
    
    if not batch_results:
        time.sleep(2) # Tunggu sebentar kalau error
        continue
        
    # Proses hasil JSON ke format CSV kita (Flattening)
    current_sentences_count = 0
    
    for item in batch_results:
        kalimat_full = item.get('kalimat', '')
        tokens_list = item.get('tokens', [])
        
        if not tokens_list: continue
        
        # Simpan per token
        for tok in tokens_list:
            word = tok['word']
            tag = tok['tag']
            
            all_data_rows.append([start_id, kalimat_full, word, tag])
            
        start_id += 1
        current_sentences_count += 1
        pbar.update(1)
        
        # Cek jika sudah mencapai target
        if pbar.n >= TOTAL_TARGET:
            break

    # Istirahat sebentar agar tidak kena Rate Limit API
    time.sleep(1)

pbar.close()

# ==========================================
# SIMPAN KE CSV
# ==========================================
df_gpt = pd.DataFrame(all_data_rows, columns=["sentence_id", "kalimat", "token", "bio_tag"])

print("\n=== CONTOH HASIL GPT ===")
print(df_gpt.head(20).to_string(index=False))

filename = "dataset_clause_GPT_500.csv"
df_gpt.to_csv(filename, index=False)
print(f"\n✅ Selesai! Dataset tersimpan di: {filename}")