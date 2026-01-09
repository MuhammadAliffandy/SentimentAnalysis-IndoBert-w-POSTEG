import pandas as pd
import openai
import json
import random
import time
from tqdm import tqdm

# ==========================================
# 1. KONFIGURASI
# ==========================================
OPENAI_API_KEY = "" 
INPUT_FILE = "dataset_v2.csv" # Nama file input
OUTPUT_FILE = "dataset_v2_balanced.csv" # Nama file output

TARGET_NON_NEGATIF = 9813
TARGET_NEGATIF = 5085

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# 2. LOAD DATA
# ==========================================
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Data berhasil dimuat: {len(df)} baris.")
except FileNotFoundError:
    # Membuat dummy data jika file tidak ada (hanya untuk contoh jalan)
    print("⚠️ File tidak ditemukan, membuat dummy data untuk simulasi...")
    data = {
        'Komentar': ['Tempat jelek'] * 1000 + ['Tempat bagus'] * 15000,
        'Label': ['Negatif'] * 1000 + ['Non-Negatif'] * 15000
    }
    df = pd.DataFrame(data)

# Cek distribusi awal
print("\n📊 Distribusi Awal:")
print(df['Label'].value_counts())

# Pisahkan Data
df_neg = df[df['Label'] == 'Negatif'].copy()
df_non = df[df['Label'] != 'Negatif'].copy() # Menganggap selain Negatif adalah Non-Negatif

# ==========================================
# 3. FILTERING (DOWNSAMPLING) NON-NEGATIF
# ==========================================
print(f"\n✂️ Mengurangi Non-Negatif menjadi {TARGET_NON_NEGATIF}...")

if len(df_non) > TARGET_NON_NEGATIF:
    # Acak dan potong
    df_non_filtered = df_non.sample(n=TARGET_NON_NEGATIF, random_state=42)
else:
    print(f"   ⚠️ Jumlah Non-Negatif ({len(df_non)}) sudah di bawah target, tidak dikurangi.")
    df_non_filtered = df_non

print(f"   ✓ Non-Negatif sekarang: {len(df_non_filtered)}")

# ==========================================
# 4. AUGMENTASI (UPSAMPLING) NEGATIF via GPT
# ==========================================
current_neg = len(df_neg)
needed_neg = TARGET_NEGATIF - current_neg

print(f"\n🚀 Memulai Augmentasi Negatif via OpenAI...")
print(f"   Data saat ini: {current_neg}")
print(f"   Target: {TARGET_NEGATIF}")
print(f"   Perlu dibuat: {needed_neg} baris baru.")

# Fungsi Generator GPT
def generate_negative_reviews(batch_size=5, existing_samples=[]): 
    # SAYA UBAH DEFAULT BATCH JADI 5 AGAR LEBIH STABIL
    
    """Meminta GPT membuat review negatif baru dengan penanganan error JSON yang lebih kuat"""
    
    # Ambil 3 contoh acak
    samples = random.sample(existing_samples, min(3, len(existing_samples)))
    sample_text = "\n".join([f"- {s}" for s in samples])

    prompt = f"""
    Buatkan {batch_size} kalimat ulasan negatif (keluhan) tentang tempat wisata di Indonesia.
    Gunakan gaya bahasa santai, kadang ada typo, atau bahasa gaul seperti contoh.
    
    PENTING:
    - Output HARUS format JSON Array murni.
    - JANGAN pakai markdown (```json).
    - JANGAN ada teks pembuka/penutup.
    
    Contoh gaya bahasa:
    {sample_text}
    
    Format Output Wajib:
    ["ulasan 1", "ulasan 2", "ulasan 3"]
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            # Kita hapus response_format json_object agar lebih fleksibel diparsing manual
        )
        
        content = response.choices[0].message.content.strip()
        
        # --- LANGKAH PEMBERSIHAN (FIX ERROR) ---
        # 1. Hapus Markdown code block jika ada
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        # 2. Coba Parsing JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Jika gagal, coba cari kurung siku [ ... ] secara manual
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                clean_content = content[start_idx:end_idx]
                data = json.loads(clean_content)
            else:
                return [] # Gagal total, skip batch ini

        # 3. Validasi Struktur Data
        if isinstance(data, dict):
            # Kadang GPT membungkus dalam key {"reviews": [...]}
            for key in data:
                if isinstance(data[key], list):
                    return data[key]
        elif isinstance(data, list):
            return data
            
        return []

    except Exception as e:
        # Error API diamkan saja biar loop lanjut terus
        return []

# Loop Utama Pembuatan Data
new_reviews = []
batch_size = 5 
existing_comments = df_neg['Komentar'].tolist()

if needed_neg > 0:
    pbar = tqdm(total=needed_neg)
    
    while len(new_reviews) < needed_neg:
        # Hitung sisa kebutuhan
        remaining = needed_neg - len(new_reviews)
        current_batch = min(batch_size, remaining)
        
        # Panggil API
        generated = generate_negative_reviews(current_batch, existing_comments)
        
        if generated:
            new_reviews.extend(generated)
            pbar.update(len(generated))
        else:
            # Jika error, tidak perlu sleep lama, langsung coba lagi
            pass 
            
    pbar.close()
    
    # Buat DataFrame baru
    df_synthetic = pd.DataFrame({
        'Komentar': new_reviews,
        'Label': ['Negatif'] * len(new_reviews)
    })
    
    # Gabung dengan data negatif asli
    df_neg_final = pd.concat([df_neg, df_synthetic])
    
else:
    print("   ✅ Data Negatif sudah cukup/lebih, tidak perlu augmentasi.")
    df_neg_final = df_neg

# ==========================================
# 5. MERGE & SAVE
# ==========================================
# Potong jika kelebihan sedikit karena batching
df_neg_final = df_neg_final.iloc[:TARGET_NEGATIF]

print(f"\n📊 Distribusi Akhir:")
print(f"   Non-Negatif : {len(df_non_filtered)}")
print(f"   Negatif     : {len(df_neg_final)}")

# Gabung Semua
df_final = pd.concat([df_non_filtered, df_neg_final]).sample(frac=1, random_state=42).reset_index(drop=True)

# Simpan
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Dataset tersimpan di: {OUTPUT_FILE}")