import pandas as pd
import re
from transformers import pipeline

# ==========================================
# 1. CONFIG & GLOBAL VARIABLES
# ==========================================

# Load Model IndoBERT POS Tagger
print("Sedang memuat model IndoBERT POS Tagger...")
model_name = "w11wo/indonesian-roberta-base-posp-tagger"
# aggregation_strategy="simple" akan menggabungkan sub-words menjadi satu kata utuh
pos_pipeline = pipeline("token-classification", model=model_name, aggregation_strategy="simple")

# Keywords negatif untuk membantu deteksi
NEGATIVE_KEYWORDS = [
    "rusak", "kotor", "bau", "bising", "lambat", "mahal", "kasar", 
    "jelek", "bocor", "mati", "pecah", "hilang", "kurang", "panas", 
    "dingin", "berisik", "parah", "kecewa", "buruk"
]

STOP_KELUHAN = ["sangat", "agak", "terlalu", "cukup"]

# ==========================================
# 2. HELPER FUNCTIONS (Implementasi Dasar)
# ==========================================

def clean_text_user(text):
    """Membersihkan text dari karakter aneh, tapi menjaga struktur kalimat."""
    if not isinstance(text, str): return ""
    # Hapus karakter non-alphanumeric dasar tapi biarkan spasi dan titik
    text = re.sub(r'[^a-zA-Z0-9\s\.]', ' ', text)
    return text.strip()

def find_original_word(full_text, word):
    """Mencari kata asli di text (mempertahankan casing asli)."""
    match = re.search(re.escape(word), full_text, re.IGNORECASE)
    return match.group(0) if match else word

def merge_noun_phrases(tokens):
    """
    Logika sederhana menggabungkan Noun berurutan.
    Contoh: [('kamar', 'NNO'), ('mandi', 'NNO')] -> [('kamar mandi', 'NNO')]
    """
    merged = []
    i = 0
    while i < len(tokens):
        word, label = tokens[i]
        
        # Jika token saat ini NNO dan token berikutnya juga NNO, gabung
        if label == "NNO" and i + 1 < len(tokens) and tokens[i+1][1] == "NNO":
            new_word = f"{word} {tokens[i+1][0]}"
            merged.append((new_word, "NNO"))
            i += 2 # Skip next token
        else:
            merged.append((word, label))
            i += 1
    return merged

def map_model_label_to_user_label(label):
    """
    Mapping label output model w11wo ke label logic user.
    Model output: NOUN, ADJ, VERB
    User logic: NNO, ADJ, VBI, dll.
    """
    label = label.upper()
    if label in ["NOUN", "PROPN"]: return "NNO" # Anggap semua noun adalah NNO
    if label == "ADJ": return "ADJ"
    if label == "VERB": return "VBI" # Simplifikasi ke VBI
    return label

# ==========================================
# 3. USER CORE LOGIC
# ==========================================

# ==========================================
# FUNGSI BARU: PERBAIKI SUBWORD
# ==========================================
def fix_broken_subwords(pos_results, original_text):
    """
    Menggabungkan token yang terpisah. 
    Contoh: 'lant' + 'ainya' -> 'lantainya'
    Logika: Jika start_index token ini == end_index token sebelumnya, gabung.
    """
    if not pos_results: return []

    merged_tokens = []
    
    # Ambil token pertama
    current_word = pos_results[0]['word']
    current_label = pos_results[0]['entity_group']
    current_end = pos_results[0]['end']

    for i in range(1, len(pos_results)):
        token = pos_results[i]
        word = token['word']
        label = token['entity_group']
        start = token['start']
        end = token['end']

        # Cek apakah token ini nempel dengan token sebelumnya
        # (Jarak start token ini == end token sebelumnya)
        if start == current_end:
            # GABUNGKAN
            # Hapus tanda pagar (##) jika ada (khas BERT)
            clean_subword = word.replace('##', '')
            current_word += clean_subword
            
            # Update posisi akhir
            current_end = end
            
            # Opsional: Kita pertahankan label dari kata depannya (root)
            # atau jika suffix mengubah makna, bisa disesuaikan.
            # Di sini kita biarkan label lama (biasanya NNO + NNO tetap NNO)
        else:
            # SIMPAN YANG SEBELUMNYA
            merged_tokens.append({'word': current_word, 'entity_group': current_label})
            
            # RESET UNTUK KATA BARU
            current_word = word
            current_label = label
            current_end = end

    # Jangan lupa simpan token terakhir
    merged_tokens.append({'word': current_word, 'entity_group': current_label})
    
    return merged_tokens


# ==========================================
# UPDATE DI DALAM EXTRACT COMPLAINTS
# ==========================================

def extract_complaints_user_algo(raw_text):
    if not pos_pipeline: return []
    
    # 1. PREPROCESSING
    # PENTING: Jangan terlalu agresif clean text agar offset index tetap akurat
    # Kita gunakan raw_text untuk pipeline agar offset-nya benar
    
    # 2. POS Tagging
    try:
        # Kita pakai raw_text agar index start/end akurat
        hasil_pos_raw = pos_pipeline(raw_text) 
    except Exception as e:
        print(f"Error POS Tagging: {e}")
        return []

    # --- LANGKAH PERBAIKAN: GABUNGKAN SUBWORD ---
    hasil_pos = fix_broken_subwords(hasil_pos_raw, raw_text)

    tokens = []
    
    # === [DEBUG START] PRINT DETEKSI SETELAH DIGABUNG ===
    # Uncomment jika ingin melihat hasilnya
    # print(f"\n🔍 POS TAG FINAL: '{raw_text}'")
    # print(f"{'KATA':<20} | {'TAG'}")
    # print("-" * 30)
    
    for t in hasil_pos:
        word = t['word'].strip().lower() # Bersihkan spasi sisa
        # Hapus karakter aneh dari tokenisasi RoBERTa (biasanya ada karakter Ġ atau semacamnya)
        word = re.sub(r'[^a-z0-9]', '', word) 
        
        if not word: continue # Skip jika kata jadi kosong

        raw_label = t['entity_group']
        user_label = map_model_label_to_user_label(raw_label)
        
        tokens.append((word, user_label))
        # print(f"{word:<20} | {user_label}")
    
    # === [DEBUG END] ===

    # 3. Gabungkan Noun Phrases (Logika User: NNO + NNO)
    tokens = merge_noun_phrases(tokens)

    # ... (Sisa kode Logika Pencarian Pasangan sama seperti sebelumnya) ...
    # ... Copy Paste logika loop for i, (word, label) di sini ...
    
    keluhan_list = []
    for i, (word, label) in enumerate(tokens):
        if label == "NNO":
            original_nno = word # Karena raw_text dipakai, kita pakai word langsung
            
            closest_keluhan = None
            min_distance = 999

            for j, (w2, l2) in enumerate(tokens):
                if j != i:
                    is_neg_indicator = (l2 in ["ADJ", "NEG", "VBI", "VBT", "VBP"] or 
                                        w2 in NEGATIVE_KEYWORDS or 
                                        w2 == "tidak")
                    
                    if is_neg_indicator:
                        if w2 in STOP_KELUHAN: continue
                        
                        distance = abs(j - i)
                        if distance <= 4 and distance < min_distance:
                            min_distance = distance
                            if w2 == "tidak":
                                if j+1 < len(tokens):
                                    next_word = tokens[j+1][0]
                                    closest_keluhan = f"{original_nno} tidak {next_word}"
                                else:
                                    closest_keluhan = f"{original_nno} tidak jelas"
                            elif j-1 >= 0 and tokens[j-1][0] == "tidak":
                                closest_keluhan = f"{original_nno} tidak {w2}"
                            else:
                                closest_keluhan = f"{original_nno} {w2}"

            if closest_keluhan:
                keluhan_list.append(closest_keluhan)

    return list(dict.fromkeys(keluhan_list))

# ==========================================
# 4. MAIN EXECUTION & CSV TESTING
# ==========================================

if __name__ == "__main__":
    # Ganti path ini dengan lokasi file CSV kamu
    csv_path = "dataset_keluhan.csv" 
    
    # --- OPSI A: Jika file CSV belum ada, pakai data dummy ini untuk test ---
    data_dummy = {
        'review_text': [
            "dan di sayangkan sekali tempatnya kotor , banyak sampah ' yang berserakan",
            "ada wahana2 yg kalau mau main harus bayar lagi",
            "Kamar mandinya kotor dan bau tidak sedap.",
            "Wifinya mati terus, tidak bisa kerja.",
            "Sprei tempat tidur ada noda coklat.",
            "Makanannya enak tapi piringnya pecah.",
            "Staff resepsionis tidak ramah saat checkin.",
            "Air shower kecil keluarnya.",
            "Lantainya lengket belum dipel.",
            "Tidak ada masalah, semua bagus."
        ]
    }
    df = pd.DataFrame(data_dummy)
    # ----------------------------------------------------------------------

    # --- OPSI B: Uncomment baris di bawah jika ingin pakai file CSV asli ---
    # try:
    #     df = pd.read_csv(csv_path)
    #     # Pastikan nama kolom text sesuai, misal 'review', 'text', 'content'
    #     if 'text' not in df.columns and 'review_text' not in df.columns:
    #         print("Kolom text tidak ditemukan. Ganti nama kolom di script.")
    #         exit()
    # except FileNotFoundError:
    #     print("File CSV tidak ditemukan, menggunakan data dummy.")
    
    print("\n" + "="*50)
    print("HASIL TEST EKSTRAKSI KELUHAN (TOP 10)")
    print("="*50)

    # Ambil kolom text (sesuaikan nama kolomnya)
    col_name = 'review_text' if 'review_text' in df.columns else df.columns[0]
    
    # Ambil 10 baris pertama
    subset = df.head(10)

    for index, row in subset.iterrows():
        text = row[col_name]
        print(f"\n[Review #{index+1}]: {text}")
        
        keluhan = extract_complaints_user_algo(text)
        
        if keluhan:
            print(f"-> Deteksi: {keluhan}")
        else:
            print("-> Deteksi: Aman / Tidak ditemukan keluhan spesifik")

    print("\nDone.")