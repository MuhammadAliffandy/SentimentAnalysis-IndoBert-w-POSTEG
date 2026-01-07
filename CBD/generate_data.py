import pandas as pd
import random
from tqdm import tqdm

# ==========================================
# 1. KAMUS "KOTOR" (SLANG & VARIATION)
# ==========================================
slang_dict = {
    "tempat": ["tmpt", "t4", "lokasi", "spot", "area"],
    "pemandangan": ["view", "pmndangan", "pemandangannya", "suasana"],
    "toilet": ["wc", "kamar mandi", "toiletnya", "kamar kecil"],
    "harga": ["harganya", "hrg", "biaya", "tarif"],
    "makanan": ["mknan", "menu", "kuliner", "jajanan"],
    "pelayanan": ["service", "pelayanannya", "staf", "karyawan"],
    "akses": ["jalan", "jln", "rute", "medan", "track"],
    "bagus": ["bgs", "mantap", "kece", "oke", "good", "jos"],
    "indah": ["cakep", "cantik", "memukau", "adem"],
    "bersih": ["bersi", "clink", "rapi", "terawat"],
    "kotor": ["ktr", "jorok", "bau", "kumuh", "berantakan"],
    "mahal": ["pricey", "mihil", "kemahalan", "ngotak"],
    "murah": ["mrh", "murmer", "terjangkau", "receh"],
    "ramah": ["friendly", "sopan", "baik", "sigap"],
    "enak": ["yummy", "sedap", "mantul", "maknyus"],
    "sangat": ["bgt", "banget", "sungguh", "super", "bener2"],
    "sekali": ["bgt", "pisan", "bet", "kali"],
    "tapi": ["tp", "tpi", "cuman", "cmn", "sayangnya", "hanya saja"],
    "dan": ["dn", "&", "plus", "serta", "sama"],
    "karena": ["krn", "soalnya", "gegara", "sebab"],
    "yang": ["yg", "sg", "nu"],
    "dengan": ["dgn", "dg", "sama"],
    "tidak": ["ga", "gak", "gk", "engga", "ndak"],
    "ke": ["k", "ke"],
    "di": ["d", "di"]
}

# ==========================================
# 2. GUDANG KOSAKATA (DIPERLUAS DENGAN FRASA)
# ==========================================
# PENTING: Kita masukkan frasa panjang agar 'ke', 'di', 'yang' 
# masuk sebagai I-CLAUSE, bukan O.

subjects_single = [
    "tempat", "lokasi", "view", "pemandangan", "suasana", "udara", 
    "toilet", "wc", "kamar mandi", "parkiran", "gazebo", "staf", "pelayanan",
    "kolam renang", "wahana", "air", "listrik", "sinyal"
]

# Frasa Subjek (Agar model belajar 'ke/di' adalah bagian dari klausa)
subjects_compound = [
    "akses ke lokasi", "jalan menuju tempat", "jalan ke sana", 
    "di dalam area", "pas masuk gerbang", "saat di parkiran",
    "tiket masuknya", "harga makanannya", "fasilitas di sini",
    "orang di sana", "petugas di loket", "masuk ke dalam",
    "pemandangan di sekitar", "suasana di malam hari"
]

adj_pos = [
    "indah", "bagus", "keren", "cakep", "memukau", "estetik", 
    "bersih", "wangi", "terawat", "higienis",
    "murah", "terjangkau", "worth it",
    "ramah", "sigap", "sopan", "cepat",
    "enak", "sedap", "mantap", "segar",
    "luas", "mudah", "aman", "nyaman", "sejuk"
]

adj_neg = [
    "kotor", "bau", "pesing", "jorok", "kumuh", 
    "mahal", "kemahalan", "overprice", 
    "jutek", "galak", "lelet", "lambat", "kasar",
    "hambar", "asin", "biasa aja",
    "sempit", "sesak", "macet", "rusak", "gelap", "panas", "gersang"
]

# Kata sambung PEMECAH KLAUSA (Tag O)
connectors_split = ["tapi", "tetapi", "namun", "sayangnya", "cuma", "padahal", 
                    "karena", "sebab", "soalnya", "meskipun", "walau"]

# Kata sambung PENYATU (Tag I-CLAUSE untuk Enumerasi)
connectors_join = ["dan", "serta", "juga", "ditambah", "&", "plus"]

intensifiers = ["sangat", "cukup", "lumayan", "banget", "parah", "abis", "bener-bener"]

# ==========================================
# 3. FUNGSI UTILS
# ==========================================
def corrupt_text(word):
    """Mengubah kata baku menjadi slang secara acak"""
    word_lower = word.lower()
    if word_lower in slang_dict and random.random() < 0.6: # 60% chance slang
        return random.choice(slang_dict[word_lower])
    return word

# ==========================================
# 4. LOGIKA PEMBUATAN KLAUSA (REVISI)
# ==========================================

def create_single_clause(sentiment="random", mode="standard"):
    if sentiment == "random": sentiment = random.choice(["pos", "neg"])
    
    # Pilih Subjek (50% kata tunggal, 50% frasa panjang)
    if random.random() < 0.5:
        s_raw = random.choice(subjects_single)
    else:
        s_raw = random.choice(subjects_compound)
        
    adj = random.choice(adj_pos if sentiment == "pos" else adj_neg)
    intense = random.choice(intensifiers) if random.random() > 0.5 else ""
    particle = random.choice(["sih", "memang", "yg", "yang", ""]) if random.random() > 0.6 else ""

    # Variasi Pola Kalimat
    # Mode 'enumeration' = untuk kalimat "enak, murah, dan bersih"
    if mode == "enumeration":
        words_raw = [adj]
        if intense: words_raw.append(intense)
    else:
        # Pola Standar
        structure = random.choice(["S+A", "S+A+I", "S+P+A", "A+S"])
        
        words_raw = []
        if structure == "S+A": words_raw = [s_raw, adj]
        elif structure == "S+A+I": words_raw = [s_raw, adj, intense]
        elif structure == "S+P+A": words_raw = [s_raw, particle, adj] if particle else [s_raw, adj]
        elif structure == "A+S": words_raw = [adj, s_raw]

    # Tokenisasi & Tagging
    final_tokens = []
    bio_tags = []
    
    # Flattening: "akses ke lokasi" -> ["akses", "ke", "lokasi"]
    first_token = True
    for raw_w in words_raw:
        if not raw_w: continue
        
        # Split frasa menjadi kata per kata
        sub_words = raw_w.split()
        for sub_w in sub_words:
            # Corrupt (slang/typo)
            clean_w = corrupt_text(sub_w)
            final_tokens.append(clean_w)
            
            # LOGIKA TAGGING BIO:
            # Token pertama dari seluruh klausa = B-CLAUSE
            # Token sisanya (termasuk preposisi di dalam frasa) = I-CLAUSE
            if first_token and mode != "continuation":
                bio_tags.append("B-CLAUSE")
                first_token = False
            else:
                bio_tags.append("I-CLAUSE")
                
    return list(zip(final_tokens, bio_tags))

# ==========================================
# 5. GENERATOR DATASET UTAMA
# ==========================================
def generate_robust_data(target_count=8000):
    dataset = []
    print(f"Generate {target_count} data (FIXED PREPOSITIONS & ENUMERATION)...")
    
    for i in tqdm(range(target_count)):
        sentence_id = i
        
        # Pilih Tipe Struktur Kalimat
        pola = random.choices(
            ["simple", "contrast", "cause", "enumeration", "comma_sep"],
            weights=[0.1, 0.4, 0.1, 0.2, 0.2], 
            k=1
        )[0]
        
        parts = []
        
        # 1. SIMPLE: "Tempatnya bagus banget"
        if pola == "simple":
            parts.append(create_single_clause())
            
        # 2. CONTRAST: "Tempat bagus TAPI mahal" (Ada Splitter 'O')
        elif pola == "contrast":
            parts.append(create_single_clause())
            conn = random.choice(connectors_split)
            parts.append([(corrupt_text(conn), "O")]) # Splitter = O
            parts.append(create_single_clause())
            
        # 3. CAUSE: "Macet PARAH SOALNYA jalannya sempit"
        elif pola == "cause":
            parts.append(create_single_clause())
            conn = random.choice(["karena", "soalnya", "gegara"])
            parts.append([(corrupt_text(conn), "O")])
            parts.append(create_single_clause())
            
        # 4. ENUMERATION (PENTING!!): "Murah, enak, dan ramah"
        # Ini mengajarkan model bahwa KOMA bisa jadi I-CLAUSE (bukan pemisah klausa)
        elif pola == "enumeration":
            # Klausa Induk: "Makanannya murah"
            clause1 = create_single_clause() 
            parts.append(clause1)
            
            # Koma di dalam klausa (I-CLAUSE)
            parts.append([(",", "I-CLAUSE")]) 
            
            # Lanjutan Sifat (tanpa subjek baru, mode continuation)
            clause2 = create_single_clause(mode="enumeration") 
            # Paksa tagnya jadi I-CLAUSE semua karena ini kelanjutan
            clause2 = [(w, "I-CLAUSE") for w, t in clause2]
            parts.append(clause2)
            
            # Dan/Serta (I-CLAUSE)
            conn = random.choice(connectors_join)
            parts.append([(corrupt_text(conn), "I-CLAUSE")])
            
            # Sifat terakhir
            clause3 = create_single_clause(mode="enumeration")
            clause3 = [(w, "I-CLAUSE") for w, t in clause3]
            parts.append(clause3)

        # 5. COMMA SEPARATOR: "Tempat bagus, toilet bersih" 
        # (Koma sebagai pemisah antar klausa independen -> O)
        elif pola == "comma_sep":
            parts.append(create_single_clause())
            parts.append([(",", "O")]) # Disini Koma jadi O karena memisahkan Subjek baru
            parts.append(create_single_clause())

        # --- COMPILE DATASET ---
        full_sent_tokens = []
        
        # Flatten parts
        flat_sentence = [item for sublist in parts for item in sublist]
        
        # Masukkan ke list dataset
        for token, tag in flat_sentence:
            # Format: [ID, Kalimat_Full(Nanti), Token, Tag]
            dataset.append([sentence_id, None, token, tag])
            full_sent_tokens.append(token)
            
        # Isi kolom Kalimat Full (untuk referensi manusia)
        full_str = " ".join(full_sent_tokens).replace(" ,", ",")
        
        # Update baris-baris yg baru dimasukkan dengan kalimat full
        start_idx = len(dataset) - len(flat_sentence)
        for idx in range(start_idx, len(dataset)):
            dataset[idx][1] = full_str

    df = pd.DataFrame(dataset, columns=["sentence_id", "kalimat", "token", "bio_tag"])
    return df

# ==========================================
# EKSEKUSI
# ==========================================
df_result = generate_robust_data(8000)

print("\n=== PREVIEW DATASET BARU ===")
# Tampilkan contoh Enumerasi (yang dulu sering salah)
print(df_result[df_result['kalimat'].str.contains(",", na=False)].head(15).to_string(index=False))

filename = "dataset_clause_FIXED_v3.csv"
df_result.to_csv(filename, index=False)
print(f"\n✓ Dataset tersimpan di: {filename}")
print("✓ Langkah selanjutnya: Gunakan file ini untuk Training Ulang BERT + CRF.")