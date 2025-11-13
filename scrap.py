# ============================================================================
# SCRAPER ULASAN GOOGLE MAPS UNTUK TEMPAT WISATA JEMBER
# Tujuan: Mengambil ulasan dari Google Maps menggunakan Selenium WebDriver
# ============================================================================

# IMPORT LIBRARY
# Library Selenium untuk otomasi browser Chrome
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Library untuk auto-download ChromeDriver sesuai versi Chrome yang terinstall
from webdriver_manager.chrome import ChromeDriverManager

# Library standar Python
import time      # Untuk pause/delay script
import json      # Untuk menyimpan data ke format JSON
import sys       # Untuk exit program
import os        # Untuk operasi folder/file


# ============================================================================
# DATA: DAFTAR TEMPAT WISATA JEMBER YANG AKAN DI-SCRAPE
# ============================================================================
data = [
  "Pantai Papuma Jember",
    "Puncak Rembangan Jember",
    "Waterboom Tiara Park Jember",
    "Pantai Cemara Jember",
    "Taman Botani Sukorambi Jember",
    "Pantai Watu Ulo Jember",
    "Dira Park Jember",
    "Taman Galaxy Jember",
    "Pantai Payangan Jember",
    "Jember Mini Zoo",
    "Kebun Teh Gunung Gambir Jember",
    "Agrowisata Gunung Pasang Jember",
    "Teluk Love Jember",
    "Mumbul Garden Jember",
    "Taman Nasional Meru Betiri Jember",
    "Air Terjun Tancak Jember",
    "Pantai Bandealit Jember",
    "Pantai Nanggelan Jember",
    "Air Terjun Antrokan Manggisan Jember",
    "Bukit Samboja Jember"
]

# ============================================================================
# KONFIGURASI CHROME BROWSER
# ============================================================================
# Buat objek Options untuk setting Chrome
options = Options()

# --headless: Jalankan Chrome tanpa tampilan GUI (lebih cepat, cocok untuk server)
options.add_argument("--headless")

# --disable-gpu: Matikan GPU rendering untuk kompatibilitas lebih baik
options.add_argument("--disable-gpu")

# --disable-software-rasterizer: Hindari fallback ke software rendering (stabilitas)
options.add_argument("--disable-software-rasterizer")

# --use-gl=swiftshader: Gunakan SwiftShader untuk rendering (headless mode)
options.add_argument("--use-gl=swiftshader")

# --enable-unsafe-webgl: Aktifkan WebGL jika dibutuhkan di headless mode
options.add_argument("--enable-unsafe-webgl")

# ChromeDriverManager otomatis download ChromeDriver yang sesuai dengan versi Chrome
service = Service(ChromeDriverManager().install())

# ============================================================================
# SETUP FOLDER OUTPUT
# ============================================================================
# Buat folder "data" jika belum ada untuk menyimpan file JSON hasil scraping
os.makedirs("data", exist_ok=True)

# ============================================================================
# MAIN SCRAPING LOGIC
# ============================================================================
try:
    # LOOP 1: Iterasi setiap tempat dalam daftar data
    for i in data:
        # Inisialisasi Chrome WebDriver dengan service dan options yang sudah dikonfigurasi
        driver = webdriver.Chrome(service=service, options=options)
        
        # Buka halaman Google Maps
        driver.get("https://www.google.com/maps")
        print(f"Google Maps dibuka untuk: {i}")
        
        # Tunggu 3 detik untuk halaman loading
        time.sleep(3)
        
        # ====================================================================
        # LANGKAH 1: PENCARIAN TEMPAT DI GOOGLE MAPS
        # ====================================================================
        search_query = i  # Nama tempat yang akan dicari
        
        # Cari elemen search box berdasarkan ID "searchboxinput"
        search_box = driver.find_element(By.ID, "searchboxinput")
        
        # Ketik nama tempat di search box
        search_box.send_keys(search_query)
        
        # Tekan tombol ENTER untuk memulai pencarian
        search_box.send_keys(Keys.ENTER)
        
        # ====================================================================
        # LANGKAH 2: KLIK HASIL PENCARIAN PERTAMA
        # ====================================================================
        try:
            # Tunggu hingga hasil pencarian muncul (maksimal 10 detik)
            # XPath mencari link di dalam div yang memiliki atribut aria-label berisi "Hasil untuk"
            first_result = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@aria-label, 'Hasil untuk')]//a"))
            )
            
            # Klik link hasil pencarian pertama (tempat dengan judul cocok)
            first_result.click()
            
            # Tunggu 5 detik untuk halaman detail tempat loading
            time.sleep(5)
        except Exception as e:
            # Jika gagal menemukan/klik hasil, cetak error dan lanjut
            print("Gagal menekan hasil pencarian:", e)
        
        # ====================================================================
        # LANGKAH 3: BUKA TAB ULASAN
        # ====================================================================
        try:
            # Cari dan tunggu tombol "ulasan" hingga bisa diklik (maksimal 10 detik)
            # XPath mencari button dengan aria-label berisi "ulasan"
            reviews_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'ulasan')]"))
            )
            
            # Klik tombol ulasan untuk membuka panel ulasan
            reviews_button.click()
            
            # Tunggu 3 detik untuk panel ulasan loading
            time.sleep(3)
        except Exception as e:
            # Jika tidak ada tombol ulasan, cetak error, tutup browser, dan lanjut ke tempat berikutnya
            print("Tidak bisa menemukan tombol ulasan:", e)
            driver.quit()
            continue
        
        # ====================================================================
        # LANGKAH 4: BUKA ULASAN LEBIH BANYAK (OPSIONAL)
        # ====================================================================
        try:
            # Cari tombol "Ulasan lainnya" untuk membuka semua ulasan
            more_reviews_button = driver.find_element(By.XPATH, "//span[contains(text(), 'Ulasan lainnya')]")
            
            # Klik tombol jika ditemukan
            more_reviews_button.click()
            
            # Tunggu 3 detik untuk halaman loading
            time.sleep(3)
        except:
            # Jika tombol tidak ada (berarti ulasan sudah lengkap), lanjut ke scraping
            print("Tidak ada tombol 'Ulasan Lainnya', lanjut ke scraping.")
        
        # ====================================================================
        # LANGKAH 5: SCROLL PANEL ULASAN UNTUK LOAD SEMUA DATA
        # ====================================================================
        # Tunggu elemen container ulasan muncul (CSS selector untuk div yang berisi ulasan)
        # Class ".m6QErb.kA9KIf" adalah container scrollable untuk ulasan
        scrollable_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".m6QErb.kA9KIf"))
        )
        
        # Ambil tinggi scroll awal (untuk mendeteksi kapan loading selesai)
        last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
        
        # Counter untuk menampilkan progress scraping
        num = 1
        
        # Loop scroll: terus scroll sampai tidak ada ulasan baru yang dimuat
        while True:
            # Tampilkan progress dengan bar "=" yang bertambah
            print('scrapping process =' + f'{"="*num}')
            
            # Scroll ke bawah container ulasan (ke tinggi maksimal)
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
            
            # Tunggu 3 detik untuk ulasan baru dimuat
            time.sleep(3)
            
            # Ambil tinggi scroll baru setelah loading
            new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
            
            # Jika tinggi tidak berubah, berarti sudah mencapai akhir (semua ulasan sudah dimuat)
            if new_height == last_height:
                break
            
            # Update tinggi untuk perulangan berikutnya
            last_height = new_height
            
            # Tambah counter progress
            num += 1
        
        # ====================================================================
        # LANGKAH 6: EKSTRAK TEKS ULASAN DARI ELEMEN DOM
        # ====================================================================
        # Cari semua elemen ulasan berdasarkan class name "wiI7pd"
        # Class ini adalah wrapper untuk setiap ulasan individual
        review_elements = driver.find_elements(By.CLASS_NAME, "wiI7pd")
        
        # List untuk menyimpan ulasan yang sudah diekstrak
        reviews = []
        
        # Loop setiap elemen ulasan
        for review in review_elements:
            try:
                # Ambil teks dari elemen dan hapus whitespace di awal/akhir
                text = review.text.strip()
                
                # Hanya simpan jika teks tidak kosong
                if text:
                    reviews.append(text)
            except:
                # Jika gagal ekstrak teks, skip elemen ini dan lanjut
                continue
        
        # ====================================================================
        # LANGKAH 7: SIMPAN HASIL ULASAN KE FILE JSON
        # ====================================================================
        # Buat nama file dari nama tempat (ganti spasi dengan underscore, jadikan lowercase)
        filename = f"./data/{i.replace(' ','_').lower()}.json"
        
        # Buka file untuk ditulis (mode "w"), dengan encoding UTF-8 untuk mendukung karakter Indonesia
        with open(filename, "w", encoding="utf-8") as f:
            # Dump ulasan ke format JSON dengan indentasi 4 spasi
            # ensure_ascii=False: pertahankan karakter non-ASCII (agar karakter Indonesia tetap utuh)
            json.dump({"reviews": reviews}, f, indent=4, ensure_ascii=False)
        
        # Cetak notifikasi bahwa data sudah disimpan
        print(f"Hasil {i} disimpan ke {filename}")
        
        # Tutup browser untuk tempat ini (siap membuka tempat berikutnya)
        driver.quit()

# ====================================================================
# ERROR HANDLING
# ====================================================================
except Exception as e:
    # Tangkap dan cetak error apapun yang terjadi selama scraping
    print(f"Error: {e}")

finally:
    # Blok ini SELALU dijalankan (baik sukses maupun error)
    print("Scrapping semua data telah selesai")
    
    # Keluar dari program dengan status code 0 (sukses)
    sys.exit(0)




