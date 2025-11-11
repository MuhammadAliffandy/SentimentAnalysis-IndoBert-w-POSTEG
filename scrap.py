from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import sys
import os


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

options = Options()
options.add_argument("--headless")  # Mode tanpa tampilan GUI
options.add_argument("--disable-gpu")  # Matikan GPU rendering
options.add_argument("--disable-software-rasterizer")  # Hindari fallback ke software rendering
options.add_argument("--use-gl=swiftshader")  # Paksa penggunaan SwiftShader
options.add_argument("--enable-unsafe-webgl")  # (Opsional) Jika butuh WebGL di headless mode

service = Service(ChromeDriverManager().install())

# Buat folder data jika belum ada
os.makedirs("data", exist_ok=True)

try:
    for i in data:
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://www.google.com/maps")
        print(f"Google Maps dibuka untuk: {i}")
        time.sleep(3)
        search_query = i
        search_box = driver.find_element(By.ID, "searchboxinput")
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.ENTER)
        try:
            first_result = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@aria-label, 'Hasil untuk')]//a"))
            )
            first_result.click()
            time.sleep(5)
        except Exception as e:
            print("Gagal menekan hasil pencarian:", e)
        try:
            reviews_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'ulasan')]"))
            )
            reviews_button.click()
            time.sleep(3)
        except Exception as e:
            print("Tidak bisa menemukan tombol ulasan:", e)
            driver.quit()
            continue
        try:
            more_reviews_button = driver.find_element(By.XPATH, "//span[contains(text(), 'Ulasan lainnya')]")
            more_reviews_button.click()
            time.sleep(3)
        except:
            print("Tidak ada tombol 'Ulasan Lainnya', lanjut ke scraping.")
        scrollable_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".m6QErb.kA9KIf"))
        )
        last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
        num = 1
        while True:
            print('scrapping process =' + f'{"="*num}')
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
            time.sleep(3)
            new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
            if new_height == last_height:
                break
            last_height = new_height
            num += 1
        review_elements = driver.find_elements(By.CLASS_NAME, "wiI7pd")
        reviews = []
        for review in review_elements:
            try:
                text = review.text.strip()
                if text:
                    reviews.append(text)
            except:
                continue
        # Simpan hasil ke file JSON
        filename = f"./data/{i.replace(' ','_').lower()}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"reviews": reviews}, f, indent=4, ensure_ascii=False)
        print(f"Hasil {i} disimpan ke {filename}")
        driver.quit()
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Scrapping semua data telah selesai")
    sys.exit(0)




