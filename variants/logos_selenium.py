import requests
import pandas as pd
import cv2
import numpy as np
import json
import time
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
from io import BytesIO
from lxml import html, etree
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

start_time = time.time()

df = pd.read_parquet("logos.snappy.parquet")
df = df.head(2)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}

def get_selenium_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def fetch_dynamic_html(domain):
    url = f"https://www.{domain}"
    try:
        driver = get_selenium_driver()
        driver.get(url)
        html_content = driver.page_source
        driver.quit()
        return html_content, url
    except Exception as e:
        print(f"Selenium failed for {domain}: {e}")
        return None, None

def extract_logo_url(html_content, base_url):
    try:
        tree = html.fromstring(html_content)
        img_tags = tree.xpath("//img")
        logo_urls = [urljoin(base_url, img.get("src", "")) for img in img_tags if "logo" in img.get("src", "").lower()]
        if not logo_urls:
            branding_divs = tree.xpath("//div[contains(@class, 'logo') or contains(@class, 'branding')]//img")
            logo_urls = [urljoin(base_url, img.get("src", "")) for img in branding_divs]
        return logo_urls[0] if logo_urls else None
    except Exception as e:
        print(f"Error extracting logo: {e}")
        return None

def fetch_clearbit_logo(domain):
    clearbit_url = f"https://logo.clearbit.com/{domain}"
    try:
        response = requests.get(clearbit_url, timeout=5)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGBA")
    except Exception as e:
        print(f"Clearbit logo failed for {domain}: {e}")
    return None

def fetch_logo(domain):
    html_content, base_url = fetch_dynamic_html(domain)
    if not html_content:
        return domain, fetch_clearbit_logo(domain)
    logo_url = extract_logo_url(html_content, base_url)
    if not logo_url:
        return domain, fetch_clearbit_logo(domain)
    try:
        headers = {"User-Agent": HEADERS["User-Agent"], "Referer": base_url}
        response = requests.get(logo_url, headers=headers, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGBA")
        img = ImageOps.contain(img, (128, 128))
        return domain, img
    except Exception as e:
        return domain, fetch_clearbit_logo(domain)

logo_data = {}
with ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(fetch_logo, df["domain"])
    for domain, logo in results:
        if logo:
            logo_data[domain] = np.array(logo.convert("RGB"))

orb = cv2.ORB_create()
fixed_size = 500
feature_vectors = []
domain_list = []

for domain, img in logo_data.items():
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    key_points, descriptors = orb.detectAndCompute(gray_img, None)
    descriptors = descriptors if descriptors is not None else np.zeros((fixed_size, 32), dtype=np.uint8)
    feature_vector = descriptors.flatten()
    feature_vector = np.pad(feature_vector, (0, fixed_size * 32 - feature_vector.shape[0]), mode='constant') if feature_vector.shape[0] < fixed_size * 32 else feature_vector[:fixed_size * 32]
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    hist /= np.sum(hist)
    feature_vectors.append(np.hstack((feature_vector, hist)))
    domain_list.append(domain)

feature_vectors = np.array(feature_vectors, dtype=np.float32)
scaler = StandardScaler()
feature_vectors = scaler.fit_transform(feature_vectors)

print("Starting clustering...")
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=50).fit(feature_vectors)

clusters = {}
for idx, label in enumerate(clustering.labels_):
    label = int(label)
    clusters.setdefault(label, []).append(domain_list[idx])

with open("logo_groups.json", "w") as f:
    json.dump(clusters, f, indent=4)

print("Clustering complete. Results saved in 'logo_groups.json'.")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")
