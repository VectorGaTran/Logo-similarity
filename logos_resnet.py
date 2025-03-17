import requests
import pandas as pd
import cv2
import numpy as np
import json
import concurrent.futures
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
from io import BytesIO
from lxml import html
from urllib.parse import urljoin
import time
import matplotlib.pyplot as plt
import cairosvg
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load ResNet50 (without classification head)
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer
#resnet.fc = torch.nn.Identity()  # Remove classification layer
resnet.eval()  # Set to evaluation mode

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

start_time = time.time()  # Start timer

# Load domains
df = pd.read_parquet("logos.snappy.parquet")
df = df.head(100)  # Limit for testing

# Custom headers to bypass 403 errors
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

# Fetch website HTML
def fetch_website_html(domain):
    url = f"https://www.{domain}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        return response.text, url
    except requests.exceptions.RequestException as e:
        print(f"Failed to load site: {domain}, Error: {e}")
        return None, None

# Extract logo URL from HTML
def extract_logo_url(html_content, base_url):
    try:
        tree = html.fromstring(html_content)
        img_tags = tree.xpath("//img")
        logo_urls = []

        for img in img_tags:
            src = img.get("src", "")
            if not src:
                continue
            full_url = urljoin(base_url, src)

            # Prioritize images with 'logo' in filename
            if "logo" in src.lower():
                logo_urls.append(full_url)

        # Check for company branding containers if no logo-named images are found
        if not logo_urls:
            branding_divs = tree.xpath(
                "//div[contains(@class, 'logo') or contains(@class, 'branding') or contains(@class, 'company')]//img"
            )
            for img in branding_divs:
                src = img.get("src", "")
                if src:
                    logo_urls.append(urljoin(base_url, src))

        if not logo_urls:
            return None  # No logo found

        # Skip "condensed" logos if other options exist
        filtered_logos = [url for url in logo_urls if "condensed" not in url]

        return filtered_logos[0] if filtered_logos else logo_urls[0]

    except Exception as e:
        print(f"Error extracting logo: {e}")
        return None

# Fetch logo from Clearbit as fallback
def fetch_clearbit_logo(domain):
    clearbit_url = f"https://logo.clearbit.com/{domain}"
    try:
        response = requests.get(clearbit_url, timeout=5)
        if response.status_code == 200:
            print(f"Using Clearbit logo for {domain}: {clearbit_url}")
            return Image.open(BytesIO(response.content)).convert("RGBA")
    except Exception as e:
        print(f"Clearbit logo failed for {domain}: {e}")
    return None

# Fetch and process logo
def fetch_logo(domain):
    html_content, base_url = fetch_website_html(domain)
    if not html_content:
        return fetch_clearbit_logo(domain)  # Try Clearbit if site fails

    logo_url = extract_logo_url(html_content, base_url)
    if not logo_url:
        print(f"No logo found on site for: {domain}, trying Clearbit...")
        return fetch_clearbit_logo(domain)  # Use Clearbit if no logo found
    
    print(f"Found logo for {domain}: {logo_url}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Referer": base_url  # Pretend the request comes from the website itself
        }
        response = requests.get(logo_url, headers=headers, timeout=5)
        response.raise_for_status()

        if logo_url.endswith(".svg"):
            return process_svg(response.content)
        else:
            img = Image.open(BytesIO(response.content)).convert("RGBA")
            img = ImageOps.contain(img, (128, 128))
            return img

    except Exception as e:
        print(f"Error processing {domain}, trying Clearbit: {e}")
        return fetch_clearbit_logo(domain)  # Fallback to Clearbit

# Convert SVG to rasterized image (PNG)
def process_svg(svg_content):
    try:
        # Convert SVG to PNG bytes
        png_data = cairosvg.svg2png(bytestring=svg_content)
        
        # Convert PNG bytes to PIL Image
        img = Image.open(BytesIO(png_data)).convert("RGBA")
        img = ImageOps.contain(img, (128, 128))  # Resize to match PNG processing
        return img
    except Exception as e:
        print(f"Error processing SVG: {e}")
        return None
    
# Download and preprocess logos
logo_data = {}

#Iterative version, for testing:
# for i, domain in enumerate(df["domain"]):
#     print(f"Processing {i+1}/{len(df)}: {domain}")
#     logo = fetch_logo(domain)
#     # plt.imshow(logo)
#     # plt.axis("off")  # Hide axes
#     # plt.title(domain)
#     # plt.show()
#     if logo:
#         logo_data[domain] = np.array(logo.convert("RGB"))  # Keep color information

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(fetch_logo, df["domain"])
    for domain, logo in zip(df["domain"], results):
        if logo:
            logo_data[domain] = np.array(logo.convert("RGB"))

# Feature extraction 
feature_vectors = []
domain_list = []

for domain, img in logo_data.items():
    img = Image.fromarray(img)  # Convert NumPy array to PIL Image
    img = transform(img).unsqueeze(0)  # Apply transformations & add batch dim
    
    with torch.no_grad():  
        features = resnet(img)  # Extract features
        features = features.view(-1).cpu().numpy()  # Flatten to 1D array

    feature_vectors.append(features)
    domain_list.append(domain)

# Convert to NumPy array
feature_vectors = np.array(feature_vectors, dtype=np.float32)
scaler = StandardScaler()
feature_vectors = scaler.fit_transform(feature_vectors)

# Clustering
print("Starting clustering...")
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=50).fit(feature_vectors)

# Group similar domains
clusters = {}
for idx, label in enumerate(clustering.labels_):
    label = int(label)  # Convert int64 to regular int
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(domain_list[idx])

# Save output
with open("logo_groups_resnet.json", "w") as f:
    json.dump(clusters, f, indent=4)

print("Clustering complete. Results saved in 'logo_groups_resnet.json'.")

end_time = time.time()  # End timer
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")