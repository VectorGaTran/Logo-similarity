import requests
import pandas as pd
import cv2
import numpy as np
import json
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
from io import BytesIO
from lxml import html
from urllib.parse import urljoin
import time
import matplotlib.pyplot as plt
import cairosvg
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import random

start_time = time.time()  # Start timer

# Load domains
df = pd.read_parquet("logos.snappy.parquet")
df = df.head(10)  # Limit for testing

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

        # **Skip "condensed" logos if other options exist**
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
for i, domain in enumerate(df["domain"]):
    print(f"Processing {i+1}/{len(df)}: {domain}")
    logo = fetch_logo(domain)
    # plt.imshow(logo)
    # plt.axis("off")  # Hide axes
    # plt.title(domain)
    # plt.show()
    if logo:
        logo_data[domain] = np.array(logo.convert("RGB"))  # Keep color information


# Step 1: Create positive and negative pairs for training
def create_pairs(logo_data, num_pairs):
    positive_pairs = []
    negative_pairs = []
    
    domains = list(logo_data.keys())

    # Generate positive pairs
    for _ in range(num_pairs):
        domain_1, domain_2 = random.sample(domains, 2)
        positive_pairs.append((logo_data[domain_1], logo_data[domain_2], 1))  # Similar pair (label 1)

    # Generate negative pairs
    for _ in range(num_pairs):
        domain_1, domain_2 = random.sample(domains, 2)
        negative_pairs.append((logo_data[domain_1], logo_data[domain_2], 0))  # Dissimilar pair (label 0)
    
    return positive_pairs + negative_pairs

# Prepare pairs (use 100 pairs for example)
pairs = create_pairs(logo_data, 100)

# Step 2: Build the Siamese network
def build_base_network(input_shape):
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(input)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    return Model(input, x)

# Define the Siamese network
input_shape = (128, 128, 3)  # Size of the logo images
input_a = layers.Input(shape=input_shape)
input_b = layers.Input(shape=input_shape)

base_network = build_base_network(input_shape)

embedding_a = base_network(input_a)
embedding_b = base_network(input_b)

# Calculate the distance between the embeddings
distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([embedding_a, embedding_b])
output = layers.Dense(1, activation='sigmoid')(distance)

siamese_network = Model(inputs=[input_a, input_b], outputs=output)

# Compile the model
siamese_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the Siamese network
def preprocess_image(img):
    img = Image.fromarray(img)
    img = img.resize((128, 128))
    return img_to_array(img)

# Prepare training data (pairing images with labels)
train_data_a = []
train_data_b = []
labels = []

for img1, img2, label in pairs:
    train_data_a.append(preprocess_image(img1))
    train_data_b.append(preprocess_image(img2))
    labels.append(label)

train_data_a = np.array(train_data_a)
train_data_b = np.array(train_data_b)
labels = np.array(labels)

siamese_network.fit([train_data_a, train_data_b], labels, epochs=10, batch_size=16)

# Step 4: Generate embeddings for logos using the trained Siamese network
def get_embedding(model, logo1, logo2):
    logo1_preprocessed = preprocess_image(logo1)
    logo2_preprocessed = preprocess_image(logo2)
    # Pass both images to the model
    return model.predict([np.expand_dims(logo1_preprocessed, axis=0), np.expand_dims(logo2_preprocessed, axis=0)])

# Generate embeddings by comparing logos to themselves or another logo
embeddings = {}
for domain, logo in logo_data.items():
    # Use the same logo for comparison (can be adjusted if needed)
    embeddings[domain] = get_embedding(siamese_network, logo, logo)

# Step 5: Clustering using embeddings
embedding_list = list(embeddings.values())

# Flatten the embeddings from 3D to 2D (samples x features)
flattened_embeddings = [embedding.flatten() for embedding in embedding_list]

# Scale the embeddings
scaler = StandardScaler()
flattened_embeddings = scaler.fit_transform(flattened_embeddings)

# Perform clustering
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5).fit(flattened_embeddings)

# Group similar domains
clusters = {}
for idx, label in enumerate(clustering.labels_):
    label = int(label)  # Convert int64 to regular int
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(list(logo_data.keys())[idx])

# Save output
with open("logo_groups_siam.json", "w") as f:
    json.dump(clusters, f, indent=4)

print("Clustering complete. Results saved in 'logo_groups_siam.json'.")

end_time = time.time()  # End timer
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")