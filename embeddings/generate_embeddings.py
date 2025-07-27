import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

# Paths
IMAGE_DIR = "data/images"
CSV_PATH = "data/styles.csv"
OUTPUT_PATH = "embeddings/clip_embeddings.pt"

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load metadata
df = pd.read_csv(CSV_PATH, on_bad_lines='skip')

# Drop rows with missing image file
df["image_path"] = df["id"].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))
df = df[df["image_path"].apply(os.path.exists)]

# Limit for quick testing (you can remove this line later)
df = df.sample(500).reset_index(drop=True)

# Combine text features
df["text"] = df["productDisplayName"].fillna("") + " " + df["gender"].fillna("") + " " + df["masterCategory"].fillna("") + " " + df["subCategory"].fillna("")

# Store embeddings
embeddings = []

print("🔄 Generating embeddings...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        image = Image.open(row["image_path"]).convert("RGB")
        text = row["text"]

        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)

        # Average text and image embeddings
        emb = (outputs.image_embeds + outputs.text_embeds) / 2
        embeddings.append(emb.squeeze().cpu())
    except Exception as e:
        print(f"⚠️ Error at index {idx}: {e}")
        continue

# Stack all into tensor and save
all_embeddings = torch.stack(embeddings)
torch.save({"embeddings": all_embeddings, "metadata": df}, OUTPUT_PATH)

print(f"\n✅ Saved {len(embeddings)} embeddings to {OUTPUT_PATH}")
