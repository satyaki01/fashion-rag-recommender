import os
import faiss
import torch
import pickle

# Paths
EMBEDDING_FILE = "embeddings/clip_embeddings.pt"
FAISS_INDEX_FILE = "vector_db/faiss_index.index"
METADATA_FILE = "vector_db/item_metadata.pkl"

# Load embeddings and metadata
data = torch.load(EMBEDDING_FILE, weights_only=False)
embeddings = data["embeddings"].detach().numpy()
metadata_df = data["metadata"].reset_index(drop=True)

# Build FAISS index (L2 or Cosine)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Cosine similarity (dot-product)

# Normalize vectors for cosine similarity
faiss.normalize_L2(embeddings)

print("🔄 Adding vectors to FAISS index...")
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, FAISS_INDEX_FILE)
with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata_df.to_dict(orient="records"), f)

print(f"✅ Saved FAISS index to {FAISS_INDEX_FILE}")
print(f"✅ Saved metadata to {METADATA_FILE}")
print(f"📦 Total items indexed: {len(embeddings)}")
