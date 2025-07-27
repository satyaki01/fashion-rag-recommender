import matplotlib.pyplot as plt
import torch
import faiss
import pickle
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_community.llms import Ollama
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===== Load CLIP model =====
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# ===== Load FAISS index and metadata =====
faiss_index = faiss.read_index("vector_db/faiss_index.index")
with open("vector_db/item_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# ===== Embed user query =====
def get_query_embedding(query):
    inputs = clip_processor(text=[query], images=[Image.new("RGB", (224, 224))], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    return outputs.text_embeds[0].detach().numpy()

# ===== Retrieve top-k similar items =====
def retrieve_items(query, top_k=5):
    query_vector = get_query_embedding(query)
    faiss.normalize_L2(query_vector.reshape(1, -1))
    distances, indices = faiss_index.search(query_vector.reshape(1, -1), top_k)
    return [metadata[i] for i in indices[0]]

# ===== Use Mistral LLM via Ollama =====
def generate_explanation(query, items):
    items_text = "\n".join(
        [f"- {item['productDisplayName']} ({item['gender']}, {item['masterCategory']}, {item['subCategory']})" for item in items]
    )

    prompt = f"""
You are a fashion stylist. A user asked: "{query}"

Here are some product options from the catalog:
{items_text}

Please recommend a few items from above, explain why they fit the occasion, and keep the tone natural and stylish.
"""

    llm = Ollama(model="mistral")
    return llm.invoke(prompt)


def run_fashion_rag(query, top_k=5):
    items = retrieve_items(query, top_k=top_k)
    explanation = generate_explanation(query, items)

    result = {
        "items": items,
        "explanation": explanation.strip()
    }
    return result

# ===== Run and Test =====
if __name__ == "__main__":
    query = input("🧠 Enter your fashion query: ")
    result = run_fashion_rag(query)
    
    print("\n🎯 Final Output:\n")
    print(result)

def show_images(items):
    plt.figure(figsize=(15, 3))
    for i, item in enumerate(items):
        img_path = item['image_path'].replace("\\", "/")  # Normalize path
        try:
            img = Image.open(img_path).resize((150, 200))
        except FileNotFoundError:
            print(f"⚠️ Image not found: {img_path}")
            img = Image.new("RGB", (150, 200), color="gray")  # Placeholder image


        plt.subplot(1, len(items), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(item['productDisplayName'][:20], fontsize=8)

    plt.tight_layout()
    plt.show()

# Call it at the end
show_images(result["items"])
