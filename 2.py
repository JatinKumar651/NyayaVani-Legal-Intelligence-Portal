import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 2026 Legal Context Model (High-speed & Reliable)
MODEL_NAME = "all-MiniLM-L6-v2" 

def load_embedding_model():
    print(f"📡 Loading model: {MODEL_NAME}...")
    return SentenceTransformer(MODEL_NAME)

def create_cosine_faiss_index(embeddings):
    """
    Creates a FAISS index optimized for Cosine Similarity.
    For normalized vectors: Inner Product == Cosine Similarity.
    """
    embeddings_np = np.array(embeddings).astype("float32")
    
    # 1. Normalize vectors to unit length (length = 1.0)
    faiss.normalize_L2(embeddings_np)
    
    dimension = embeddings_np.shape[1]
    
    # 2. Use IndexFlatIP (Inner Product) for Cosine Similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    return index

def save_parent_child_index(index, child_chunks, output_dir="faiss_index"):
    """
    Saves the vector index and the child chunks (which carry parent metadata).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the FAISS vector binary
    faiss.write_index(index, os.path.join(output_dir, "index.bin"))
    
    # Save the chunks mapping. Each child here has 'parent_text' in its metadata.
    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(child_chunks, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    INPUT_FILE = "hybrid_chunks.json"
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ '{INPUT_FILE}' not found!")
        exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        full_data = json.load(f)
        
        # FIX: Access the 'children' list specifically
        # Your previous script saved data as {"parents": [...], "children": [...]}
        if isinstance(full_data, dict) and "children" in full_data:
            child_chunks = full_data["children"]
        else:
            # Fallback in case your file is already a flat list
            child_chunks = full_data

    model = load_embedding_model()
    
    # Now this list comprehension will work correctly
    texts_to_embed = [chunk["text"] for chunk in child_chunks]

    print(f"🚀 Generating embeddings for {len(texts_to_embed)} CHILD chunks...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, batch_size=32)

    print("🛠️ Building FAISS Index (Parent-Child Optimized)...")
    index = create_cosine_faiss_index(embeddings)
    
    # Save child_chunks so the metadata (parent_text) is available for File 3
    save_parent_child_index(index, child_chunks)

    print(f"\n✨ Done! Indexed {index.ntotal} children.")