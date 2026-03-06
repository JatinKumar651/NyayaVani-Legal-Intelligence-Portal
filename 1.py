
import os
import json
import uuid
import re
import glob
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
PARENT_SIZE = 2500
CHILD_SIZE = 500
OVERLAP = 100

def load_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        return "".join([p.extract_text() + "\n" for p in reader.pages if p.extract_text()])
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""

def hybrid_parent_splitter(text):
    """
    Hybrid Strategy: First split by Legal Sections/Chapters, 
    then sub-split large sections recursively.
    """
    # Pattern for "Section X", "Chapter X", or "Article X"
    legal_pattern = r'\n(?=(?:Section|Section|CHAPTER|Chapter|Article|Clause)\s+\d+)'
    
    # Initial split by structural markers
    sections = re.split(legal_pattern, text)
    
    refined_parents = []
    # Fallback splitter for sections that are still too long
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_SIZE, 
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", ". "]
    )
    
    for sec in sections:
        if len(sec) > PARENT_SIZE:
            refined_parents.extend(fallback_splitter.split_text(sec))
        else:
            refined_parents.append(sec)
    return refined_parents

def process_hybrid_documents(data_dir="data"):
    all_data = {"parents": [], "children": []}
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_SIZE, chunk_overlap=OVERLAP)
    
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        raw_text = load_pdf(pdf_path)
        if not raw_text: continue

        # 1. Hybrid Parent Splitting (Structure + Size)
        parent_chunks = hybrid_parent_splitter(raw_text)

        for p_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            
            # Store Parent
            all_data["parents"].append({
                "parent_id": parent_id,
                "text": p_text.strip(),
                "metadata": {"source": filename, "type": "legal_section"}
            })

            # 2. Child Splitting for Vector Search
            child_chunks = child_splitter.split_text(p_text)
            for c_text in child_chunks:
                all_data["children"].append({
                    "text": c_text.strip(),
                    "metadata": {
                        "parent_id": parent_id,
                        "source": filename,
                        "law_type": "BNS" if "BNS" in filename.upper() else "IPC"
                    }
                })
                
    return all_data

if __name__ == "__main__":
    data = process_hybrid_documents()
    with open("hybrid_chunks.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Created {len(data['parents'])} Parents and {len(data['children'])} Children.")