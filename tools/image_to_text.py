import os
import torch
import fitz  # PyMuPDF for PDFs
from docx import Document  # python-docx for Word
from PIL import Image
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ===============================
# Load environment variables
# ===============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ===============================
# OCR & Model Setup (GLM-OCR)
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# CRITICAL FIX: Added trust_remote_code=True
try:
    processor = AutoProcessor.from_pretrained(
        "zai-org/GLM-OCR", 
        trust_remote_code=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "zai-org/GLM-OCR", 
        trust_remote_code=True
    ).to(device)
except Exception as e:
    print(f"⚠️ Model Loading Warning: {e}. Ensure you have an internet connection for the first run.")

def image_to_text(image_input) -> str:
    """Extract text from a PIL image object using GLM-OCR."""
    try:
        inputs = processor(images=image_input.convert("RGB"), return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    except Exception as e:
        return f"[OCR Error: {str(e)}]"

# ===============================
# MULTI-FORMAT EXTRACTION LOGIC
# ===============================

def extract_text_from_file(file_path: str) -> str:
    """Main router to extract text from PDF, DOCX, or Images."""
    ext = os.path.splitext(file_path)[1].lower()
    full_text = ""

    # 1. Handle PDFs (Hybrid: Text-based or Scanned)
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                full_text += page_text + "\n"
            else:
                # If no text layer exists, run OCR on the page
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                full_text += image_to_text(img) + "\n"
        doc.close()

    # 2. Handle Word Documents
    elif ext == ".docx":
        doc = Document(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs])

    # 3. Handle Standard Images
    elif ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(file_path)
        full_text = image_to_text(img)

    return full_text.strip()

# ===============================
# MAIN ENTRY (Process Task)
# ===============================

def process_image_task(user_query, retrieved_text=None, file_path=None):
    """
    Unified task handler for Images, PDFs, and Docs.
    Sync this name with main_agent.py routing logic.
    """
    if not file_path or not os.path.exists(file_path):
        return "No valid file found. Please upload a document first."

    # Step 1: Extract Text
    content = extract_text_from_file(file_path)
    if not content:
        return "Could not extract any readable text from the file."

    # Step 2: Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        groq_api_key=GROQ_API_KEY, 
        temperature=0.1
    )

    # Step 3: Legal Analysis Prompt
    analysis_prompt = f"""
    You are a Senior Indian Legal Consultant.
    
    USER QUESTION: {user_query}
    DOCUMENT CONTENT: {content[:8000]}

    INSTRUCTIONS:
    1. Verify if the content relates to Indian Law (BNS, IPC, FIR, Court, Rights, Indian Constitution).
    2. If NOT related, output 'REJECT'.
    3. If VALID, answer the user's question specifically using the document content.
    4. Provide a concise summary of the document's legal significance.
    """
    
    response = llm.invoke(analysis_prompt).content.strip()
    
    if "REJECT" in response.upper():
        return "This document does not belong to the Indian legal domain or does not contain legal context."
    
    return response