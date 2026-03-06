import os
import requests
import io
from PIL import Image
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ===============================
# CONFIGURATION
# ===============================

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TEXT_TO_IMAGE_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# ===============================
# LEGAL VISUAL PROMPT GENERATOR
# ===============================

def generate_visual_prompt(user_query, legal_context):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY,
        temperature=0.1
    )

    prompt = f"""
    You are a Legal Visual Consultant.

    Create a visual prompt ONLY if the context is related to Indian Law.

    LEGAL CONTEXT:
    {legal_context}

    USER REQUEST:
    {user_query}

    RULE:
    If context is missing, generic, or not about Indian legal matters
    (BNS, IPC, Court, Rights, FIR, Judiciary),
    output ONLY the word REJECT.

    Otherwise output a detailed artistic prompt for image generation.
    """

    return llm.invoke(prompt).content.strip()


# ===============================
# TEXT → IMAGE (FLUX)
# ===============================

def text_to_image_api(visual_prompt):
    payload = {
        "inputs": visual_prompt,
        "parameters": {
            "num_inference_steps": 4
        }
    }

    response = requests.post(
        TEXT_TO_IMAGE_URL,
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        image.save("legal_visual.png")
        return "legal_visual.png"

    return f"Text-to-Image Error: {response.status_code} - {response.text}"


# ===============================
# MAIN IMAGE GENERATION ENTRY
# ===============================

def process_image_task1(user_query, retrieved_text=None):

    if not retrieved_text or "No specific matches" in retrieved_text:
        return "No relevant legal context found. Please ask a specific legal question first."

    print("⚖️ Verifying legal domain...")
    visual_prompt = generate_visual_prompt(user_query, retrieved_text)

    if "REJECT" in visual_prompt.upper():
        return "This request is not a verified Indian legal scenario."

    print("🎨 Generating legal visual...")
    result = text_to_image_api(visual_prompt)

    return f"Visual generated: {result}"
