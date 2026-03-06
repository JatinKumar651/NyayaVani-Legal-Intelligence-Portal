import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Custom Tools
from tools.legal_rag import legal_chat_flow
from tools.image_tool import process_image_task1
# from tools.audio_tool import process_audio_task
# Updated imports for non-blocking voice
from tools.speech_to_text import start_live_recording, stop_and_transcribe
from tools.image_to_text import process_image_task

load_dotenv()

def get_routing_brain():
    """Initializes the Groq Llama-3.3 router."""
    return ChatGroq(
        model="llama-3.3-70b-versatile", 
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0 
    )

def main_agent_router(user_input):
    """Routes queries to specialized legal tools."""
    brain = get_routing_brain()
    
    routing_prompt = f"""
    You are a Legal Project Manager. Route the request to the correct department.
    CATEGORIES:
    1. LEGAL_TEXT: General legal questions.
    2. IMAGE_GEN: User wants to 'visualize', 'see', or 'generate' a legal scene.
    3. IMAGE_DESC: User wants to 'read', 'describe' an image or document.
    4. AUDIO_TTS: User wants to 'hear' the response or get audio response.

    Respond with ONLY the category name.
    USER REQUEST: {user_input}
    """
    
    intent_response = brain.invoke(routing_prompt)
    intent = intent_response.content.strip().upper()
    print(f"🤖 Brain Intent: {intent}")

    # Routing Logic
    if "IMAGE_GEN" in intent:
        legal_context = legal_chat_flow(user_input) 
        return process_image_task1(user_query=user_input, retrieved_text=legal_context)

    elif "IMAGE_DESC" in intent:
        # Check for various file types that might have been uploaded
        target_file = None
        for ext in [".pdf", ".docx", ".png", ".jpg", ".jpeg"]:
            if os.path.exists(f"input_file{ext}"):
                target_file = f"input_file{ext}"
                break
        
        if target_file:
            return process_image_task(user_query=user_input, file_path=target_file)
        else:
            return "No uploaded file found. Please upload a document (PDF/DOCX/Image) in the UI first."

    elif "AUDIO_TTS" in intent:
        answer = legal_chat_flow(user_input)
        audio_file = process_audio_task(response_text=answer)
        return f"Legal Answer: {answer}\n\n🔊 Audio generated: {audio_file}"

    else:
        return legal_chat_flow(user_input)

if __name__ == "__main__":
    print("⚖️ Indian Legal AI Agent | Multi-Modal CLI")
    print("Commands: 'voice' to record, 'exit' to quit.")
    
    

    while True:
        query = input("\nUser (or 'voice'): ").strip()
        
        if query.lower() in ['exit', 'quit']: 
            break
        
        # --- IMPROVED VOICE MODE ---
        if query.lower() == "voice":
            try:
                # Start background stream
                stream = start_live_recording()
                input("🎤 Recording... Press ENTER to stop and process.")
                
                # Stop stream and get Whisper transcription
                query = stop_and_transcribe(stream)
                print(f"🗣️ Transcribed: {query}")
                
                if not query:
                    print("⚠️ No speech detected.")
                    continue
            except Exception as e:
                print(f"❌ Voice Error: {e}")
                continue
        
        if not query: 
            continue

        try:
            # Send either the typed query or the transcribed voice to the router
            response = main_agent_router(query)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"❌ Processing Error: {e}")