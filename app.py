import streamlit as st
import os
import json
import uuid
from datetime import datetime
from langchain_groq import ChatGroq
from main_agent import main_agent_router
# Import the live speech function
# from tools.speech_to_text import transcribe_live_speech 
from tools.speech_to_text import start_live_recording, stop_and_transcribe
from tools.legal_rag import legal_chat_flow
from tools.legal_rag_eval import run_evaluation_cycle, import_from_nested_history

# --- CONFIGURATION ---
HISTORY_FILE = "chat_history.json"
st.set_page_config(page_title="NyayaVani: Indian Legal AI", layout="wide")

# --- DATA PERSISTENCE ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_to_history(session_id, messages, title):
    history = load_history()
    history[session_id] = {
        "title": title,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": messages
    }
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "recording_stream" not in st.session_state:
    st.session_state.recording_stream = None
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# --- SIDEBAR: CHAT HISTORY & VOICE ---
with st.sidebar:
    st.title("⚖️ NyayaVani AI")


# # --- Step 1: Import Data ---
    if st.button("📥 Import Chat History to Golden Set"):
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r") as f:
                history_data = json.load(f)
            added = import_from_nested_history(history_data)
            st.success(f"Added {added} new unique cases to golden.json!")
        else:
            st.error("chat_history.json not found.")

# st.divider()

# # --- Step 2: Run Evaluation ---
    if st.button("🚀 Run Full Evaluation"):
        llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    
        with st.status("Evaluating Golden Set..."):
            metrics = run_evaluation_cycle(llm)
            
        if metrics:
            st.success("Evaluation Complete!")
        
        # Displaying the Stats
            st.subheader("Core Accuracy")
            c1, c2 = st.columns(2)
            c1.metric("Avg Accuracy", f"{metrics['average_accuracy']:.1%}")
            c2.metric("Domain Relevance", f"{metrics['domain_relevance_rate']:.1%}")

            st.subheader("Retrieval Quality")
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{metrics['precision']:.2f}")
            m2.metric("Recall", f"{metrics['recall']:.2f}")
            m3.metric("MRR", f"{metrics['mrr']:.2f}")
        else:
            st.error("Evaluation failed. Ensure golden.json has data.")
    
    # REAL-TIME VOICE BUTTON
    st.subheader("Voice Consultation")
    # We use a session state trick to inject this into the chat flow

    if not st.session_state.is_recording:
        if st.button("🎤 Start Speaking", use_container_width=True):
            st.session_state.is_recording = True
            st.session_state.recording_stream = start_live_recording()
            st.rerun()
    else:
        if st.button("🛑 Stop & Process", use_container_width=True, type="primary"):
            with st.spinner("Processing speech..."):
                voice_text = stop_and_transcribe(st.session_state.recording_stream)
                st.session_state.is_recording = False
                st.session_state.recording_stream = None
                if voice_text:
                    st.session_state.voice_prompt = voice_text
            st.rerun()
        st.warning("Recording in progress... Click Stop when finished.")
    
    st.divider()
    if st.button("➕ New Consultation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.write("📂 **Previous Chats**")
    past_chats = load_history()
    for sid, data in reversed(list(past_chats.items())):
        if st.button(f"{data['title']} ({data['date']})", key=sid):
            st.session_state.messages = data['messages']
            st.session_state.session_id = sid
            st.rerun()

# --- MAIN UI ---
st.title("Legal Intelligence Portal")
st.caption("Real-time Voice & Document Intelligence Enabled")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- ATTACHMENTS AREA ---
col1, col2 = st.columns(2)

with col1:
    # We name it 'uploaded_image' here so the rest of your 
    # agent code (which expects this name) doesn't crash.
    uploaded_image = st.file_uploader(
        "🖼️ Evidence / PDF / Doc", 
        type=["png", "jpg", "jpeg", "pdf", "docx"]
    )

    if uploaded_image is not None:
        # Get extension: .pdf, .docx, etc.
        file_extension = os.path.splitext(uploaded_image.name)[1].lower()
        save_path = f"input_file{file_extension}"
        
        # Cleanup old versions of the file
        for ext in [".png", ".jpg", ".jpeg", ".pdf", ".docx"]:
            old_file = f"input_file{ext}"
            if os.path.exists(old_file) and old_file != save_path:
                os.remove(old_file)

        # Save the file with the correct extension
        with open(save_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        st.success(f"Successfully uploaded: {uploaded_image.name}")



# --- CHAT LOGIC ---
# Get prompt from either Text Input or Voice Session State
prompt = st.chat_input("Ask about BNS, IPC, or legal procedures...")

if "voice_prompt" in st.session_state:
    prompt = st.session_state.voice_prompt
    del st.session_state.voice_prompt # Clear it after use

if prompt:
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("NyayaVani is thinking..."):
        try:
            input_query = prompt
            file_to_pass = None

            # 2. Handle Document/Image Uploads
            if uploaded_image:
                ext = uploaded_image.name.split('.')[-1]
                file_to_pass = f"input_file.{ext}"
                with open(file_to_pass, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                input_query += f" [FILE ATTACHED: {uploaded_image.name}]"
            
            # 3. Handle Audio File Uploads
            # if uploaded_audio:
            #     with open("user_voice.mp3", "wb") as f:
            #         f.write(uploaded_audio.getbuffer())
            #     input_query += " [AUDIO FILE ATTACHED]"

            # 4. Route to Brain
            # We update the router to accept the specific file path
            response = main_agent_router(input_query)
            
            # 5. Add AI response
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
                # If audio was generated by the TTS tool, play it
                if "🔊 Audio generated" in response:
                    st.audio("legal_response_audio.flac")
            
            # 6. Save
            save_to_history(st.session_state.session_id, st.session_state.messages, prompt[:30] + "...")

        except Exception as e:
            st.error(f"Error: {e}")