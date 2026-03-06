import whisper
import sounddevice as sd
import numpy as np

# Load model once at startup
model = whisper.load_model("base")

# Global variable to store the recording state for the current session
recording_data = []

def callback(indata, frames, time, status):
    """This is called for each audio block by sounddevice."""
    if status:
        print(status)
    recording_data.append(indata.copy())

def start_live_recording(samplerate=16000):
    """Starts the background audio stream."""
    global recording_data
    recording_data = [] # Reset buffer
    stream = sd.InputStream(samplerate=samplerate, channels=1, callback=callback)
    stream.start()
    return stream

def stop_and_transcribe(stream):
    """Stops the stream and processes the collected audio."""
    stream.stop()
    stream.close()
    
    if not recording_data:
        return ""

    # Flatten the list of arrays into one large numpy array
    audio_data = np.concatenate(recording_data, axis=0).flatten().astype(np.float32)
    
    # Transcribe using Whisper
    result = model.transcribe(audio_data, fp16=False)
    return result["text"].strip()