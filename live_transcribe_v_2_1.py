from RealtimeSTT import AudioToTextRecorder

def process_transcription(text):
    print("Transcribed text:", text)

if __name__ == "__main__":
    # Set the model size here
    model_size = "tiny.en"  # You can change this to another model like 'tiny', 'base', etc.
    
    # Start the real-time transcription process
    with AudioToTextRecorder(model=model_size, enable_realtime_transcription=True) as recorder:
        print(f"Recording started with {model_size} model. Speak into the microphone.")
        
        while True:
            recorder.text(process_transcription)
