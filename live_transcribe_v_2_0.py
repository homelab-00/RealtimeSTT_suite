# Step 1: Import the necessary module from the Realtime STT library
from RealtimeSTT import AudioToTextRecorder

# Step 2: Define a function to process the transcribed text
def process_transcription(text):
    print("Transcribed text:", text)

# Step 3: Set up a variable to define the model you want to use
# Options: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'
model_size = "tiny.en"  # Set this to the model you want (for example, 'tiny.en' for English only)

# Step 4: Create an instance of the AudioToTextRecorder with the chosen model
with AudioToTextRecorder(model=model_size, enable_realtime_transcription=True) as recorder:
    print(f"Recording started with {model_size} model. Speak into the microphone.")
    
    # Step 5: Run the transcription in a loop, printing each transcribed chunk
    while True:
        recorder.text(process_transcription)
