import RealtimeSTT  # Ensure this is the correct import for the package
import sounddevice as sd  # Assuming sounddevice is used for capturing audio

def callback(indata, frames, time, status):
    """Callback function to process incoming audio."""
    if status:
        print(status)
    # Assuming the STT library provides a method for real-time transcription
    transcription = RealtimeSTT.transcribe(indata)
    print(f"Transcribed: {transcription}")

def main():
    print("Starting live transcription...")
    try:
        # Open the audio stream for capturing audio
        with sd.InputStream(callback=callback):
            print("Listening...")
            input()  # Keep the program running until you press Enter
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
