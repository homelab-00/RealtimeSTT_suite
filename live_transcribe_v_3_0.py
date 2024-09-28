import argparse
import sys
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from RealtimeSTT import RealtimeSTT

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def main():
    parser = argparse.ArgumentParser(description="Real-time Speech-to-Text Transcription using RealtimeSTT")
    parser.add_argument(
        "-m", "--model", type=str, default="tiny.en",
        help="Whisper model to use (e.g., tiny, tiny.en, base, base.en, etc.)"
    )
    parser.add_argument(
        "-d", "--device", type=int_or_str,
        help="Input device ID or substring"
    )
    parser.add_argument(
        "-r", "--samplerate", type=int, default=16000,
        help="Sampling rate (default: 16000)"
    )
    args = parser.parse_args()

    # Initialize RealtimeSTT with the selected model
    try:
        stt = RealtimeSTT(model=args.model)
    except Exception as e:
        print(f"Error initializing RealtimeSTT: {e}")
        sys.exit(1)

    q = queue.Queue()

    def audio_callback(indata, frames, time_, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        # Open the microphone stream
        with sd.InputStream(samplerate=args.samplerate, device=args.device,
                            channels=1, callback=audio_callback):
            print(f"Listening... Using model '{args.model}'. Press Ctrl+C to stop.")
            stt.start()
            while True:
                data = q.get()
                audio = np.frombuffer(data, dtype=np.float32)
                stt.process(audio)
    except KeyboardInterrupt:
        print("\nStopping transcription.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stt.stop()

if __name__ == "__main__":
    main()
