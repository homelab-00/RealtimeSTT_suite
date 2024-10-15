"""
RealtimeSTT Test Script
=======================

This script demonstrates the usage of the RealtimeSTT library for real-time speech-to-text transcription.
It initializes the AudioToTextRecorder, starts listening for speech (optionally activated by a wake word),
and processes the transcribed text in real-time.

Author: Kolja Beigel
Email: kolja.beigel@web.de
GitHub: https://github.com/KoljaB/RealtimeSTT
"""

import logging
import sys
import threading
import time

from RealtimeSTT import AudioToTextRecorder

# Configure logging level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_text(text):
    """
    Callback function to handle transcribed text.

    Args:
        text (str): The transcribed text from audio input.
    """
    print(f"Transcribed Text: {text}")

def on_recording_start():
    """
    Callback function triggered when recording starts.
    """
    logging.info("Recording has started.")

def on_recording_stop():
    """
    Callback function triggered when recording stops.
    """
    logging.info("Recording has stopped.")

def on_transcription_start():
    """
    Callback function triggered when transcription starts.
    """
    logging.info("Transcription has started.")

def on_wakeword_detected():
    """
    Callback function triggered when a wake word is detected.
    """
    logging.info("Wake word detected. Listening for speech...")

def on_wakeword_timeout():
    """
    Callback function triggered when wake word activation times out.
    """
    logging.info("Wake word activation timed out. Returning to inactive state.")

def main():
    """
    Main function to initialize and run the RealtimeSTT recorder.
    """
    # Initialize the AudioToTextRecorder with desired configurations
    recorder = AudioToTextRecorder(
        model="tiny",  # Use the 'tiny' model for faster performance
        language="",   # Auto-detect language
        compute_type="float16",
        device="cuda",  # Use GPU if available
        on_recording_start=on_recording_start,
        on_recording_stop=on_recording_stop,
        on_transcription_start=on_transcription_start,
        use_microphone=True,
        spinner=True,
        level=logging.INFO,
        wakeword_backend="pvporcupine",
        wake_words="jarvis",  # Example wake word
        wake_words_sensitivity=0.5,
        on_wakeword_detected=on_wakeword_detected,
        on_wakeword_timeout=on_wakeword_timeout,
        beam_size=5,
        initial_prompt=None,
        suppress_tokens=[-1],
        print_transcription_time=True
    )

    try:
        with recorder:
            logging.info("RealtimeSTT is now running. Press Ctrl+C to exit.")
            while True:
                transcription = recorder.text(process_text)
                if transcription:
                    print(f"Final Transcription: {transcription}")
                time.sleep(0.1)  # Prevents tight looping
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting gracefully...")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        recorder.shutdown()
        logging.info("RealtimeSTT has been shut down.")

if __name__ == "__main__":
    main()
