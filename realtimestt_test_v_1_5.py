from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Style
import colorama
import os
import keyboard  # For global hotkeys
import threading
import sys
import time

def main():
    print("Initializing RealtimeSTT test...")

    colorama.init()

    full_sentences = []
    displayed_text = ""
    muted = True  # Start in muted state
    recorder = None
    recorder_lock = threading.Lock()  # To manage access to recorder

    def text_detected(text):
        nonlocal displayed_text
        sentences_with_style = [
            f"{Fore.YELLOW + sentence + Style.RESET_ALL if i % 2 == 0 else Fore.CYAN + sentence + Style.RESET_ALL} "
            for i, sentence in enumerate(full_sentences)
        ]
        new_text = "".join(sentences_with_style).strip() + " " + text if len(sentences_with_style) > 0 else text

        if new_text != displayed_text:
            displayed_text = new_text
            # clear_console()
            print(f"\rLanguage: {recorder.detected_language} (realtime: {recorder.detected_realtime_language})")
            print(displayed_text, end="", flush=True)

    def process_text(text):
        nonlocal full_sentences
        full_sentences.append(text)
        text_detected("")

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': 'tiny.en',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.4,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.2,
        'realtime_model_type': 'tiny.en',
        'on_realtime_transcription_update': text_detected, 
        'silero_deactivity_detection': True,
    }

    # Initialize the recorder and load the model at the start
    try:
        recorder = AudioToTextRecorder(**recorder_config)
        print("Model loaded and recorder initialized.")
    except Exception as e:
        print(f"Failed to initialize the recorder: {e}")
        sys.exit(1)

    def mute_microphone():
        nonlocal muted
        with recorder_lock:
            if not muted:
                muted = True
                print("\nMicrophone muted.")
            else:
                print("\nMicrophone is already muted.")

    def unmute_microphone():
        nonlocal muted
        with recorder_lock:
            if muted:
                muted = False
                print("\nMicrophone unmuted.")
            else:
                print("\nMicrophone is already unmuted.")

    # Register hotkeys
    keyboard.add_hotkey('ctrl+1', mute_microphone, suppress=True)
    keyboard.add_hotkey('ctrl+2', unmute_microphone, suppress=True)

    print("Press Ctrl+1 to mute the microphone and Ctrl+2 to unmute.")
    print("Press Ctrl+C to exit.")

    def recording_loop():
        while True:
            with recorder_lock:
                if not muted:
                    try:
                        recorder.text(process_text)
                    except Exception as e:
                        print(f"\nError during recording: {e}")
                        # Optionally, you can choose to mute the microphone on error
                        muted = True
                # If muted, skip processing
            time.sleep(0.1)  # Small delay to prevent high CPU usage

    # Start the recording loop in a separate thread
    thread = threading.Thread(target=recording_loop, daemon=True)
    thread.start()

    try:
        # Keep the main thread alive to listen for hotkeys
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        with recorder_lock:
            if recorder:
                try:
                    recorder.stop()  # Assuming there's a stop method
                except AttributeError:
                    print("Warning: The recorder does not have a 'stop()' method.")
                except Exception as e:
                    print(f"Failed to stop the recorder: {e}")
        sys.exit(0)

if __name__ == '__main__':
    main()
