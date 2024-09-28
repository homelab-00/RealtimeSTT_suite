from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Style
import colorama
import os
import keyboard
import threading
import time
import sys

def main():
    print("Initializing RealtimeSTT test...")

    colorama.init()

    full_sentences = []
    displayed_text = ""
    recorder = None  # Initialize recorder as None
    recorder_thread = None  # Thread for the recorder

    # Remove the clear_console function as per your request

    def text_detected(text):
        nonlocal displayed_text
        sentences_with_style = [
            f"{Fore.YELLOW + sentence + Style.RESET_ALL if i % 2 == 0 else Fore.CYAN + sentence + Style.RESET_ALL} "
            for i, sentence in enumerate(full_sentences)
        ]
        new_text = "".join(sentences_with_style).strip() + " " + text if len(sentences_with_style) > 0 else text

        if new_text != displayed_text:
            displayed_text = new_text
            # Removed clear_console()
            print(f"Language: {recorder.detected_language} (realtime: {recorder.detected_realtime_language})")
            print(displayed_text, end="", flush=True)

    def process_text(text):
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
        'realtime_model_type': 'tiny',
        'on_realtime_transcription_update': text_detected, 
        'silero_deactivity_detection': True,
    }

    # Event to handle graceful shutdown
    exit_event = threading.Event()

    def recorder_worker():
        """Worker function to run the recorder."""
        try:
            recorder.text(process_text)
        except Exception as e:
            print(f"\nRecorder encountered an error: {e}")
        finally:
            print("\nRecorder thread exiting.")

    def start_recording():
        nonlocal recorder, recorder_thread
        if recorder is None:
            recorder = AudioToTextRecorder(**recorder_config)
            recorder_thread = threading.Thread(target=recorder_worker, daemon=True)
            recorder_thread.start()
            print("\nRecording started...")
        else:
            print("\nAlready recording.")

    def stop_recording():
        nonlocal recorder, recorder_thread
        if recorder is not None:
            recorder.stop()  # Assuming the recorder has a stop method
            recorder_thread.join()
            recorder = None
            recorder_thread = None
            print("\nRecording stopped.")
        else:
            print("\nNot currently recording.")

    # Register global hotkeys with suppression
    keyboard.add_hotkey('ctrl+1', start_recording, suppress=True)
    keyboard.add_hotkey('ctrl+2', stop_recording, suppress=True)

    print("Press Control+1 to start recording.")
    print("Press Control+2 to stop recording.")
    print("Press Control+C to exit.")

    try:
        while not exit_event.is_set():
            time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        # Clean up resources
        keyboard.unhook_all_hotkeys()
        if recorder is not None:
            recorder.stop()
        sys.exit(0)

if __name__ == '__main__':
    main()
