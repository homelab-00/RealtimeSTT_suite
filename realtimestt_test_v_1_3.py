from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Style
import colorama
import os
import keyboard  # For global hotkeys
import threading
import sys

def main():
    print("Initializing RealtimeSTT test...")

    colorama.init()

    full_sentences = []
    displayed_text = ""
    recording = False  # Flag to indicate recording state
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

    def start_recording():
        nonlocal recording, recorder
        with recorder_lock:
            if not recording:
                recorder = AudioToTextRecorder(**recorder_config)
                recording = True
                print("\nRecording started...")

    def stop_recording():
        nonlocal recording, recorder
        with recorder_lock:
            if recording and recorder:
                recorder.stop()  # Assuming there's a stop method
                recording = False
                print("\nRecording stopped.")

    # Register hotkeys
    keyboard.add_hotkey('ctrl+1', start_recording, suppress=True)
    keyboard.add_hotkey('ctrl+2', stop_recording, suppress=True)

    print("Press Ctrl+1 to start recording and Ctrl+2 to stop recording.")
    print("Press Ctrl+C to exit.")

    def recording_loop():
        while True:
            with recorder_lock:
                if recording and recorder:
                    try:
                        recorder.text(process_text)
                    except Exception as e:
                        print(f"\nError during recording: {e}")
                        recorder = None
                        recording = False
            keyboard.wait('ctrl+1')  # Small wait to prevent tight loop

    # Run the recording loop in a separate thread
    thread = threading.Thread(target=recording_loop, daemon=True)
    thread.start()

    try:
        # Keep the main thread alive to listen for hotkeys
        keyboard.wait()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        with recorder_lock:
            if recorder:
                recorder.stop()  # Assuming there's a stop method
        sys.exit(0)

if __name__ == '__main__':
    main()
