import os
import threading
from colorama import Fore, Style
import colorama
import keyboard
import multiprocessing
import warnings
import pyautogui  # Import PyAutoGUI
from collections import deque  # Import deque for efficient buffer management

import sys
sys.path.insert(0, './')  # This assumes the file is in the same directory
from audio_recorder_v_1_1 import AudioToTextRecorder

# Initialize colorama
colorama.init()

# Global variables
max_buffer_size = 20  # Maximum number of sentences in the buffer
full_sentences = deque(maxlen=max_buffer_size)  # Use deque to automatically discard old sentences
displayed_text = ""
muted = True  # Initially muted

# Lock for thread-safe access to 'muted'
mute_lock = threading.Lock()

def clear_console():
    """Clears the terminal console."""
    os.system('clear' if os.name == 'posix' else 'cls')

def text_detected(text, recorder):
    """Callback function to handle detected text."""
    global displayed_text
    with mute_lock:
        if muted:
            return  # Do not process text when muted

    # Apply alternating colors to sentences
    sentences_with_style = [
        f"{Fore.YELLOW + sentence + Style.RESET_ALL if i % 2 == 0 else Fore.CYAN + sentence + Style.RESET_ALL} "
        for i, sentence in enumerate(full_sentences)
    ]
    new_text = "".join(sentences_with_style).strip() + " " + text if full_sentences else text

    if new_text != displayed_text:
        displayed_text = new_text
        clear_console()
        print(f"Language: {recorder.detected_language} (Realtime: {recorder.detected_realtime_language})")
        print(displayed_text, end="", flush=True)

def process_text(text, recorder):
    """Processes and appends the transcribed text."""
    with mute_lock:
        if muted:
            return
    if text:  # Ensure that empty strings are not appended
        full_sentences.append(text)
        text_detected("", recorder)
        
        # Use PyAutoGUI to type the new transcription into the active window
        try:
            # Add a slight delay to ensure the active window is ready to receive input
            pyautogui.sleep(0.2)  # Reduced delay
            # Type the text with a faster typing speed
            pyautogui.write(text, interval=0.02)  # Increased typing speed
            # pyautogui.press('enter')  # Optional: Press Enter after typing
        except Exception as e:
            print(f"\n[PyAutoGUI Error]: {e}")

def unmute_microphone():
    """Unmutes the microphone."""
    global muted
    with mute_lock:
        if not muted:
            print("\nMicrophone is already unmuted.")
            return
        muted = False
        print("\nMicrophone unmuted.")

def mute_microphone():
    """Mutes the microphone."""
    global muted
    with mute_lock:
        if muted:
            print("\nMicrophone is already muted.")
            return
        muted = True
        print("\nMicrophone muted.")

def setup_hotkeys():
    """Sets up global hotkeys for muting and unmuting the microphone."""
    # Register Ctrl+1 to unmute
    keyboard.add_hotkey('ctrl+1', unmute_microphone, suppress=True)
    # Register Ctrl+2 to mute
    keyboard.add_hotkey('ctrl+2', mute_microphone, suppress=True)
    print("Hotkeys set: Ctrl+1 to Unmute, Ctrl+2 to Mute")

def handle_exit(recorder):
    """Handles graceful exit on Ctrl+C."""
    print("\nExiting RealTimeSTT...")
    recorder.stop()
    os._exit(0)  # Force exit to ensure all threads are terminated

def main():
    """Main function to run the Real-Time STT script."""
    clear_console()
    print("Initializing RealTimeSTT...")

    # Capture and handle warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Configuration for the AudioToTextRecorder
        recorder_config = {
            'spinner': False,
            'model': 'base.en',  # Using the base.en model
            'silero_sensitivity': 0.4,
            'webrtc_sensitivity': 2,
            'post_speech_silence_duration': 0.4,
            'min_length_of_recording': 0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0.5,  # Increased pause to handle longer speech
            'realtime_model_type': 'base.en',  # Ensure the model type matches
            'on_realtime_transcription_update': lambda text: text_detected(text, recorder), 
            'silero_deactivity_detection': True,
            # Additional parameters to handle longer speech
            'handle_buffer_overflow': False,  # We'll manage the buffer ourselves
            'pre_recording_buffer_duration': 0.5,  # Increased buffer to handle longer speech
            'on_realtime_transcription_stabilized': lambda text: process_text(text, recorder),
        }

        # Initialize the recorder inside the main function
        recorder = AudioToTextRecorder(**recorder_config)

        # Set up global hotkeys in a separate thread
        hotkey_thread = threading.Thread(target=setup_hotkeys, daemon=True)
        hotkey_thread.start()

        # Set up graceful exit handling
        exit_thread = threading.Thread(target=keyboard.wait, daemon=True)
        exit_thread.start()

        # Register Ctrl+C to trigger handle_exit
        try:
            while True:
                # Continuously listen and process audio
                recorder.text(lambda text: process_text(text, recorder))
        except KeyboardInterrupt:
            handle_exit(recorder)
        finally:
            recorder.stop()  # Ensure the recorder is properly stopped

    # Print captured warnings
    for warning in w:
        print(f"{warning.message}")

    # Print the message after handling warnings
    print("Say something! Press Ctrl+1 to Unmute, Ctrl+2 to Mute.", end="", flush=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows support
    main()
