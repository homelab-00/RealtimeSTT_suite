import os
import threading
from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Style
import colorama
import keyboard
import multiprocessing
import warnings
import pyautogui  # Import PyAutoGUI
import time

# Initialize colorama
colorama.init()

# Global variables
full_sentences = []
displayed_text = ""
muted = False  # Initially unmuted
action_mode = False  # Indicates if waiting for an action word

# Lock for thread-safe access to shared variables
state_lock = threading.Lock()

# Define wake word and action word
COMMAND_WAKE_WORD = "samantha"  # This is the command word to activate action listening
ACTION_WORD = "delete"           # This is the action to perform

def clear_console():
    """Clears the terminal console."""
    os.system('clear' if os.name == 'posix' else 'cls')

def text_detected_transcription(text, recorder):
    """Callback function to handle detected text for transcription."""
    global displayed_text
    with state_lock:
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

def process_transcription(text, recorder):
    """Processes and appends the transcribed text."""
    global full_sentences
    with state_lock:
        if muted:
            return
    full_sentences.append(text)
    text_detected_transcription("", recorder)
    
    # Use PyAutoGUI to type the new transcription into the active window
    try:
        # Add a slight delay to ensure the active window is ready to receive input
        pyautogui.sleep(0.1)  # Reduced delay
        # Type the text with a faster typing speed
        pyautogui.write(text, interval=0.02)  # Increased typing speed
    except Exception as e:
        print(f"\n[PyAutoGUI Error]: {e}")

def delete_last_word():
    """Deletes the last word typed in the active window using PyAutoGUI."""
    try:
        # Move the cursor to the end if not already
        pyautogui.sleep(0.1)
        # Press backspace multiple times to delete the last word
        # Assuming words are separated by spaces
        for _ in range(20):  # Maximum of 20 backspaces to ensure deletion
            pyautogui.press('backspace')
            pyautogui.sleep(0.01)
            # Optionally, break early if deletion is detected
            # This requires more sophisticated detection which is not implemented here
    except Exception as e:
        print(f"\n[PyAutoGUI Delete Error]: {e}")

def unmute_microphone():
    """Unmutes the microphone."""
    global muted
    with state_lock:
        if not muted:
            print("\nMicrophone is already unmuted.")
            return
        muted = False
        print("\nMicrophone unmuted.")

def mute_microphone():
    """Mutes the microphone."""
    global muted
    with state_lock:
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

def transcription_thread_function():
    """Function to run the real-time transcription."""
    global full_sentences, displayed_text

    def on_transcription_update(text):
        process_transcription(text, recorder_transcription)

    clear_console()
    print("Initializing RealTimeSTT Transcription...")

    # Capture and handle warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Configuration for the AudioToTextRecorder for transcription
        recorder_config_transcription = {
            'spinner': False,
            'model': 'tiny.en',  # Using the tiny.en model
            'silero_sensitivity': 0.4,
            'webrtc_sensitivity': 2,
            'post_speech_silence_duration': 0.4,
            'min_length_of_recording': 0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0.2,
            'realtime_model_type': 'tiny.en',  # Ensure the model type matches
            'on_realtime_transcription_update': on_transcription_update, 
            'silero_deactivity_detection': True,
        }

        # Initialize the recorder for transcription
        with AudioToTextRecorder(**recorder_config_transcription) as recorder_transcription:
            # Set up global hotkeys in the main thread
            setup_hotkeys()

            try:
                while True:
                    # Continuously listen and process audio
                    recorder_transcription.text(on_transcription_update)
            except KeyboardInterrupt:
                print("\nExiting RealTimeSTT Transcription...")
            finally:
                recorder_transcription.stop()  # Ensure the recorder is properly stopped

    # Print captured warnings
    for warning in w:
        print(f"{warning.message}")

    # Print the message after handling warnings
    print("Say something! Press Ctrl+1 to Unmute, Ctrl+2 to Mute.", end="", flush=True)

def command_listener_thread_function():
    """Function to run the command word listener."""
    def on_wakeword_detected():
        global muted, action_mode
        print(f"\nCommand word '{COMMAND_WAKE_WORD}' detected.")
        with state_lock:
            muted = True  # Mute transcription
            action_mode = True  # Enter action listening mode
        # Allow some time for 'samantha' to be typed before deleting
        time.sleep(0.5)
        delete_last_word()
        with state_lock:
            muted = False  # Unmute transcription
            action_mode = False  # Exit action listening mode
        print("Action performed: Last word deleted. Microphone unmuted.")

    def on_recording_start():
        print("Recording for command word...")

    def on_recording_stop():
        print("Stopped recording for command word.")

    def on_wakeword_timeout():
        pass  # Optional: Handle timeout if needed

    def on_wakeword_detection_start():
        print(f"\nListening for command word '{COMMAND_WAKE_WORD}'.")

    def on_vad_detect_start():
        pass  # Optional: Handle voice activity detection start

    def dummy_text_detected(text):
        """Dummy callback since we handle wake words."""
        pass

    print("Initializing Command Listener...")

    # Configuration for the AudioToTextRecorder for command listening
    recorder_config_command = {
        'spinner': False,
        'model': 'base.en',
        'language': 'en', 
        'wakeword_backend': 'oww',
        'wake_words_sensitivity': 0.35,
        # Load your wake word models here
        'openwakeword_model_paths': "suh_man_tuh.onnx,suh_mahn_thuh.onnx",  # Update with your actual model paths
        'on_wakeword_detected': on_wakeword_detected,
        'on_recording_start': on_recording_start,
        'on_recording_stop': on_recording_stop,
        'on_wakeword_timeout': on_wakeword_timeout,
        'on_wakeword_detection_start': on_wakeword_detection_start,
        'on_vad_detect_start': on_vad_detect_start,
        'wake_word_buffer_duration': 1,
    }

    with AudioToTextRecorder(**recorder_config_command) as recorder_command:
        try:
            while True:
                # Continuously listen and process audio
                recorder_command.text(dummy_text_detected)
        except KeyboardInterrupt:
            print("\nExiting Command Listener...")
        finally:
            recorder_command.stop()  # Ensure the recorder is properly stopped

def main():
    """Main function to run both transcription and command listener."""
    # Create threads for transcription and command listener
    transcription_thread = threading.Thread(target=transcription_thread_function, daemon=True)
    command_listener_thread = threading.Thread(target=command_listener_thread_function, daemon=True)

    # Start both threads
    transcription_thread.start()
    command_listener_thread.start()

    # Keep the main thread alive to allow daemon threads to run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting the integrated STT and Command Listener script...")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows support
    main()
