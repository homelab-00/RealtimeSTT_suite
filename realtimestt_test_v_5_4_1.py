import os
import threading
from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Style
import colorama
import keyboard
import multiprocessing
import warnings
import pyautogui  # Import PyAutoGUI

# Initialize colorama
colorama.init()

# Global variables
full_sentences = []
displayed_text = ""
muted = True  # Initially muted
command_mode = False  # Tracks if the system is in command mode

# Define command and action words
command_words = [
    "luna",   # Standard
    "lunna",  # Slight variation
    "lunnaa", # Extended vowel
    "lina",   # Alternative spelling
    "lyna",   # Phonetic variation
    "louna",  # Different vowel placement
    "loonah", # Different ending
    # Add more as needed based on testing
]
action_words = ["delete"]  # Expand this list with more actions in the future

# Lock for thread-safe access to 'muted' and 'command_mode'
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
    global muted, command_mode
    with mute_lock:
        if muted:
            return  # Do not process text when muted

    # Append the new text to full_sentences and update display
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
    
    # Convert text to lowercase for case-insensitive matching
    lower_text = text.lower()
    
    if not command_mode:
        # Check if any command word is in the transcribed text
        if any(cmd in lower_text for cmd in command_words):
            command_mode = True
            print("\nCommand mode activated. Awaiting action word...")
    else:
        # In command mode, check for action words
        if any(action in lower_text for action in action_words):
            # Execute corresponding action
            for action in action_words:
                if action in lower_text:
                    execute_action(action)
                    break  # Execute only the first matched action
            # Reset to normal mode after action execution
            command_mode = False
            print("\nReturned to normal mode.")
        else:
            # If no action word is detected, ignore the input
            print("\nAction word not recognized. Returning to normal mode.")
            command_mode = False

def execute_action(action):
    """Executes the specified action."""
    if action == "delete":
        print("Executing 'delete' action: Deleting the last word.")
        try:
            # Simulate Ctrl + Backspace to delete the last word
            pyautogui.hotkey('ctrl', 'backspace')
            # Optionally, provide feedback
            print("Last word deleted.")
        except Exception as e:
            print(f"[PyAutoGUI Error]: {e}")
    else:
        print(f"Action '{action}' is not defined.")

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

def main():
    """Main function to run the Real-Time STT script."""
    clear_console()
    print("Initializing RealTimeSTT...")

    # Set up global hotkeys in a separate thread
    hotkey_thread = threading.Thread(target=setup_hotkeys, daemon=True)
    hotkey_thread.start()

    # Capture and handle warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Configuration for the AudioToTextRecorder
        recorder_config = {
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
            'on_realtime_transcription_update': lambda text: process_text(text, recorder), 
            'silero_deactivity_detection': True,
            # Removed wakeword parameters
        }

        # Initialize the recorder inside the main function
        recorder = AudioToTextRecorder(**recorder_config)

        try:
            while True:
                # Continuously listen and process audio
                recorder.text(lambda text: process_text(text, recorder))
        except KeyboardInterrupt:
            print("\nExiting RealTimeSTT...")
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
