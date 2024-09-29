import os
import threading
from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Style
import colorama
import keyboard
import multiprocessing
import warnings
import pyautogui  # Import PyAutoGUI
import pygetwindow as gw  # Import PyGetWindow for window management
import time
import logging

# Initialize colorama
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("realtimestt.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
full_sentences = []
displayed_text = ""
muted = True  # Initially muted
command_mode = False  # Tracks if the system is in command mode

# Define command and action words
command_words = [
    "luna", "lunna", "lunnaa", "lina", "lyna", "louna", "loonah", "loona", "loonya"
    # Add more unique variations as needed
]
action_words = ["delete"]  # Expand this list with more actions in the future

# Lock for thread-safe access to 'muted' and 'command_mode'
state_lock = threading.Lock()

def clear_console():
    """Clears the terminal console."""
    os.system('clear' if os.name == 'posix' else 'cls')

def focus_window(window_title):
    """
    Focuses the window with the given title.

    Parameters:
        window_title (str): The title of the window to focus.
    """
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        window.activate()
        logger.info(f"Focused window: {window_title}")
    except IndexError:
        logger.warning(f"Window titled '{window_title}' not found. Ensure the application is open and the title is correct.")

def text_detected(text, recorder):
    """Callback function to handle detected text."""
    global displayed_text
    with state_lock:
        if muted:
            return  # Do not process text when muted

    # Append the new text to full_sentences and update display
    full_sentences.append(text)

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

def execute_action(action):
    """Executes the specified action."""
    if action == "delete":
        logger.info("Executing 'delete' action: Deleting the last word.")
        try:
            # Simulate Ctrl + Backspace to delete the last word
            pyautogui.hotkey('ctrl', 'backspace')
            # Optionally, provide feedback
            print("Last word deleted.")
            logger.info("Last word deleted successfully.")
        except Exception as e:
            logger.error(f"[PyAutoGUI Error]: {e}")
            print(f"[PyAutoGUI Error]: {e}")
    else:
        logger.warning(f"Action '{action}' is not defined.")
        print(f"Action '{action}' is not defined.")

def process_text(text, recorder):
    """Processes and appends the transcribed text."""
    global muted, command_mode
    with state_lock:
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
    
    with state_lock:
        if not command_mode:
            # Check if any command word is in the transcribed text
            if any(cmd in lower_text for cmd in command_words):
                command_mode = True
                print("\nCommand mode activated. Awaiting action word...")
                logger.info("Command mode activated.")
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
                logger.info("Returned to normal mode after executing action.")
            else:
                # If no action word is detected, ignore the input and reset
                print("\nAction word not recognized. Returning to normal mode.")
                logger.warning("Unrecognized action word. Returning to normal mode.")
                command_mode = False

def unmute_microphone():
    """Unmutes the microphone."""
    global muted
    with state_lock:
        if not muted:
            logger.info("Microphone is already unmuted.")
            print("\nMicrophone is already unmuted.")
            return
        muted = False
        logger.info("Microphone unmuted.")
        print("\nMicrophone unmuted.")

def mute_microphone():
    """Mutes the microphone."""
    global muted
    with state_lock:
        if muted:
            logger.info("Microphone is already muted.")
            print("\nMicrophone is already muted.")
            return
        muted = True
        logger.info("Microphone muted.")
        print("\nMicrophone muted.")

def setup_hotkeys():
    """Sets up global hotkeys for muting and unmuting the microphone."""
    # Register Ctrl+1 to unmute
    keyboard.add_hotkey('ctrl+1', unmute_microphone, suppress=True)
    # Register Ctrl+2 to mute
    keyboard.add_hotkey('ctrl+2', mute_microphone, suppress=True)
    logger.info("Hotkeys set: Ctrl+1 to Unmute, Ctrl+2 to Mute")
    print("Hotkeys set: Ctrl+1 to Unmute, Ctrl+2 to Mute")
    print("Available Commands:")
    print(" - Say 'Luna' followed by 'delete' to delete the last word.")

def main():
    """Main function to run the Real-Time STT script."""
    clear_console()
    print("Initializing RealTimeSTT...")
    logger.info("RealtimeSTT initialization started.")

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
        try:
            recorder = AudioToTextRecorder(**recorder_config)
            logger.info("AudioToTextRecorder initialized successfully.")
            print("Model loaded successfully. Ready to start transcribing.")
        except Exception as e:
            logger.error(f"Failed to initialize AudioToTextRecorder: {e}")
            print(f"Failed to initialize AudioToTextRecorder: {e}")
            return

        try:
            print("Say something! Press Ctrl+1 to Unmute, Ctrl+2 to Mute.")
            while True:
                # Continuously listen and process audio
                recorder.text(lambda text: process_text(text, recorder))
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting RealTimeSTT...")
            print("\nExiting RealTimeSTT...")
        finally:
            recorder.stop()  # Ensure the recorder is properly stopped
            logger.info("AudioToTextRecorder stopped.")

    # Print captured warnings
    for warning in w:
        logger.warning(f"{warning.message}")
        print(f"{warning.message}")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows support
    main()
