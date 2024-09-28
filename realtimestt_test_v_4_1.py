import os
import time
import threading
from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Style
import colorama
import pyautogui
import keyboard

# Initialize colorama for colored terminal output
colorama.init()

# Global variables
is_muted = False
mute_lock = threading.Lock()
full_sentences = []
displayed_text = ""
recorder = None  # Declare recorder as a global variable

def unmute_microphone():
    """Unmutes the microphone."""
    global is_muted
    with mute_lock:
        is_muted = False
    print(Fore.GREEN + "Microphone unmuted" + Style.RESET_ALL)

def mute_microphone():
    """Mutes the microphone."""
    global is_muted
    with mute_lock:
        is_muted = True
    print(Fore.RED + "Microphone muted" + Style.RESET_ALL)

def setup_hotkeys():
    """
    Sets up global hotkeys:
    - Ctrl+1 to unmute the microphone
    - Ctrl+2 to mute the microphone
    """
    print("Setting up global hotkeys: Ctrl+1 to Unmute, Ctrl+2 to Mute")
    try:
        keyboard.add_hotkey('ctrl+1', unmute_microphone, suppress=True)
        keyboard.add_hotkey('ctrl+2', mute_microphone, suppress=True)
        print("Global hotkeys are now active.")
        keyboard.wait()  # Keep the thread alive to listen for hotkeys
    except Exception as e:
        print(Fore.RED + f"Failed to set up hotkeys: {e}" + Style.RESET_ALL)

def send_text_to_active_window(text):
    """
    Sends the transcribed text to the active window using pyautogui.
    
    Args:
        text (str): The text to send to the active window.
    """
    try:
        # Optionally, you can add a short delay to ensure the window is ready to receive input
        time.sleep(0.1)
        pyautogui.typewrite(text + ' ', interval=0.05)  # Types the text with a slight interval between keystrokes
        print(Fore.CYAN + f"Sent to active window: {text}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error sending text to active window: {e}" + Style.RESET_ALL)

def process_text(text):
    """
    Processes the transcribed text:
    - Appends it to the list of full sentences
    - Updates the displayed text
    - Sends the text to the active window if not muted
    
    Args:
        text (str): The newly transcribed text segment.
    """
    with mute_lock:
        if not is_muted:
            full_sentences.append(text)
            text_detected("")  # Pass an empty string to avoid duplication
        else:
            print(Fore.YELLOW + "Microphone is muted. Ignoring transcription." + Style.RESET_ALL)

def text_detected(text):
    """
    Updates the terminal display with the new text and sends it to the active window.
    
    Args:
        text (str): The newly transcribed text segment.
    """
    global displayed_text
    sentences_with_style = [
        f"{Fore.YELLOW + sentence + Style.RESET_ALL if i % 2 == 0 else Fore.CYAN + sentence + Style.RESET_ALL} "
        for i, sentence in enumerate(full_sentences)
    ]
    new_text = "".join(sentences_with_style).strip()
    # In the original script, text_detected("") was called, so we don't need to add 'text'

    if new_text != displayed_text:
        displayed_text = new_text
        print(f"\nLanguage: {recorder.detected_language} (Realtime: {recorder.detected_realtime_language})")
        print(displayed_text, end="", flush=True)
        # Send only the latest text to active window
        if text.strip():
            send_text_to_active_window(text.strip())

def main():
    """
    Main function to initialize components and start the transcription loop.
    """
    global recorder
    print(Fore.MAGENTA + "Initializing RealtimeSTT transcription script..." + Style.RESET_ALL)

    # Start hotkeys in a separate daemon thread
    hotkey_thread = threading.Thread(target=setup_hotkeys, daemon=True)
    hotkey_thread.start()

    # Recorder configuration
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
        'on_realtime_transcription_update': process_text, 
        'silero_deactivity_detection': True,
    }

    try:
        print(Fore.BLUE + "Initializing AudioToTextRecorder with the following configuration:" + Style.RESET_ALL)
        for key, value in recorder_config.items():
            print(f"  {key}: {value}")
        
        recorder = AudioToTextRecorder(**recorder_config)
        print(Fore.GREEN + "AudioToTextRecorder initialized successfully." + Style.RESET_ALL)

        print(Fore.YELLOW + "Ready for transcription. Press Ctrl+1 to Unmute and Ctrl+2 to Mute the microphone." + Style.RESET_ALL)
        print(Fore.YELLOW + "Start speaking..." + Style.RESET_ALL)

        # Start the transcription loop
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        print(Fore.MAGENTA + "\nTranscription stopped by user." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
    finally:
        if recorder:
            try:
                recorder.stop()  # Assuming AudioToTextRecorder has a stop method
                print(Fore.GREEN + "AudioToTextRecorder has been stopped." + Style.RESET_ALL)
            except AttributeError:
                print(Fore.YELLOW + "Recorder does not have a stop method. Exiting." + Style.RESET_ALL)

if __name__ == '__main__':
    main()
