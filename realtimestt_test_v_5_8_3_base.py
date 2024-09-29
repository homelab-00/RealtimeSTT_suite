import os
import threading
from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Style
import colorama
import keyboard
import multiprocessing
import warnings
import pyautogui  # Import PyAutoGUI
import logging  # Import logging module

# Initialize colorama
colorama.init()

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logging.txt"),
                        logging.StreamHandler()
                    ])

# Global variables
full_sentences = []
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
        logging.error(f"[PyAutoGUI Error]: {e}")

def unmute_microphone():
    """Unmutes the microphone."""
    global muted
    with mute_lock:
        if not muted:
            logging.info("Microphone is already unmuted.")
            return
        muted = False
        logging.info("Microphone unmuted.")

def mute_microphone():
    """Mutes the microphone."""
    global muted
    with mute_lock:
        if muted:
            logging.info("Microphone is already muted.")
            return
        muted = True
        logging.info("Microphone muted.")

def setup_hotkeys():
    """Sets up global hotkeys for muting and unmuting the microphone."""
    # Register Ctrl+1 to unmute
    keyboard.add_hotkey('ctrl+1', unmute_microphone, suppress=True)
    # Register Ctrl+2 to mute
    keyboard.add_hotkey('ctrl+2', mute_microphone, suppress=True)
    logging.info("Hotkeys set: Ctrl+1 to Unmute, Ctrl+2 to Mute")

def main():
    """Main function to run the Real-Time STT script."""
    clear_console()
    logging.info("Initializing RealTimeSTT...")

    # Capture and handle warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Configuration for the AudioToTextRecorder (these override the default values from audio_recorder.py)
        recorder_config = {
            'spinner': False, # I love the spinner, but I'm pretty sure it's creating some issues
            'model': 'base.en',  # Using the base.en model
            'language': 'en', # Set the language to English
            'device': 'cpu', # Use the CPU instead of GPU for processing (as GPU is not available)
            'debug_mode': False, # Enable debug mode for detailed logs
            'use_main_model_for_realtime': False,  # Better performance when using two models, also see comment below for 'realtime_model_type'
            'ensure_sentence_starting_uppercase': True, # Ensure the first letter of the sentence is capitalized
            'ensure_sentence_ends_with_period': True, # Ensure the sentence ends with a period
            'handle_buffer_overflow': True, # If set, the system will log a warning when an input overflow occurs 
                                            # during recording and remove the data from the buffer.
            'silero_sensitivity': 0.4, # Sensitivity for Silero's voice activity detection ranging from 0 (least sensitive) 
                                       # to 1 (most sensitive). Default is 0.6.
            'webrtc_sensitivity': 2, #  Sensitivity for the WebRTC Voice Activity Detection engine ranging from 0 
                                     # (least aggressive / most sensitive) to 3 (most aggressive, least sensitive). Default is 3.
            'post_speech_silence_duration': 0.4, # Duration in seconds of silence that must follow speech before the recording is
                                                 # considered to be completed. This ensures that any brief pauses during speech
                                                 # don't prematurely end the recording. Default is 0.2
            'min_length_of_recording': 0, # Specifies the minimum time interval in seconds that should exist between the end of one recording
                                          # session and the beginning of another to prevent rapid consecutive recordings. Default is 1.0
            'min_gap_between_recordings': 0, # Specifies the minimum duration in seconds that a recording session should last to ensure meaningful
                                             # audio capture, preventing excessively short or fragmented recordings. Default is 1.0
            'pre_recording_buffer_duration': 0.2, # The time span, in seconds, during which audio is buffered prior to formal recording. This helps
                                                  # counterbalancing the latency inherent in speech activity detection, ensuring no initial audio is 
                                                  # missed. Default is 0.2
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0.2, # Specifies the time interval in seconds after a chunk of audio gets transcribed. Lower values will
                                              # result in more "real-time" (frequent) transcription updates but may increase computational load.
                                              # Default is 0.2
            'realtime_model_type': 'tiny.en',  # Using the tiny.en model for the first pass of real-time transcription,
                                               # and the main model (in this case base.en) for the final transcription.
            'on_realtime_transcription_update': lambda text: text_detected(text, recorder), 
            'silero_deactivity_detection': True, # Enables the Silero model for end-of-speech detection. More robust against background
                                                 # noise. Utilizes additional GPU resources but improves accuracy in noisy environments.
                                                 # When False, uses the default WebRTC VAD, which is more sensitive but may continue 
                                                 # recording longer due to background sounds. Default is False.
        }

        # Initialize the recorder inside the main function
        recorder = AudioToTextRecorder(**recorder_config)

        # Set up global hotkeys in a separate thread
        hotkey_thread = threading.Thread(target=setup_hotkeys, daemon=True)
        hotkey_thread.start()

        try:
            while True:
                # Continuously listen and process audio
                recorder.text(lambda text: process_text(text, recorder))
        except KeyboardInterrupt:
            logging.info("Exiting RealTimeSTT...")
        finally:
            recorder.stop()  # Ensure the recorder is properly stopped

    # Print captured warnings
    for warning in w:
        logging.warning(f"{warning.message}")

    # Print the message after handling warnings
    logging.info("Say something! Press Ctrl+1 to Unmute, Ctrl+2 to Mute.")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows support
    main()