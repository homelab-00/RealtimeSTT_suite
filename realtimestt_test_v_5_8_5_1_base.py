import os
import threading

import sys
sys.path.insert(0, './')  # This assumes the file is in the same directory
from audio_recorder_v_1_1 import AudioToTextRecorder

from colorama import Fore, Style
import colorama
import keyboard
import multiprocessing
import warnings
import pyautogui
import sys

import logging
from logging.handlers import RotatingFileHandler

# Initialize logging
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Remove all existing handlers to prevent duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

# Create a RotatingFileHandler to write logs to debug.log
file_handler = RotatingFileHandler(
    'debug.log',
    maxBytes=10_000_000,  # 10 MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)  # Ensure all DEBUG and above logs are captured

# Define the log format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Initialize colorama
colorama.init()

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
            logger.debug("Text detected but microphone is muted")
            return  # Do not process text when muted

    logger.debug(f"Text detected: {text}")

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
            logger.debug("Text processed but microphone is muted")
            return
    logger.info(f"Processing text: {text}")
    full_sentences.append(text)
    text_detected("", recorder)
    
    try:
        logger.debug("Attempting to type text with PyAutoGUI")
        pyautogui.sleep(0.2)
        pyautogui.write(text, interval=0.02)
        logger.debug("Text typed successfully")
    except Exception as e:
        logger.error(f"PyAutoGUI Error: {e}")

def unmute_microphone():
    """Unmutes the microphone."""
    global muted
    with mute_lock:
        if not muted:
            logger.info("Microphone is already unmuted")
            return
        muted = False
        logger.info("Microphone unmuted")

def mute_microphone():
    """Mutes the microphone."""
    global muted
    with mute_lock:
        if muted:
            logger.info("Microphone is already muted")
            return
        muted = True
        logger.info("Microphone muted")

def setup_hotkeys():
    """Sets up global hotkeys for muting and unmuting the microphone."""
    logger.info("Setting up hotkeys")
    keyboard.add_hotkey('ctrl+1', unmute_microphone, suppress=True)
    keyboard.add_hotkey('ctrl+2', mute_microphone, suppress=True)
    logger.info("Hotkeys set: Ctrl+1 to Unmute, Ctrl+2 to Mute")

def main():
    """Main function to run the Real-Time STT script."""
    logger.info("Initializing RealTimeSTT...")

    # Capture and handle warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Configuration for the AudioToTextRecorder (these override the default values from audio_recorder.py)
        recorder_config = {
            'spinner': False, # I love the spinner, but I'm pretty sure it's creating some issues
            'model': 'base.en',  # Using the base.en model
            'language': 'en', # Set the language to English
            'device': 'cpu', # Use the CPU instead of GPU for processing (as GPU is not available)
            'debug_mode': True, # Enable debug mode for detailed logs
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
            'min_length_of_recording': 0.3, # Specifies the minimum time interval in seconds that should exist between the end of one recording
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

        logger.debug(f"Recorder configuration: {recorder_config}")

        recorder = AudioToTextRecorder(**recorder_config)
        logger.info("AudioToTextRecorder initialized")

        hotkey_thread = threading.Thread(target=setup_hotkeys, daemon=True)
        hotkey_thread.start()
        logger.info("Hotkey thread started")

        try:
            logger.info("Entering main loop")
            while True:
                recorder.text(lambda text: process_text(text, recorder))
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, exiting RealTimeSTT...")
        finally:
            logger.info("Stopping recorder")
            recorder.stop()

    for warning in w:
        logger.warning(f"Warning during execution: {warning.message}")

    logger.info("RealTimeSTT execution completed")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    logger.info("Starting main function")
    main()