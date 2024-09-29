import os
import threading

from audio_recorder_v_1_1 import AudioToTextRecorder

from colorama import Fore, Style
import colorama
import keyboard
import multiprocessing
import warnings
import pyautogui
import logging
import sys

# Configure logging to save to 'app.log' only
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    filename='app.log',  # Log file name
    filemode='w'  # 'w' to overwrite the log file each time, 'a' to append
)

# Create a logger instance
logger = logging.getLogger(__name__)

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
    try:
        os.system('clear' if os.name == 'posix' else 'cls')
        logger.debug("Console cleared.")
    except Exception as e:
        logger.error(f"Error clearing console: {e}")

def text_detected(text, recorder):
    """Callback function to handle detected text."""
    global displayed_text
    with mute_lock:
        if muted:
            logger.debug("Text detected but microphone is muted. Ignoring text.")
            return  # Do not process text when muted

    # Apply alternating colors to sentences
    try:
        sentences_with_style = [
            f"{Fore.YELLOW + sentence + Style.RESET_ALL if i % 2 == 0 else Fore.CYAN + sentence + Style.RESET_ALL} "
            for i, sentence in enumerate(full_sentences)
        ]
        new_text = "".join(sentences_with_style).strip() + " " + text if full_sentences else text

        if new_text != displayed_text:
            displayed_text = new_text
            clear_console()
            logger.info(f"Language: {recorder.detected_language} (Realtime: {recorder.detected_realtime_language})")
            # Since we are not printing to console, we can log the displayed text
            logger.info(f"Displayed Text: {displayed_text}")
    except Exception as e:
        logger.error(f"Error in text_detected: {e}")

def process_text(text, recorder):
    """Processes and appends the transcribed text."""
    with mute_lock:
        if muted:
            logger.debug("Process_text called but microphone is muted. Ignoring text.")
            return
    try:
        full_sentences.append(text)
        logger.info(f"Appended new text: {text}")
        text_detected("", recorder)

        # Use PyAutoGUI to type the new transcription into the active window
        try:
            # Add a slight delay to ensure the active window is ready to receive input
            pyautogui.sleep(0.2)  # Reduced delay
            logger.debug("PyAutoGUI sleep for 0.2 seconds before typing.")
            # Type the text with a faster typing speed
            pyautogui.write(text, interval=0.02)  # Increased typing speed
            logger.info(f"Typed text using PyAutoGUI: {text}")
            # pyautogui.press('enter')  # Optional: Press Enter after typing
        except Exception as e:
            logger.error(f"PyAutoGUI encountered an error: {e}")
    except Exception as e:
        logger.error(f"Error in process_text: {e}")

def unmute_microphone():
    """Unmutes the microphone."""
    global muted
    with mute_lock:
        if not muted:
            logger.info("Attempted to unmute, but microphone is already unmuted.")
            return
        muted = False
        logger.info("Microphone has been unmuted.")

def mute_microphone():
    """Mutes the microphone."""
    global muted
    with mute_lock:
        if muted:
            logger.info("Attempted to mute, but microphone is already muted.")
            return
        muted = True
        logger.info("Microphone has been muted.")

def setup_hotkeys():
    """Sets up global hotkeys for muting and unmuting the microphone."""
    try:
        # Register Ctrl+1 to unmute
        keyboard.add_hotkey('ctrl+1', unmute_microphone, suppress=True)
        # Register Ctrl+2 to mute
        keyboard.add_hotkey('ctrl+2', mute_microphone, suppress=True)
        logger.info("Hotkeys set: Ctrl+1 to Unmute, Ctrl+2 to Mute")
    except Exception as e:
        logger.error(f"Error setting up hotkeys: {e}")

def main():
    """Main function to run the Real-Time STT script."""
    try:
        clear_console()
        logger.info("Initializing RealTimeSTT...")

        # Capture and handle warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Configuration for the AudioToTextRecorder
            recorder_config = {
                'spinner': False,  # I love the spinner, but I'm pretty sure it's creating some issues
                'model': 'base.en',  # Using the base.en model
                'language': 'en',  # Set the language to English
                'device': 'cpu',  # Use the CPU instead of GPU for processing (as GPU is not available)
                'debug_mode': False,  # Enable debug mode for detailed logs
                'use_main_model_for_realtime': False,  # Better performance when using two models
                'ensure_sentence_starting_uppercase': True,  # Ensure the first letter of the sentence is capitalized
                'ensure_sentence_ends_with_period': True,  # Ensure the sentence ends with a period
                'handle_buffer_overflow': True,  # Log a warning and remove buffer overflow
                'silero_sensitivity': 0.4,  # Sensitivity for Silero's VAD
                'webrtc_sensitivity': 2,  # Sensitivity for WebRTC VAD
                'post_speech_silence_duration': 0.4,  # Silence duration after speech
                'min_length_of_recording': 0.3,  # Minimum interval between recordings
                'min_gap_between_recordings': 0,  # Minimum duration of a recording session
                'pre_recording_buffer_duration': 0.2,  # Buffer duration before recording
                'enable_realtime_transcription': True,
                'realtime_processing_pause': 0.2,  # Pause interval after transcription
                'realtime_model_type': 'tiny.en',  # Model type for real-time transcription
                'on_realtime_transcription_update': lambda text: text_detected(text, recorder),
                'silero_deactivity_detection': True,  # Enable Silero VAD
            }

            # Initialize the recorder inside the main function
            recorder = AudioToTextRecorder(**recorder_config)
            logger.info("AudioToTextRecorder initialized with provided configuration.")

            # Set up global hotkeys in a separate daemon thread
            hotkey_thread = threading.Thread(target=setup_hotkeys, daemon=True)
            hotkey_thread.start()
            logger.debug("Hotkey thread started.")

            try:
                logger.info("Starting continuous audio listening and processing.")
                while True:
                    # Continuously listen and process audio
                    recorder.text(lambda text: process_text(text, recorder))
            except KeyboardInterrupt:
                logger.info("Received KeyboardInterrupt. Initiating shutdown...")
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
            finally:
                logger.info("Stopping AudioToTextRecorder...")
                recorder.stop()  # Ensure the recorder is properly stopped
                logger.info("AudioToTextRecorder stopped.")
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
    finally:
        # Print captured warnings to the log
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                for warning in w:
                    logger.warning(f"Warning captured: {warning.message}")
        except Exception as e:
            logger.error(f"Error while logging warnings: {e}")

        logger.info("Say something! Press Ctrl+1 to Unmute, Ctrl+2 to Mute.")

if __name__ == '__main__':
    try:
        multiprocessing.freeze_support()  # For Windows support
        logger.info("Starting the RealTimeSTT application.")
        main()
        logger.info("RealTimeSTT application has exited successfully.")
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        sys.exit(1)
