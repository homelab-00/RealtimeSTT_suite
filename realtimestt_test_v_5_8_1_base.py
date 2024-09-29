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

# Lock for thread-safe access to 'muted' and 'is_manual_recording'
mute_lock = threading.Lock()
manual_recording_lock = threading.Lock()

# Flag to indicate if manual recording is active
is_manual_recording = False

def clear_console():
    """Clears the terminal console."""
    os.system('clear' if os.name == 'posix' else 'cls')

def text_detected(text, recorder):
    """Callback function to handle detected text."""
    global displayed_text
    with mute_lock:
        if muted:
            return  # Do not process text when muted

    with manual_recording_lock:
        if is_manual_recording:
            return  # Do not process live transcription during manual recording

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
    with manual_recording_lock:
        if is_manual_recording:
            return  # Do not process live transcription during manual recording
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

def start_manual_recording(recorder):
    """Starts manual audio recording."""
    global is_manual_recording
    with manual_recording_lock:
        if is_manual_recording:
            print("\nManual recording is already in progress.")
            return
        is_manual_recording = True
        print("\nManual recording started.")
        try:
            recorder.start()
        except Exception as e:
            print(f"\n[Manual Recording Error]: {e}")
            is_manual_recording = False

def stop_manual_recording(recorder):
    """Stops manual audio recording and processes the audio."""
    global is_manual_recording
    with manual_recording_lock:
        if not is_manual_recording:
            print("\nNo manual recording is in progress.")
            return
        try:
            transcribed_text = recorder.stop()
            print("\nManual recording stopped. Transcribed Text:")
            print(transcribed_text)
            # Optionally, you can process the transcribed text further here
        except Exception as e:
            print(f"\n[Manual Recording Stop Error]: {e}")
        finally:
            is_manual_recording = False

def setup_hotkeys(recorder):
    """Sets up global hotkeys for muting/unmuting and manual recording."""
    # Register Ctrl+1 to unmute
    keyboard.add_hotkey('ctrl+1', unmute_microphone, suppress=True)
    # Register Ctrl+2 to mute
    keyboard.add_hotkey('ctrl+2', mute_microphone, suppress=True)
    # Register Ctrl+3 to start manual recording
    keyboard.add_hotkey('ctrl+3', lambda: start_manual_recording(recorder), suppress=True)
    # Register Ctrl+4 to stop manual recording
    keyboard.add_hotkey('ctrl+4', lambda: stop_manual_recording(recorder), suppress=True)
    print("Hotkeys set:")
    print("  Ctrl+1 to Unmute Microphone")
    print("  Ctrl+2 to Mute Microphone")
    print("  Ctrl+3 to Start Manual Recording")
    print("  Ctrl+4 to Stop Manual Recording")

def main():
    """Main function to run the Real-Time STT script with manual recording functionality."""
    clear_console()
    print("Initializing RealTimeSTT...")

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
            'post_speech_silence_duration': 1.0, # Duration in seconds of silence that must follow speech before the recording is
                                                 # considered to be completed. This ensures that any brief pauses during speech
                                                 # don't prematurely end the recording. Default is 0.2
            'min_length_of_recording': 1.0, # Specifies the minimum time interval in seconds that should exist between the end of one recording
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
        hotkey_thread = threading.Thread(target=setup_hotkeys, args=(recorder,), daemon=True)
        hotkey_thread.start()

        try:
            print("Say something! Press Ctrl+1 to Unmute, Ctrl+2 to Mute, Ctrl+3 to Start Manual Recording, Ctrl+4 to Stop Manual Recording.", end="", flush=True)
            while True:
                # Continuously listen and process audio
                recorder.text(lambda text: process_text(text, recorder))
        except KeyboardInterrupt:
            print("\nExiting RealTimeSTT...")
        except Exception as e:
            print(f"\n[Unexpected Error]: {e}")
        finally:
            try:
                with manual_recording_lock:
                    if is_manual_recording:
                        print("\nStopping manual recording before exit...")
                        recorder.stop()
                        is_manual_recording = False
                recorder.stop()  # Ensure the recorder is properly stopped
            except Exception as e:
                print(f"\n[Error during recorder shutdown]: {e}")

        # Print captured warnings
        for warning in w:
            print(f"{warning.message}")

    # Print the message after handling warnings
    print("Say something! Press Ctrl+1 to Unmute, Ctrl+2 to Mute, Ctrl+3 to Start Manual Recording, Ctrl+4 to Stop Manual Recording.", end="", flush=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows support
    main()
