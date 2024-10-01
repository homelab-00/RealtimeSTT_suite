# Modified to work with a custom 'deque' external buffer

import os
import threading

# "from RealtimeSTT import AudioToTextRecorder" - We're commenting out this line as we're importing
# a local audio_recorder.py file instead of using that one that comes with the RealtimeSTT library.
import sys
sys.path.insert(0, './')  # This assumes audio_recorder.py is same directory as this script
from audio_recorder_v_1_4 import AudioToTextRecorder

from colorama import Fore, Style
import colorama
import keyboard
import multiprocessing
import warnings
import pyautogui
from collections import deque
import pyaudio
import numpy as np
import time
import logging
# import webrtcvad  # Import WebRTC VAD (Removed)

# Initialize colorama
colorama.init()

# # Configure logging to write to debug.log
# logging.basicConfig(
#     filename='debug.log',
#     level=logging.DEBUG,
#     format='%(asctime)s %(levelname)s:%(message)s'
# )

# Global variables
full_sentences = []
displayed_text = ""
muted = True  # Initially muted

# Lock for thread-safe access to 'muted'
mute_lock = threading.Lock()

# Initialize audio buffer
buffer_capacity = 1875  # Approximately one minute of audio at 16000 Hz with chunk size 512
audio_buffer = deque(maxlen=buffer_capacity)

# Initialize WebRTC VAD (Removed)
# vad = webrtcvad.Vad()
# vad.set_mode(1)  # 0: least aggressive, 3: most aggressive

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

def capture_audio_to_buffer():
    """Captures live audio from the microphone and stores it in the buffer without VAD."""
    p = pyaudio.PyAudio()
    try:
        # Open stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=512)  # Kept at 512 as per your setup
        while True:
            data = stream.read(512, exception_on_overflow=False)
            # Append all audio data to the buffer without VAD filtering
            audio_buffer.append(data)
            logging.debug(f"Audio chunk appended. Audio buffer size: {len(audio_buffer)}")
    except Exception as e:
        print(f"Error in audio capture: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def feed_audio_from_buffer(recorder):
    """Feeds audio chunks from the buffer into the transcription engine when internal buffer is not full."""
    while True:
        if len(audio_buffer) > 0:
            internal_buffer_size = recorder.get_internal_buffer_size()
            if internal_buffer_size < recorder.allowed_latency_limit:
                chunk = audio_buffer.popleft()
                recorder.feed_audio(chunk)
                logging.debug(f"Feeding audio chunk. Internal buffer size: {internal_buffer_size + 1}")
            else:
                # Internal buffer is full; wait before feeding more chunks
                logging.debug("Internal buffer is full. Waiting to feed more chunks.")
                time.sleep(0.1)
        else:
            time.sleep(0.01)  # Adjust sleep time as needed

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
            'model': 'base.en',  # Using the base.en model for the main transcription model
            'language': 'en',
            'device': 'cpu',
            'debug_mode': False,
            'use_main_model_for_realtime': False,
            'ensure_sentence_starting_uppercase': True,
            'ensure_sentence_ends_with_period': True,
            'handle_buffer_overflow': True,  # Enable internal buffer overflow handling
            'silero_sensitivity': 0.4,
            'webrtc_sensitivity': 2,
            'post_speech_silence_duration': 0.4,
            'min_length_of_recording': 0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0.2,
            'realtime_model_type': 'base.en',  # Use the base.en model for real-time transcription
            'on_realtime_transcription_update': lambda text: text_detected(text, recorder), 
            'silero_deactivity_detection': True,
            'use_microphone': False,  # Disable built-in microphone usage
        }

        # Initialize the recorder inside the main function
        recorder = AudioToTextRecorder(**recorder_config)

        # Start audio capture thread
        audio_capture_thread = threading.Thread(target=capture_audio_to_buffer, daemon=True)
        audio_capture_thread.start()

        # Start audio feeding thread
        audio_feeding_thread = threading.Thread(target=feed_audio_from_buffer, args=(recorder,), daemon=True)
        audio_feeding_thread.start()

        # Set up global hotkeys in a separate thread
        hotkey_thread = threading.Thread(target=setup_hotkeys, daemon=True)
        hotkey_thread.start()

        try:
            while True:
                # Continuously listen and process audio
                recorder.text(lambda text: process_text(text, recorder))
        except KeyboardInterrupt:
            print("\nExiting RealTimeSTT...")
        finally:
            recorder.shutdown()  # Ensure the recorder is properly stopped

    # Print captured warnings
    for warning in w:
        print(f"{warning.message}")

    # Print the message after handling warnings
    print("Say something! Press Ctrl+1 to Unmute, Ctrl+2 to Mute.", end="", flush=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows support
    main()