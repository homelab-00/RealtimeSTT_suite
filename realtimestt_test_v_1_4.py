if __name__ == '__main__':

    EXTENDED_LOGGING = False  # Set to True for detailed logging

    if EXTENDED_LOGGING:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    import os
    import sys
    import time
    import copy
    from colorama import Fore, Style
    import colorama

    # Add the current directory to the system path
    sys.path.insert(0, './')  # Assumes audio_recorder.py is in the same directory

    from audio_recorder_v_1_1 import AudioToTextRecorder

    # Initialize colorama
    colorama.init()

    # Toggle to use both Real-Time and Main Models or only Real-Time Model
    USE_MAIN_MODEL = False  # Set to False to use only the real-time model

    # Configuration for the recorder
    recorder_config = {
        'spinner': False,
        'model': 'large-v2',  # Main model
        'realtime_model_type': 'tiny.en',  # Real-time model
        'language': 'en',
        'input_device_index': 1,  # Adjust based on your microphone device index
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': 0.4,  # Seconds
        'min_length_of_recording': 0.7,  # Seconds
        'min_gap_between_recordings': 0,  # Seconds
        'enable_realtime_transcription': True,
        'use_main_model_for_realtime': False,
        'realtime_processing_pause': 0.1,  # Seconds
        'on_realtime_transcription_update': None,  # To be set later
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,  # Disable early transcription
        'beam_size': 5,
        'beam_size_realtime': 1,
        'no_log_file': False,
    }

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    # Initialize lists and variables for managing transcriptions
    full_sentences = []
    displayed_text = ""
    prev_text = ""

    # Function to clear the console
    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    # Callback function for real-time transcription updates
    def text_detected(text):
        global displayed_text, prev_text, full_sentences
        sentence_end_marks = ['.', '!', '?', '。'] 

        # Adjust post-speech silence duration based on sentence ending
        if text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = mid_sentence_detection_pause

        prev_text = text

        # Apply color formatting to previous full sentences
        sentences_with_style = [
            f"{Fore.YELLOW + sentence + Style.RESET_ALL if i % 2 == 0 else Fore.CYAN + sentence + Style.RESET_ALL} "
            for i, sentence in enumerate(full_sentences)
        ]
        # Combine formatted sentences with current real-time text
        new_text = "".join(sentences_with_style).strip() + " " + text if len(sentences_with_style) > 0 else text

        # Update the displayed text if there are changes
        if new_text != displayed_text:
            displayed_text = new_text
            clear_console()
            print(displayed_text, end="", flush=True)

    # Callback function for finalized transcription from the main model
    def process_text(text):
        recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        full_sentences.append(text)
        prev_text = ""
        text_detected("")  # Refresh the display with the new full sentence

    # Initialize the AudioToTextRecorder with the configuration
    recorder = AudioToTextRecorder(**recorder_config)

    # Assign the real-time transcription update callback
    recorder.on_realtime_transcription_update = text_detected

    # If not using the main model, enable recording on voice activity
    if not USE_MAIN_MODEL:
        recorder.start_recording_on_voice_activity = True

    # Define silence durations
    end_of_sentence_detection_pause = 0.4  # Seconds
    mid_sentence_detection_pause = 0.7  # Seconds

    # Clear the console and prompt the user
    clear_console()
    print("Say something...", end="", flush=True)

    try:
        if USE_MAIN_MODEL:
            # Use both Real-Time and Main Models
            while True:
                recorder.text(process_text)
        else:
            # Use only the Real-Time Model
            try:
                while True:
                    time.sleep(0.1)  # Keep the script running
            except KeyboardInterrupt:
                recorder.shutdown()
                print("\nExiting application due to keyboard interrupt")
    except KeyboardInterrupt:
        recorder.shutdown()
        print("\nExiting application due to keyboard interrupt")
