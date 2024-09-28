from RealtimeSTT import AudioToTextRecorder
import pyautogui
import keyboard
import threading
import time

def on_realtime_transcription_update(text):
    print(f"\r{text}", end="")

def process_text(text):
    print(f"\nFinal transcription: {text}")
    # Simulate typing the text into the active window
    pyautogui.typewrite(text)

def on_recording_start():
    print("Transcription started.")

def on_recording_stop():
    print("Transcription stopped.")

def recorder_worker(recorder, is_unmuted_event, stop_event):
    while not stop_event.is_set():
        if is_unmuted_event.is_set():
            recorder.text(process_text)
        else:
            time.sleep(0.1)

def main():
    print("Initializing script...")

    recorder_config = {
        'spinner': False,
        'model': 'tiny.en',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.4,
        'min_length_of_recording': 0.5,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.2,
        'realtime_model_type': 'tiny.en',
        'on_realtime_transcription_update': on_realtime_transcription_update,
        'silero_deactivity_detection': True,
        'on_recording_start': on_recording_start,
        'on_recording_stop': on_recording_stop,
    }

    recorder = AudioToTextRecorder(**recorder_config)

    # Start with the recorder stopped
    recorder.stop()

    # Events to control the recorder worker thread
    is_unmuted_event = threading.Event()
    stop_event = threading.Event()

    # Start the recorder worker thread
    worker_thread = threading.Thread(target=recorder_worker, args=(recorder, is_unmuted_event, stop_event))
    worker_thread.start()

    def unmute_microphone():
        if not is_unmuted_event.is_set():
            print("Microphone unmuted. Ready to start transcribing.")
            is_unmuted_event.set()
            recorder.start()

    def mute_microphone():
        if is_unmuted_event.is_set():
            recorder.stop()
            print("Microphone muted.")
            is_unmuted_event.clear()

    # Set up hotkeys
    keyboard.add_hotkey('ctrl+a', unmute_microphone)
    keyboard.add_hotkey('ctrl+1', mute_microphone)

    print("Script is ready. Waiting for hotkeys.")
    print("Press Ctrl+A to unmute microphone and start transcribing.")
    print("Press Ctrl+1 to mute microphone.")

    try:
        while True:
            time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stop_event.set()
        worker_thread.join()
        recorder.shutdown()

if __name__ == '__main__':
    main()
