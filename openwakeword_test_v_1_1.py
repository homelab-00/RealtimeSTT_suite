if __name__ == '__main__':
    print("Starting...")
    from RealtimeSTT import AudioToTextRecorder
    import logging
    import socket
    import threading

    detected = False

    say_wakeword_str = "Listening for wakeword 'samantha'."

    def send_command(command):
        """Sends a command to script 1 via socket."""
        host = 'localhost'
        port = 65432  # Port that script 1 is listening on
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                s.sendall(command.encode('utf-8'))
        except ConnectionRefusedError:
            print("Could not connect to script 1. Is it running?")

    def on_wakeword_detected():
        global detected
        detected = True
        send_command('mute')
        threading.Timer(0.5, lambda: send_command('delete_last_word')).start()

    def on_recording_stop():
        print("Transcribing...")
        send_command('unmute')
    
    def on_wakeword_timeout():
        global detected
        if not detected:
            print(f"Timeout. {say_wakeword_str}")

        detected = False

    def on_wakeword_detection_start():
        print(f"\n{say_wakeword_str}")

    def on_recording_start():
        print("Recording...")

    def on_vad_detect_start():
        print()
        print()

    def text_detected(text):
        print(f">> {text}")
        if "delete" in text.lower():
            send_command('delete_last_two_words')

    with AudioToTextRecorder(
        spinner=False,
        model="base.en",
        language="en", 
        wakeword_backend="oww",
        wake_words_sensitivity=0.35,
        openwakeword_model_paths="suh_man_tuh.onnx,suh_mahn_thuh.onnx",  # Load the appropriate models
        on_wakeword_detected=on_wakeword_detected,
        on_recording_start=on_recording_start,
        on_recording_stop=on_recording_stop,
        on_wakeword_timeout=on_wakeword_timeout,
        on_wakeword_detection_start=on_wakeword_detection_start,
        on_vad_detect_start=on_vad_detect_start,
        wake_word_buffer_duration=1,
        ) as recorder:

        while True:                
            recorder.text(text_detected)
