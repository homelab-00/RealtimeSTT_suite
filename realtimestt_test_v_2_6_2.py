EXTENDED_LOGGING = False

if __name__ == '__main__':

    import subprocess
    import sys

    def install_rich():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])

    try:
        import rich
    except ImportError:
        user_input = input("This demo needs the 'rich' library, which is not installed.\nDo you want to install it now? (y/n): ")
        if user_input.lower() == 'y':
            try:
                install_rich()
                import rich
                print("Successfully installed 'rich'.")
            except Exception as e:
                print(f"An error occurred while installing 'rich': {e}")
                sys.exit(1)
        else:
            print("The program requires the 'rich' library to run. Exiting...")
            sys.exit(1)

    if EXTENDED_LOGGING:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    console = Console()
    console.print("System initializing, please wait")

    import os

    sys.path.insert(0, './')  # This assumes audio_recorder.py is in the same directory
    from audio_recorder_v_1_0 import AudioToTextRecorder  # Import your module here

    import colorama
    colorama.init()

    # Import pyautogui
    import pyautogui

    import threading
    import pyaudio
    import numpy as np

    # Import keyboard module
    try:
        import keyboard
    except ImportError:
        user_input = input("This script requires the 'keyboard' library, which is not installed.\nDo you want to install it now? (y/n): ")
        if user_input.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"])
                import keyboard
                print("Successfully installed 'keyboard'.")
            except Exception as e:
                print(f"An error occurred while installing 'keyboard': {e}")
                sys.exit(1)
        else:
            print("The program requires the 'keyboard' library to run. Exiting...")
            sys.exit(1)

    # Import pyperclip for clipboard operations
    try:
        import pyperclip
    except ImportError:
        user_input = input("This script requires the 'pyperclip' library, which is not installed.\nDo you want to install it now? (y/n): ")
        if user_input.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyperclip"])
                import pyperclip
                print("Successfully installed 'pyperclip'.")
            except Exception as e:
                print(f"An error occurred while installing 'pyperclip': {e}")
                sys.exit(1)
        else:
            print("The program requires the 'pyperclip' library to run. Exiting...")
            sys.exit(1)

    # Initialize Rich Console and Live
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""  # Used for tracking text that was already displayed

    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0

    prev_text = ""

    def preprocess_text(text):
        # Remove leading whitespaces
        text = text.lstrip()

        # Remove starting ellipses if present
        if text.startswith("..."):
            text = text[3:]

        # Remove any leading whitespaces again after ellipses removal
        text = text.lstrip()

        # Uppercase the first letter
        if text:
            text = text[0].upper() + text[1:]

        return text

    def text_detected(text):
        global prev_text, displayed_text, rich_text_stored

        text = preprocess_text(text)

        sentence_end_marks = ['.', '!', '?', '。']
        if text.endswith("..."):
            recorder.post_speech_silence_duration = mid_sentence_detection_pause
        elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        prev_text = text

        # Build Rich Text with alternating colors
        rich_text = Text()
        for i, sentence in enumerate(full_sentences):
            if i % 2 == 0:
                rich_text += Text(sentence, style="yellow") + Text(" ")
            else:
                rich_text += Text(sentence, style="cyan") + Text(" ")

        # If the current text is not a sentence-ending, display it in real-time
        if text:
            rich_text += Text(text, style="bold yellow")

        new_displayed_text = rich_text.plain

        if new_displayed_text != displayed_text:
            displayed_text = new_displayed_text
            panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
            live.update(panel)
            rich_text_stored = rich_text

    def process_text(text):
        global recorder, full_sentences, prev_text
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause
        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]

        full_sentences.append(text)
        prev_text = ""
        text_detected("")

        # Type the finalized sentence to the active window quickly if typing is enabled
        try:
            # Release modifier keys to prevent stuck keys
            for key in ['ctrl', 'shift', 'alt', 'win']:
                keyboard.release(key)
                pyautogui.keyUp(key)

            # Use clipboard to paste text
            pyperclip.copy(text + ' ')
            pyautogui.hotkey('ctrl', 'v')

        except Exception as e:
            console.print(f"[bold red]Failed to type the text: {e}[/bold red]")

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': 'tiny.en',
        'input_device_index': 1,
        'realtime_model_type': 'tiny.en',
        'language': 'en',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': False,
        'realtime_processing_pause': 0.02,
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 5,
        'beam_size_realtime': 3,
        'no_log_file': True,
        'initial_prompt': "Use ellipses for incomplete sentences like: I went to the..."
    }

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    recorder = AudioToTextRecorder(**recorder_config)

    initial_text = Panel(Text("Say something...", style="cyan bold"), title="[bold yellow]Waiting for Input[/bold yellow]", border_style="bold yellow")
    live.update(initial_text)

    # Print available hotkeys
    console.print("[bold green]Available Hotkeys:[/bold green]")
    console.print("[bold cyan]F1[/bold cyan]: Mute Microphone")
    console.print("[bold cyan]F2[/bold cyan]: Unmute Microphone")
    console.print("[bold cyan]F3[/bold cyan]: Start Static Recording")
    console.print("[bold cyan]F4[/bold cyan]: Stop Static Recording")

    # Global variables for static recording
    static_recording_active = False
    static_recording_thread = None
    static_audio_frames = []
    live_recording_enabled = True  # Track whether live recording was enabled before static recording

    # Note: The maximum recommended length of static recording is about 5 minutes.

    def static_recording_worker():
        """
        Worker function to record audio statically.
        """
        global static_audio_frames, static_recording_active
        # Set up pyaudio
        p = pyaudio.PyAudio()
        # Use the same audio format as the recorder
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000  # Sample rate
        CHUNK = 1024  # Buffer size

        # Open the audio stream
        try:
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        except Exception as e:
            console.print(f"[bold red]Failed to open audio stream for static recording: {e}[/bold red]")
            static_recording_active = False
            p.terminate()
            return

        while static_recording_active:
            try:
                data = stream.read(CHUNK)
                static_audio_frames.append(data)
            except Exception as e:
                console.print(f"[bold red]Error during static recording: {e}[/bold red]")
                break

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

    def start_static_recording():
        """
        Starts the static audio recording.
        """
        global static_recording_active, static_recording_thread, static_audio_frames, live_recording_enabled
        if static_recording_active:
            console.print("[bold yellow]Static recording is already in progress.[/bold yellow]")
            return

        # Mute the live recording microphone
        live_recording_enabled = recorder.use_microphone.value
        if live_recording_enabled:
            recorder.set_microphone(False)
            console.print("[bold yellow]Live microphone muted during static recording.[/bold yellow]")

        console.print("[bold green]Starting static recording... Press F8 to stop.[/bold green]")
        static_audio_frames = []
        static_recording_active = True
        static_recording_thread = threading.Thread(target=static_recording_worker)
        static_recording_thread.start()

    def stop_static_recording():
        """
        Stops the static audio recording and processes the transcription.
        """
        global static_recording_active, static_recording_thread, static_audio_frames, live_recording_enabled
        if not static_recording_active:
            console.print("[bold yellow]No static recording is in progress.[/bold yellow]")
            return

        console.print("[bold green]Stopping static recording...[/bold green]")
        static_recording_active = False
        static_recording_thread.join()

        # Process the recorded audio
        console.print("[bold green]Processing static recording...[/bold green]")

        # Convert audio data to numpy array
        audio_data = b''.join(static_audio_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe the audio data
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            console.print("[bold red]faster_whisper is not installed. Please install it to use static transcription.[/bold red]")
            return

        # Load the model
        model_size = recorder_config.get('model', 'tiny.en')
        device = recorder_config.get('device', 'cpu')
        compute_type = recorder_config.get('compute_type', 'default')

        console.print("Loading transcription model... This may take a moment.")
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            console.print(f"[bold red]Failed to load transcription model: {e}[/bold red]")
            return

        # Transcribe the audio
        try:
            segments, info = model.transcribe(audio_array, beam_size=5)
            transcription = ' '.join([segment.text for segment in segments]).strip()
        except Exception as e:
            console.print(f"[bold red]Error during transcription: {e}[/bold red]")
            return

        # Display the transcription
        console.print("Static Recording Transcription:")
        console.print(f"[bold cyan]{transcription}[/bold cyan]")

        # Type the transcription into the active window
        try:
            # Release modifier keys to prevent stuck keys
            for key in ['ctrl', 'shift', 'alt', 'win']:
                keyboard.release(key)
                pyautogui.keyUp(key)

            # Use clipboard to paste text
            pyperclip.copy(transcription + ' ')
            pyautogui.hotkey('ctrl', 'v')

        except Exception as e:
            console.print(f"[bold red]Failed to type the static transcription: {e}[/bold red]")

        # Unmute the live recording microphone if it was enabled before
        if live_recording_enabled:
            recorder.set_microphone(True)
            console.print("[bold yellow]Live microphone unmuted.[/bold yellow]")

    # Hotkey Callback Functions

    def mute_microphone():
        recorder.set_microphone(False)
        console.print("[bold red]Microphone muted.[/bold red]")

    def unmute_microphone():
        recorder.set_microphone(True)
        console.print("[bold green]Microphone unmuted.[/bold green]")

    # Hotkey Listener Function using keyboard module

    def hotkey_listener():
        # Define the hotkey combinations and their corresponding functions
        keyboard.add_hotkey('F1', mute_microphone, suppress=True)
        keyboard.add_hotkey('F2', unmute_microphone, suppress=True)
        keyboard.add_hotkey('F3', start_static_recording, suppress=True)
        keyboard.add_hotkey('F4', stop_static_recording, suppress=True)

        # Keep the program running
        keyboard.wait()

    # Start the hotkey listener in a separate daemon thread
    hotkey_thread = threading.Thread(target=hotkey_listener, daemon=True)
    hotkey_thread.start()

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        pass
    finally:
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        sys.exit(0)
