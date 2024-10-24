# Filename: transcription_refinement_pipeline_with_gptneo.py

EXTENDED_LOGGING = False

if __name__ == '__main__':

    import subprocess
    import sys
    import threading
    import time

    def install_package(package_name):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

    try:
        import rich
    except ImportError:
        user_input = input("This demo needs the 'rich' library, which is not installed.\nDo you want to install it now? (y/n): ")
        if user_input.lower() == 'y':
            try:
                install_package("rich")
                import rich
                print("Successfully installed 'rich'.")
            except Exception as e:
                print(f"An error occurred while installing 'rich': {e}")
                sys.exit(1)
        else:
            print("The program requires the 'rich' library to run. Exiting...")
            sys.exit(1)

    import keyboard
    import pyperclip

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
    sys.path.insert(0, './')  # This assumes audio_recorder_v_1_0.py is in the same directory
    from audio_recorder_v_1_0 import AudioToTextRecorder  # Ensure this module has stop() or close() methods

    import colorama
    colorama.init()

    # Import pyautogui
    import pyautogui

    import pyaudio
    import numpy as np

    # Initialize Rich Console and Live
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    # Global variables
    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""  # Used for tracking text that was already displayed

    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0

    prev_text = ""

    # Events to signal threads to exit or reset
    exit_event = threading.Event()
    reset_event = threading.Event()

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

    # ---------------- GPT-Neo-1.3B Initialization ----------------

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def initialize_gpt_neo(model_name="EleutherAI/gpt-neo-1.3B"):
        """
        Initializes the GPT-Neo-1.3B model with 8-bit quantization.
        """
        console.print("[bold green]Initializing GPT-Neo-1.3B model... This may take a moment.[/bold green]")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,             # Enables 8-bit quantization
            device_map='auto',             # Automatically maps to available devices
            torch_dtype=torch.float16      # Uses mixed precision
        )
        console.print("[bold green]GPT-Neo-1.3B model initialized successfully.[/bold green]")
        return tokenizer, model

    tokenizer, gpt_neo = initialize_gpt_neo()

    def refine_text_gpt_neo(text):
        """
        Refines the transcribed text using GPT-Neo-1.3B.
        Corrects grammar and ensures semantic clarity while preserving original meaning.
        """
        prompt = (
            "Please correct the following text for grammar and clarity while preserving the original meaning:\n\n"
            f"{text}"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(gpt_neo.device)
        with torch.no_grad():
            outputs = gpt_neo.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
        refined_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the refined text after the prompt
        refined_text = refined_text.replace(prompt, "").strip()
        return refined_text

    # ---------------- End of GPT-Neo-1.3B Initialization ----------------

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
        global recorder, full_sentences, prev_text, displayed_text
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause
        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]

        full_sentences.append(text)
        prev_text = ""
        text_detected("")

        # Check if reset_event is set
        if reset_event.is_set():
            # Clear buffers
            full_sentences.clear()
            displayed_text = ""
            reset_event.clear()
            console.print("[bold magenta]Transcription buffer reset.[/bold magenta]")
            return

        # ------------------ Refinement with GPT-Neo-1.3B ------------------
        try:
            refined_text = refine_text_gpt_neo(text)
            console.print("\n[bold green]Refined Text:[/bold green]")
            console.print(f"[bold cyan]{refined_text}[/bold cyan]\n")
        except Exception as e:
            console.print(f"[bold red]Error during text refinement: {e}[/bold red]")
            refined_text = text  # Fallback to original text if refinement fails
        # ---------------------------------------------------------------------

        # Type the refined transcription into the active window
        try:
            # Release modifier keys to prevent stuck keys
            for key in ['ctrl', 'shift', 'alt', 'win']:
                keyboard.release(key)
                pyautogui.keyUp(key)

            # Use clipboard to paste text
            pyperclip.copy(refined_text + ' ')
            pyautogui.hotkey('ctrl', 'v')

        except Exception as e:
            console.print(f"[bold red]Failed to type the refined text: {e}[/bold red]")

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': 'Systran/faster-distil-whisper-large-v3',  # or distil-medium.en or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or Systran/faster-distil-whisper-large-v3 or ...
        'input_device_index': 1,
        'realtime_model_type': 'tiny.en',  # or small.en or distil-small.en or ...
        'language': 'en',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.02,
        'on_realtime_transcription_update': text_detected,
        # 'on_realtime_transcription_stabilized': text_detected,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 5,
        'beam_size_realtime': 5,  # Matching beam_size for consistency
        'no_log_file': True,
        'initial_prompt': "Use ellipses for incomplete sentences like: I went to the...",
        'device': 'cuda',          # Added device configuration
        'compute_type': 'float16'  # Added compute_type configuration
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
    console.print("[bold cyan]F5[/bold cyan]: Reset Transcription")

    # Global variables for static recording
    static_recording_active = False
    static_recording_thread = None
    static_audio_frames = []
    live_recording_enabled = True  # Track whether live recording was enabled before static recording

    # Audio settings for static recording
    audio_settings = {
        'FORMAT': pyaudio.paInt16,  # PyAudio format
        'CHANNELS': 1,               # Mono audio
        'RATE': 16000,               # Sample rate
        'CHUNK': 1024                # Buffer size
    }

    # Note: The maximum recommended length of static recording is about 5 minutes.

    def static_recording_worker():
        """
        Worker function to record audio statically.
        """
        global static_audio_frames, static_recording_active
        # Set up pyaudio
        p = pyaudio.PyAudio()
        # Use the same audio format as defined in audio_settings
        FORMAT = audio_settings['FORMAT']
        CHANNELS = audio_settings['CHANNELS']
        RATE = audio_settings['RATE']  # Sample rate
        CHUNK = audio_settings['CHUNK']  # Buffer size

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

        while static_recording_active and not exit_event.is_set():
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

        console.print("[bold green]Starting static recording... Press F4 or F5 to stop/reset.[/bold green]")
        static_audio_frames = []
        static_recording_active = True
        static_recording_thread = threading.Thread(target=static_recording_worker, daemon=True)
        static_recording_thread.start()

    def stop_static_recording():
        """
        Stops the static audio recording and processes the transcription.
        """
        global static_recording_active, static_recording_thread
        if not static_recording_active:
            console.print("[bold yellow]No static recording is in progress.[/bold yellow]")
            return

        console.print("[bold green]Stopping static recording...[/bold green]")
        static_recording_active = False
        if static_recording_thread is not None:
            static_recording_thread.join()
            static_recording_thread = None

        # Start a new thread to process the transcription
        processing_thread = threading.Thread(target=process_static_transcription, daemon=True)
        processing_thread.start()

    def process_static_transcription():
        global static_audio_frames, live_recording_enabled
        if exit_event.is_set():
            return
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

        # Load the model using recorder_config
        model_size = recorder_config['model']
        device = recorder_config['device']
        compute_type = recorder_config['compute_type']

        console.print("Loading transcription model... This may take a moment.")
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            console.print(f"[bold red]Failed to load transcription model: {e}[/bold red]")
            return

        # Transcribe the audio
        try:
            segments, info = model.transcribe(audio_array, beam_size=recorder_config['beam_size'])
            transcription = ' '.join([segment.text for segment in segments]).strip()
        except Exception as e:
            console.print(f"[bold red]Error during transcription: {e}[/bold red]")
            return

        # ------------------ Refinement with GPT-Neo-1.3B ------------------
        try:
            refined_text = refine_text_gpt_neo(transcription)
            console.print("\n[bold green]Refined Static Transcription:[/bold green]")
            console.print(f"[bold cyan]{refined_text}[/bold cyan]\n")
        except Exception as e:
            console.print(f"[bold red]Error during static text refinement: {e}[/bold red]")
            refined_text = transcription  # Fallback to original text if refinement fails
        # ---------------------------------------------------------------------

        # Type the refined transcription into the active window
        try:
            # Release modifier keys to prevent stuck keys
            for key in ['ctrl', 'shift', 'alt', 'win']:
                keyboard.release(key)
                pyautogui.keyUp(key)

            # Use clipboard to paste text
            pyperclip.copy(refined_text + ' ')
            pyautogui.hotkey('ctrl', 'v')

        except Exception as e:
            console.print(f"[bold red]Failed to type the static transcription: {e}[/bold red]")

        # Unmute the live recording microphone if it was enabled before
        if live_recording_enabled and not exit_event.is_set():
            recorder.set_microphone(True)
            console.print("[bold yellow]Live microphone unmuted.[/bold yellow]")

    def reset_transcription():
        """
        Resets the transcription by flushing ongoing recordings or buffers.
        """
        global static_recording_active, static_recording_thread, static_audio_frames
        console.print("[bold magenta]Resetting transcription...[/bold magenta]")
        if static_recording_active:
            console.print("[bold magenta]Flushing static recording...[/bold magenta]")
            # Stop static recording
            static_recording_active = False
            if static_recording_thread is not None:
                static_recording_thread.join()
                static_recording_thread = None
            # Clear static audio frames
            static_audio_frames = []
            # Unmute microphone if it was muted during static recording
            if live_recording_enabled:
                recorder.set_microphone(True)
                console.print("[bold yellow]Live microphone unmuted after reset.[/bold yellow]")
        elif recorder.use_microphone.value:
            # Live transcription is active and microphone is not muted
            console.print("[bold magenta]Resetting live transcription buffer...[/bold magenta]")
            reset_event.set()
        else:
            # Microphone is muted; nothing to reset
            console.print("[bold yellow]Microphone is muted. Nothing to reset.[/bold yellow]")

    # Hotkey Callback Functions

    def mute_microphone():
        recorder.set_microphone(False)
        console.print("[bold red]Microphone muted.[/bold red]")

    def unmute_microphone():
        recorder.set_microphone(True)
        console.print("[bold green]Microphone unmuted.[/bold green]")

    # Start the transcription loop in a separate thread
    def transcription_loop():
        try:
            while not exit_event.is_set():
                # The original script likely expects a callback; adjust accordingly
                # Assuming recorder.text(callback) processes incoming text
                recorder.text(process_text)
        except Exception as e:
            console.print(f"[bold red]Error in transcription loop: {e}[/bold red]")
        finally:
            # Do not call sys.exit() here
            pass

    # Start the transcription loop thread
    transcription_thread = threading.Thread(target=transcription_loop, daemon=True)
    transcription_thread.start()

    # Define the hotkey combinations and their corresponding functions
    keyboard.add_hotkey('F1', mute_microphone, suppress=True)
    keyboard.add_hotkey('F2', unmute_microphone, suppress=True)
    keyboard.add_hotkey('F3', start_static_recording, suppress=True)
    keyboard.add_hotkey('F4', stop_static_recording, suppress=True)
    keyboard.add_hotkey('F5', reset_transcription, suppress=True)

    # Keep the main thread running and handle graceful exit
    try:
        keyboard.wait()  # Waits indefinitely, until a hotkey triggers an exit or Ctrl+C
    except KeyboardInterrupt:
        console.print("[bold yellow]KeyboardInterrupt received. Exiting...[/bold yellow]")
    finally:
        # Signal threads to exit
        exit_event.set()

        # Reset transcription if needed
        reset_transcription()

        # Stop the recorder
        try:
            if hasattr(recorder, 'stop'):
                recorder.stop()
            elif hasattr(recorder, 'close'):
                recorder.close()
        except Exception as e:
            console.print(f"[bold red]Error stopping recorder: {e}[/bold red]")

        # Allow some time for threads to finish
        time.sleep(1)

        # Wait for transcription_thread to finish
        if transcription_thread.is_alive():
            transcription_thread.join(timeout=5)

        # Stop the Live console
        live.stop()

        console.print("[bold red]Exiting gracefully...[/bold red]")
        sys.exit(0)
