EXTENDED_LOGGING = False

if __name__ == '__main__':

    import subprocess
    import sys

    def install_rich():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])

    def install_pyautogui():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyautogui"])

    def install_keyboard():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"])

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
    from rich.spinner import Spinner
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    # console.print("[bold yellow]System initializing, please wait...[/bold yellow]")
    console.print("System initializing, please wait")

    import os
    import sys

    sys.path.insert(0, './')  # This assumes audio_recorder.py is same directory as this script
    from audio_recorder_v_1_0 import AudioToTextRecorder

    from colorama import Fore, Style
    import colorama

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()    

    colorama.init()

    # Install and import pyautogui
    try:
        import pyautogui
    except ImportError:
        user_input = input("This demo needs the 'pyautogui' library, which is not installed.\nDo you want to install it now? (y/n): ")
        if user_input.lower() == 'y':
            try:
                install_pyautogui()
                import pyautogui
                print("Successfully installed 'pyautogui'.")
            except Exception as e:
                print(f"An error occurred while installing 'pyautogui': {e}")
                sys.exit(1)
        else:
            print("The program requires the 'pyautogui' library to run. Exiting...")
            sys.exit(1)

    # Install and import keyboard
    try:
        import keyboard
    except ImportError:
        user_input = input("This demo needs the 'keyboard' library, which is not installed.\nDo you want to install it now? (y/n): ")
        if user_input.lower() == 'y':
            try:
                install_keyboard()
                import keyboard
                print("Successfully installed 'keyboard'.")
            except Exception as e:
                print(f"An error occurred while installing 'keyboard': {e}")
                sys.exit(1)
        else:
            print("The program requires the 'keyboard' library to run. Exiting...")
            sys.exit(1)

    import threading

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

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    prev_text = ""

    def preprocess_text(text):
        # Remove leading whitespaces
        text = text.lstrip()

        #  Remove starting ellipses if present
        if text.startswith("..."):
            text = text[3:]

        # Remove any leading whitespaces again after ellipses removal
        text = text.lstrip()

        # Uppercase the first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text


    # Global Flags
    typing_enabled = True  # Start with typing enabled
    running = True         # Control the main loop

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
                #rich_text += Text(sentence, style="bold yellow") + Text(" ")
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
        global recorder, full_sentences, prev_text, typing_enabled
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause
        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]
                
        full_sentences.append(text)
        prev_text = ""
        text_detected("")

        # Type the finalized sentence to the active window quickly if typing is enabled
        if typing_enabled:
            try:
                pyautogui.typewrite(text + ' ', interval=0.01)
            except Exception as e:
                console.print(f"[bold red]Failed to type the text: {e}[/bold red]")


    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': 'tiny.en', # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
        'input_device_index': 1,
        'realtime_model_type': 'tiny.en', # or small.en or distil-small.en or ...
        'language': 'en',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,        
        'min_gap_between_recordings': 0,                
        'enable_realtime_transcription': False,
        'realtime_processing_pause': 0.02,
        'on_realtime_transcription_update': text_detected,
        #'on_realtime_transcription_stabilized': text_detected,
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

    # Global Hotkey Functionality

    def disable_typing():
        global typing_enabled
        if typing_enabled:
            typing_enabled = False
            console.print("[bold red]Typing disabled.[/bold red]")
        else:
            console.print("[bold yellow]Typing is already disabled.[/bold yellow]")

    def enable_typing():
        global typing_enabled
        if not typing_enabled:
            typing_enabled = True
            console.print("[bold green]Typing enabled.[/bold green]")
        else:
            console.print("[bold yellow]Typing is already enabled.[/bold yellow]")

    def exit_script():
        global running
        console.print("[bold red]Exiting script...[/bold red]")
        running = False  # This will cause the main loop to exit

    def hotkey_listener():
        keyboard.add_hotkey('ctrl+1', disable_typing)
        keyboard.add_hotkey('ctrl+2', enable_typing)
        keyboard.add_hotkey('ctrl+3', exit_script)
        keyboard.wait()  # Block forever, as hotkeys are handled in callbacks

    # Start the hotkey listener in a separate daemon thread
    hotkey_thread = threading.Thread(target=hotkey_listener, daemon=True)
    hotkey_thread.start()

    try:
        while running:
            recorder.text(process_text)
    except KeyboardInterrupt:
        pass
    finally:
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        sys.exit(0)
