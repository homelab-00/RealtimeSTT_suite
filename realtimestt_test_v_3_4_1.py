import threading
import time
import sys
import keyboard
import pyautogui
import pyperclip
import numpy as np
from RealtimeSTT import AudioToTextRecorder
from rich.console import Console
from rich.live import Live
from rich.table import Table
import torch
import logging

def main():
    # Initialize the console for rich output
    console = Console()

    # Configuration for the recorder
    recorder_config = {
        'model': 'Systran/faster-whisper-large-v2-ct2',  # Use your desired model here
        'language': '',
        'compute_type': 'float16',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'beam_size': 5,
        'initial_prompt': None,
        'suppress_tokens': [-1],
        'use_microphone': True,
        'spinner': False,
        'level': logging.WARNING,
    }

    # Initialize the recorder
    recorder = AudioToTextRecorder(**recorder_config)

    # Event to signal the exit of threads
    exit_event = threading.Event()

    # Flags for controlling the program flow
    live_recording_enabled = True

    def process_text(text):
        """Process transcribed text by typing it into the active window."""
        try:
            # Use clipboard to paste text
            pyperclip.copy(text + ' ')
            pyautogui.hotkey('ctrl', 'v')
        except Exception as e:
            console.print(f"[bold red]Failed to type the transcription: {e}[/bold red]")

    def reset_transcription():
        """Resets the transcription by flushing ongoing recordings or buffers."""
        console.print("[bold magenta]Resetting transcription...[/bold magenta]")
        # Since the recorder handles buffering internally, we can reset it
        recorder.clear_audio_queue()
        console.print("[bold magenta]Transcription buffer cleared.[/bold magenta]")

    # Hotkey Callback Functions
    def mute_microphone():
        recorder.set_microphone(False)
        console.print("[bold red]Microphone muted.[/bold red]")

    def unmute_microphone():
        recorder.set_microphone(True)
        console.print("[bold green]Microphone unmuted.[/bold green]")

    def start_recording():
        console.print("[bold green]Starting recording...[/bold green]")
        recorder.start()

    def stop_recording():
        console.print("[bold red]Stopping recording...[/bold red]")
        recorder.stop()

    # Start the transcription loop in a separate thread
    def transcription_loop():
        try:
            while not exit_event.is_set():
                recorder.text(process_text)
        except Exception as e:
            console.print(f"[bold red]Error in transcription loop: {e}[/bold red]")

    transcription_thread = threading.Thread(target=transcription_loop, daemon=True)
    transcription_thread.start()

    # Define the hotkey combinations and their corresponding functions
    keyboard.add_hotkey('F1', mute_microphone, suppress=True)
    keyboard.add_hotkey('F2', unmute_microphone, suppress=True)
    keyboard.add_hotkey('F3', start_recording, suppress=True)
    keyboard.add_hotkey('F4', stop_recording, suppress=True)
    keyboard.add_hotkey('F5', reset_transcription, suppress=True)

    console.print("[bold cyan]Press F1 to mute microphone.")
    console.print("Press F2 to unmute microphone.")
    console.print("Press F3 to start recording.")
    console.print("Press F4 to stop recording.")
    console.print("Press F5 to reset transcription buffer.")
    console.print("Press Ctrl+C to exit.[/bold cyan]")

    # Keep the main thread running and handle graceful exit
    try:
        while not exit_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("[bold yellow]KeyboardInterrupt received. Exiting...[/bold yellow]")
    finally:
        # Signal threads to exit
        exit_event.set()

        # Stop the recorder
        recorder.shutdown()

        # Wait for transcription_thread to finish
        if transcription_thread.is_alive():
            transcription_thread.join(timeout=5)

        console.print("[bold red]Exiting gracefully...[/bold red]")
        sys.exit(0)

if __name__ == '__main__':
    main()
