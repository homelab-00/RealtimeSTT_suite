#!/usr/bin/env python3
# orchestrator.py
#
# Main controller for the Speech-to-Text system
#
# This script:
# - Imports and integrates the three transcription modules:
#   * Real-time transcription for immediate feedback
#   * Long-form transcription for extended dictation
#   * Static file transcription for pre-recorded audio/video
# - Sets up a TCP server to listen for commands from the AutoHotkey script
# - Manages the state of different transcription modes
# - Provides a clean interface for hotkey-based control
# - Handles command processing and module coordination
# - Implements lazy loading of transcription models
#
# The system is designed to be controlled via the following hotkeys:
# - F2: Toggle real-time transcription on/off
# - F3: Start long-form recording
# - F4: Stop long-form recording and transcribe
# - F10: Run static file transcription
# - F7: Quit application

import os
import sys
import socket
import threading
import time
import logging
from typing import Optional, Dict, Any
import importlib.util
import subprocess
import signal
import atexit
import io
import psutil

# Configure logging to file only (not to console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stt_orchestrator.log"),
    ]
)

# Console output with color support
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None

# TCP server settings
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 35000

# Module paths
MODULE_PATHS = {
    "realtime": "realtime_module.py",
    "longform": "longform_module.py",
    "static": "static_module.py"
}

def safe_print(message):
    """Print function that handles I/O errors gracefully."""
    try:
        if HAS_RICH:
            console.print(message)
        else:
            print(message)
    except ValueError as e:
        if "I/O operation on closed file" in str(e):
            pass  # Silently ignore closed file errors
        else:
            # For other ValueErrors, log them
            logging.error(f"Error in safe_print: {e}")

class STTOrchestrator:
    """
    Main orchestrator for the Speech-to-Text system.
    Coordinates between different transcription modes and handles hotkey commands.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.script_dir, "config.json")
        
        # Application state
        self.running = False
        self.server_thread = None
        self.current_mode = None  # Can be "realtime", "longform", or "static"
        self.ahk_pid = None

        # Initialize configuration
        self._load_or_create_config()
        
        # Module information - we'll import modules lazily
        self.modules = {}
        self.transcribers = {}
        
        # Register cleanup handler
        atexit.register(self.stop)

    def _load_or_create_config(self):
        """Load configuration from file or create it if it doesn't exist."""
        import json

        # Define default configuration with full model names and English language
        default_config = {
            "realtime": {
                "model": "Systran/faster-whisper-large-v3",
                "language": "en",
                "compute_type": "default",
                "device": "cuda",
                "input_device_index": None,
                "gpu_device_index": 0,
                "silero_sensitivity": 0.4,
                "silero_use_onnx": False,
                "silero_deactivity_detection": False,
                "webrtc_sensitivity": 3,
                "post_speech_silence_duration": 0.6,
                "min_length_of_recording": 1.0,
                "min_gap_between_recordings": 1.0,
                "pre_recording_buffer_duration": 0.2,
                "ensure_sentence_starting_uppercase": True,
                "ensure_sentence_ends_with_period": True,
                "batch_size": 16,
                "beam_size": 5,
                "beam_size_realtime": 3,
                "initial_prompt": None,
                "allowed_latency_limit": 100,
                "early_transcription_on_silence": 0,
                "enable_realtime_transcription": True,
                "realtime_processing_pause": 0.2,
                "realtime_model_type": "tiny.en",
                "realtime_batch_size": 16
            },
            "longform": {
                "model": "Systran/faster-whisper-large-v3",
                "language": "en",
                "compute_type": "default",
                "device": "cuda",
                "input_device_index": None,
                "gpu_device_index": 0,
                "silero_sensitivity": 0.4,
                "silero_use_onnx": False,
                "silero_deactivity_detection": False,
                "webrtc_sensitivity": 3,
                "post_speech_silence_duration": 0.6,
                "min_length_of_recording": 1.0,
                "min_gap_between_recordings": 1.0,
                "pre_recording_buffer_duration": 0.2,
                "ensure_sentence_starting_uppercase": True,
                "ensure_sentence_ends_with_period": True,
                "batch_size": 16,
                "beam_size": 5,
                "initial_prompt": None,
                "allowed_latency_limit": 100
            },
            "static": {
                "model": "Systran/faster-whisper-large-v3",
                "language": "en",
                "compute_type": "float16",
                "device": "cuda",
                "gpu_device_index": 0,
                "beam_size": 5,
                "batch_size": 16,
                "vad_aggressiveness": 2
            }
        }

        # Try to load existing config
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)

                    # Update default config with loaded values
                    for module_type in default_config:
                        if module_type in loaded_config:
                            for param, value in loaded_config[module_type].items():
                                if param in default_config[module_type]:
                                    default_config[module_type][param] = value

                    self.config = default_config
                    self.log_info("Configuration loaded from file")
            except Exception as e:
                self.log_error(f"Error loading configuration: {e}")
                self.config = default_config
        else:
            # Use defaults and save to file
            self.config = default_config
            self._save_config()
            self.log_info("Default configuration created")

    def _save_config(self):
        """Save current configuration to file."""
        import json

        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.log_info("Configuration saved to file")
        except Exception as e:
            self.log_error(f"Error saving configuration: {e}")

    def import_module_lazily(self, module_name):
        """Import a module only when needed."""
        if module_name in self.modules and self.modules[module_name]:
            return self.modules[module_name]
            
        filepath = os.path.join(self.script_dir, MODULE_PATHS.get(module_name, ""))
        
        try:
            # First check if the file exists
            if not os.path.exists(filepath):
                self.log_error(f"Module file not found: {filepath}")
                return None
                
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None:
                self.log_error(f"Could not find module: {module_name} at {filepath}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # Add to sys.modules to avoid import errors
            spec.loader.exec_module(module)
            
            # Store the imported module
            self.modules[module_name] = module
            self.log_info(f"Successfully imported module: {module_name}")
            return module
        except Exception as e:
            self.log_error(f"Error importing {module_name} from {filepath}: {e}")
            return None
    
    def initialize_transcriber(self, module_type):
        """Initialize a transcriber only when needed."""
        if module_type in self.transcribers and self.transcribers[module_type]:
            return self.transcribers[module_type]
            
        module = self.import_module_lazily(module_type)
        if not module:
            self.log_error(f"Failed to import {module_type} module")
            return None
            
        try:
            # Get configuration for this module
            module_config = self.config.get(module_type, {})
            
            # Use a different initialization approach for each module type
            if module_type == "realtime":
                safe_print(f"Initializing real-time transcriber...")
                # Pass all configuration parameters
                self.transcribers[module_type] = module.LongFormTranscriber(
                    model=module_config.get("model", "Systran/faster-whisper-large-v3"),
                    language=module_config.get("language", "en"),
                    compute_type=module_config.get("compute_type", "default"),
                    device=module_config.get("device", "cuda"),
                    input_device_index=module_config.get("input_device_index"),
                    gpu_device_index=module_config.get("gpu_device_index", 0),
                    silero_sensitivity=module_config.get("silero_sensitivity", 0.4),
                    silero_use_onnx=module_config.get("silero_use_onnx", False),
                    silero_deactivity_detection=module_config.get("silero_deactivity_detection", False),
                    webrtc_sensitivity=module_config.get("webrtc_sensitivity", 3),
                    post_speech_silence_duration=module_config.get("post_speech_silence_duration", 0.6),
                    min_length_of_recording=module_config.get("min_length_of_recording", 1.0),
                    min_gap_between_recordings=module_config.get("min_gap_between_recordings", 1.0),
                    pre_recording_buffer_duration=module_config.get("pre_recording_buffer_duration", 0.2),
                    ensure_sentence_starting_uppercase=module_config.get("ensure_sentence_starting_uppercase", True),
                    ensure_sentence_ends_with_period=module_config.get("ensure_sentence_ends_with_period", True),
                    batch_size=module_config.get("batch_size", 16),
                    beam_size=module_config.get("beam_size", 5),
                    beam_size_realtime=module_config.get("beam_size_realtime", 3),
                    initial_prompt=module_config.get("initial_prompt"),
                    allowed_latency_limit=module_config.get("allowed_latency_limit", 100),
                    early_transcription_on_silence=module_config.get("early_transcription_on_silence", 0),
                    enable_realtime_transcription=module_config.get("enable_realtime_transcription", True),
                    realtime_processing_pause=module_config.get("realtime_processing_pause", 0.2),
                    realtime_model_type=module_config.get("realtime_model_type", "tiny.en"),
                    realtime_batch_size=module_config.get("realtime_batch_size", 16)
                )
                    
            elif module_type == "longform":
                safe_print(f"Initializing long-form transcriber...")
                # Pass all configuration parameters
                self.transcribers[module_type] = module.LongFormTranscriber(
                    model=module_config.get("model", "Systran/faster-whisper-large-v3"),
                    language=module_config.get("language", "en"),
                    compute_type=module_config.get("compute_type", "default"),
                    device=module_config.get("device", "cuda"),
                    input_device_index=module_config.get("input_device_index"),
                    gpu_device_index=module_config.get("gpu_device_index", 0),
                    silero_sensitivity=module_config.get("silero_sensitivity", 0.4),
                    silero_use_onnx=module_config.get("silero_use_onnx", False),
                    silero_deactivity_detection=module_config.get("silero_deactivity_detection", False),
                    webrtc_sensitivity=module_config.get("webrtc_sensitivity", 3),
                    post_speech_silence_duration=module_config.get("post_speech_silence_duration", 0.6),
                    min_length_of_recording=module_config.get("min_length_of_recording", 1.0),
                    min_gap_between_recordings=module_config.get("min_gap_between_recordings", 1.0),
                    pre_recording_buffer_duration=module_config.get("pre_recording_buffer_duration", 0.2),
                    ensure_sentence_starting_uppercase=module_config.get("ensure_sentence_starting_uppercase", True),
                    ensure_sentence_ends_with_period=module_config.get("ensure_sentence_ends_with_period", True),
                    batch_size=module_config.get("batch_size", 16),
                    beam_size=module_config.get("beam_size", 5),
                    initial_prompt=module_config.get("initial_prompt"),
                    allowed_latency_limit=module_config.get("allowed_latency_limit", 100),
                    preload_model=True
                )
                    
            elif module_type == "static":
                safe_print(f"Initializing static file transcriber...")
                self.transcribers[module_type] = module.DirectFileTranscriber(
                    use_tk_mainloop=False,
                    model=module_config.get("model", "Systran/faster-whisper-large-v3"),
                    language=module_config.get("language", "en"),
                    compute_type=module_config.get("compute_type", "float16"),
                    device=module_config.get("device", "cuda"),
                    gpu_device_index=module_config.get("gpu_device_index", 0),
                    beam_size=module_config.get("beam_size", 5),
                    batch_size=module_config.get("batch_size", 16),
                    vad_aggressiveness=module_config.get("vad_aggressiveness", 2)
                )
                    
            self.log_info(f"{module_type.capitalize()} transcriber initialized successfully")
            return self.transcribers[module_type]
                
        except Exception as e:
            self.log_error(f"Error initializing {module_type} transcriber: {e}")
            return None
    
    def log_info(self, message):
        """Log an info message."""
        logging.info(message)
    
    def log_error(self, message):
        """Log an error message."""
        logging.error(message)
    
    def start_server(self):
        """Start the TCP server to listen for commands from AutoHotkey."""
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        self.log_info(f"TCP server started on {SERVER_HOST}:{SERVER_PORT}")
    
    def _run_server(self):
        """Run the TCP server loop."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((SERVER_HOST, SERVER_PORT))
            server_socket.listen(5)
            server_socket.settimeout(1)  # Allow checking self.running every second
            
            while self.running:
                try:
                    client_socket, addr = server_socket.accept()
                    data = client_socket.recv(1024).decode('utf-8').strip()
                    self.log_info(f"Received command: {data}")
                    
                    # Process command
                    self._handle_command(data)
                    
                    client_socket.close()
                except socket.timeout:
                    continue  # Just a timeout, check self.running and continue
                except Exception as e:
                    self.log_error(f"Error handling client connection: {e}")
        except Exception as e:
            self.log_error(f"Server error: {e}")
        finally:
            server_socket.close()
            self.log_info("TCP server stopped")
    
    def _handle_command(self, command):
        """Process commands received from AutoHotkey."""
        try:
            if command == "TOGGLE_REALTIME":
                self._toggle_realtime()
            elif command == "START_LONGFORM":
                self._start_longform()
            elif command == "STOP_LONGFORM":
                self._stop_longform()
            elif command == "RUN_STATIC":
                self._run_static()
            elif command == "QUIT":
                self._quit()
            else:
                self.log_error(f"Unknown command: {command}")
        except Exception as e:
            self.log_error(f"Error handling command {command}: {e}")
    
    def _toggle_realtime(self):
        """Toggle real-time transcription on/off."""
        # Check if another mode is running
        if self.current_mode and self.current_mode != "realtime":
            safe_print(f"Cannot start real-time mode while in {self.current_mode} mode. Please finish the current operation first.")
            return
            
        if self.current_mode == "realtime":
            # Real-time transcription is already running, so stop it
            safe_print("Stopping real-time transcription...")
            
            try:
                transcriber = self.transcribers.get("realtime")
                if transcriber:
                    transcriber.running = False
                    transcriber.stop()
                self.current_mode = None
                safe_print("Real-time transcription stopped.")
            except Exception as e:
                self.log_error(f"Error stopping real-time transcription: {e}")
        else:
            # Start real-time transcription
            try:
                # Initialize the real-time transcriber if not already done
                transcriber = self.initialize_transcriber("realtime")
                if not transcriber:
                    safe_print("Failed to initialize real-time transcriber.")
                    return
                    
                safe_print("Starting real-time transcription...")
                self.current_mode = "realtime"
                
                # Start real-time transcription in a separate thread
                threading.Thread(target=self._run_realtime, daemon=True).start()
            except Exception as e:
                self.log_error(f"Error starting real-time transcription: {e}")
                self.current_mode = None
    
    def _run_realtime(self):
        """Run real-time transcription in a separate thread."""
        try:
            transcriber = self.transcribers.get("realtime")
            if not transcriber:
                safe_print("Realtime transcriber not available.")
                self.current_mode = None
                return
                
            # Clear any previous text
            transcriber.text_buffer = ""
            
            # Start transcription
            safe_print("Real-time transcription active. Speak now...")
            
            # This will run until stopped
            transcriber.start()
            
            self.log_info("Real-time transcription stopped")
            self.current_mode = None
            
        except Exception as e:
            self.log_error(f"Error in _run_realtime: {e}")
            self.current_mode = None
            
            # Make sure to clean up properly
            try:
                if "realtime" in self.transcribers and self.transcribers["realtime"]:
                    self.transcribers["realtime"].stop()
            except Exception as cleanup_e:
                self.log_error(f"Error during cleanup: {cleanup_e}")
            
            self.current_mode = None
    
    def _start_longform(self):
        """Start long-form recording."""
        # Check if another mode is running
        if self.current_mode:
            safe_print(f"Cannot start long-form mode while in {self.current_mode} mode. Please finish the current operation first.")
            return
            
        try:
            # Initialize the long-form transcriber if not already done
            transcriber = self.initialize_transcriber("longform")
            if not transcriber:
                safe_print("Failed to initialize long-form transcriber.")
                return
                
            safe_print("Starting long-form recording...")
            self.current_mode = "longform"
            transcriber.start_recording()
            
        except Exception as e:
            self.log_error(f"Error starting long-form recording: {e}")
            self.current_mode = None
    
    def _stop_longform(self):
        """Stop long-form recording and transcribe."""
        if self.current_mode != "longform":
            safe_print("No active long-form recording to stop.")
            return
            
        try:
            transcriber = self.transcribers.get("longform")
            if not transcriber:
                safe_print("Long-form transcriber not available.")
                self.current_mode = None
                return
                
            safe_print("Stopping long-form recording and transcribing...")
            transcriber.stop_recording()
            self.current_mode = None
            
        except Exception as e:
            self.log_error(f"Error stopping long-form recording: {e}")
            self.current_mode = None
    
    def _run_static(self):
        """Run static file transcription."""
        # Check if another mode is running
        if self.current_mode:
            safe_print(f"Cannot start static mode while in {self.current_mode} mode. Please finish the current operation first.")
            return
            
        try:
            # Initialize the static transcriber if not already done
            transcriber = self.initialize_transcriber("static")
            if not transcriber:
                safe_print("Failed to initialize static transcriber.")
                return
                
            safe_print("Opening file selection dialog...")
            self.current_mode = "static"
            
            # Run in a separate thread to avoid blocking
            threading.Thread(target=self._run_static_thread, daemon=True).start()
            
        except Exception as e:
            self.log_error(f"Error starting static transcription: {e}")
            self.current_mode = None
    
    def _run_static_thread(self):
        """Run static transcription in a separate thread."""
        try:
            transcriber = self.transcribers.get("static")
            if not transcriber:
                safe_print("Static transcriber not available.")
                self.current_mode = None
                return
                
            # Select and process the file
            transcriber.select_file()
            
            # Wait until transcription is complete
            while transcriber.transcribing:
                time.sleep(0.5)
                
            self.log_info("Static file transcription completed")
            self.current_mode = None
            
        except Exception as e:
            self.log_error(f"Error in static transcription: {e}")
            self.current_mode = None
    
    def _quit(self):
        """Stop all processes and exit."""
        safe_print("Quitting application...")
        self.stop()
        os._exit(0)  # Force exit
    
    def _kill_leftover_ahk(self):
        """Kill any existing AHK processes using our script."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if (
                    proc.info['name'] == 'AutoHotkeyU64.exe'
                    and proc.info['cmdline'] is not None
                    and "STT_hotkeys.ahk" in ' '.join(proc.info['cmdline'])
                ):
                    self.log_info(f"Killing leftover AHK process with PID={proc.pid}")
                    psutil.Process(proc.pid).kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    
    def start_ahk_script(self):
        """Start the AutoHotkey script."""
        # First kill any leftover AHK processes
        self._kill_leftover_ahk()
        
        # Record existing AHK PIDs before launching
        pre_pids = set()
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] == 'AutoHotkeyU64.exe':
                    pre_pids.add(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Launch the AHK script
        ahk_path = os.path.join(self.script_dir, "STT_hotkeys.ahk")
        self.log_info("Launching AHK script...")
        subprocess.Popen(
            [ahk_path],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            shell=True
        )

        # Give it a moment to start
        time.sleep(1.0)
        
        # Find the new AHK process
        post_pids = set()
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] == 'AutoHotkeyU64.exe':
                    post_pids.add(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Store the PID of the new process
        new_pids = post_pids - pre_pids
        if len(new_pids) == 1:
            self.ahk_pid = new_pids.pop()
            self.log_info(f"Detected new AHK script PID: {self.ahk_pid}")
        else:
            self.log_info("Could not detect a single new AHK script PID. No PID stored.")
            self.ahk_pid = None
    
    def stop_ahk_script(self):
        """Kill AHK script if we know its PID."""
        if self.ahk_pid is not None:
            self.log_info(f"Killing AHK script with PID={self.ahk_pid}")
            try:
                psutil.Process(self.ahk_pid).kill()
            except Exception as e:
                self.log_error(f"Failed to kill AHK process: {e}")
    
    def run(self):
        """Run the orchestrator."""
        # Start the TCP server
        self.start_server()
        
        # Start the AutoHotkey script
        self.start_ahk_script()
        
        # Pre-load the longform transcription model at startup as requested
        safe_print("Pre-loading the long-form transcription model...")
        longform_transcriber = self.initialize_transcriber("longform")
        if longform_transcriber:
            # Force complete initialization including the AudioToTextRecorder
            if hasattr(longform_transcriber, 'force_initialize'):
                if longform_transcriber.force_initialize():
                    safe_print("Long-form transcription model fully loaded and ready to use.")
                else:
                    safe_print("Failed to fully initialize the long-form transcription model.")
            else:
                safe_print("Long-form transcription model loaded but may require additional initialization on first use.")
        else:
            safe_print("Failed to pre-load the long-form transcription model.")
        
        # Display startup banner
        if HAS_RICH:
            console.print(Panel(
                "[bold]Speech-to-Text Orchestrator[/bold]\n\n"
                "Control the system using these hotkeys:\n"
                "  [cyan]F2[/cyan]: Toggle real-time transcription\n"
                "  [cyan]F3[/cyan]: Start long-form recording\n"
                "  [cyan]F4[/cyan]: Stop long-form recording and transcribe\n"
                "  [cyan]F10[/cyan]: Run static file transcription\n"
                "  [cyan]F7[/cyan]: Quit application",
                title="Speech-to-Text System",
                border_style="green"
            ))
        else:
            safe_print("="*50)
            safe_print("Speech-to-Text Orchestrator Running")
            safe_print("="*50)
            safe_print("Hotkeys:")
            safe_print("  F2: Toggle real-time transcription")
            safe_print("  F3: Start long-form recording")
            safe_print("  F4: Stop long-form recording and transcribe")
            safe_print("  F10: Run static file transcription")
            safe_print("  F7: Quit application")
            safe_print("="*50)
        
        # Keep the main thread running
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            safe_print("\nKeyboard interrupt received, shutting down...")
        except Exception as e:
            self.log_error(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop all processes and clean up."""
        try:
            if not self.running:
                return

            self.running = False

            # Stop any active transcription mode
            try:
                if self.current_mode == "realtime" and "realtime" in self.transcribers:
                    self.transcribers["realtime"].stop()
                elif self.current_mode == "longform" and "longform" in self.transcribers:
                    self.transcribers["longform"].stop_recording()
                elif self.current_mode == "static" and "static" in self.transcribers and hasattr(self.transcribers["static"], 'transcribing'):
                    # Let it finish naturally
                    pass
            except Exception as e:
                self.log_error(f"Error stopping active mode: {e}")

            # Stop the AutoHotkey script
            self.stop_ahk_script()

            # Only try to join the server thread if we're not currently in it
            current_thread_id = threading.get_ident()
            server_thread_id = self.server_thread.ident if self.server_thread else None

            try:
                if (self.server_thread and self.server_thread.is_alive() and 
                    current_thread_id != server_thread_id):
                    self.server_thread.join(timeout=2)
            except Exception as e:
                self.log_error(f"Error joining server thread: {e}")

            # Clean up resources
            for module_type, transcriber in self.transcribers.items():
                try:
                    if module_type == "realtime":
                        transcriber.stop()
                    elif module_type == "longform" and hasattr(transcriber, 'recorder') and transcriber.recorder:
                        transcriber.clean_up()
                except Exception as e:
                    self.log_error(f"Error cleaning up {module_type} transcriber: {e}")

            self.log_info("Orchestrator stopped successfully")

        except Exception as e:
            self.log_error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    orchestrator = STTOrchestrator()
    orchestrator.run()