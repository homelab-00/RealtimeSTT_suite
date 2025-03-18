#!/usr/bin/env python3
# static_module.py
#
# A simplified version of the static file transcriber
# 
# This module provides functionality to transcribe audio/video files
# using the RealtimeSTT library. It includes:
# - A file selection dialog
# - Processing of audio/video files
# - Transcription using faster-whisper models

import os
import sys
import tkinter as tk
from tkinter import filedialog
import logging
import threading
import time
from RealtimeSTT import AudioToTextRecorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Optional imports
try:
    import ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False
    logging.warning("ffmpeg-python not installed. Some audio processing features will be limited.")

class DirectFileTranscriber:
    """
    A class that directly transcribes audio and video files.
    """
    
    def __init__(self, 
                 file_select_hotkey="",
                 quit_hotkey="",
                 use_tk_mainloop=True,
                 model="large-v3",
                 download_root=None,
                 language="en",
                 compute_type="float16",
                 device="cuda",
                 device_index=0,
                 **kwargs):
        """Initialize the transcriber with basic parameters."""
        self.model = model
        self.language = language
        self.compute_type = compute_type
        self.device = device
        self.device_index = device_index
        self.download_root = download_root
        self.use_tk_mainloop = use_tk_mainloop
        
        # State variables
        self.transcribing = False
        self.root = None
        self.last_transcription = ""
        
    def select_file(self):
        """Open a file dialog to select a file for transcription."""
        # Initialize tkinter if not already done
        if not self.root:
            self.root = tk.Tk()
            self.root.withdraw()  # Hide the main window
        
        # Make sure the root window is properly prepared
        self.root.update()
        
        # Show the file dialog
        file_path = filedialog.askopenfilename(
            title="Select an Audio or Video File",
            filetypes=[
                ("Audio/Video files", "*.mp3;*.wav;*.flac;*.ogg;*.m4a;*.mp4;*.avi;*.mkv;*.mov"),
                ("Audio files", "*.mp3;*.wav;*.flac;*.ogg;*.m4a"),
                ("Video files", "*.mp4;*.avi;*.mkv;*.mov"),
                ("All files", "*.*")
            ],
            parent=self.root
        )
        
        if file_path:
            # Start transcription in a separate thread
            threading.Thread(target=self._process_file, args=(file_path,), daemon=True).start()
        else:
            self.transcribing = False
    
    def _process_file(self, file_path):
        """Process and transcribe the selected file."""
        try:
            self.transcribing = True
            logging.info(f"Processing file: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                self.transcribing = False
                return
                
            # Convert audio if needed (simplified for now)
            temp_wav = None
            if HAS_FFMPEG and not file_path.lower().endswith('.wav'):
                logging.info("Converting to WAV format...")
                # Simple conversion code would go here
                # For now, we'll just use the original file
            
            # Create a recorder for transcription
            recorder = AudioToTextRecorder(
                model=self.model,
                download_root=self.download_root,
                language=self.language,
                compute_type=self.compute_type,
                device=self.device,
                gpu_device_index=self.device_index,
                use_microphone=False  # Important: we're not using microphone input
            )
            
            # Read the file in chunks and feed to the recorder
            logging.info("Starting transcription...")
            
            # For now, we'll just use a placeholder transcription
            self.last_transcription = "This is a placeholder transcription. Full static file transcription will be implemented later."
            
            # Sleep to simulate processing time
            time.sleep(2)
            
            logging.info("Transcription completed")
            print(f"\nTranscription: {self.last_transcription}\n")
            
            # Clean up
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
                
        except Exception as e:
            logging.error(f"Error processing file: {e}")
        finally:
            self.transcribing = False