if __name__ == '__main__':

    import os
    import sys
    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()

    import sys
    sys.path.insert(0, './')  # This assumes audio_recorder.py is same directory as this script
    from audio_recorder_v_1_0 import AudioToTextRecorder

    recorder = AudioToTextRecorder(
        spinner=False,
        silero_sensitivity=0.01,
        model="tiny.en",
        language="en",
        )

    print("Say something...")
    
    try:
        while (True):
            print("Detected text: " + recorder.text())
    except KeyboardInterrupt:
        print("Exiting application due to keyboard interrupt")
