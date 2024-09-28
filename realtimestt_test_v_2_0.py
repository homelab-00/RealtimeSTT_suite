from RealtimeSTT import AudioToTextRecorder

def text_detected(text):
    print(f"\r{text}", end="")

def process_text(text):
    print(f"\nFinal transcription: {text}")

def main():
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
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': True,
    }

    with AudioToTextRecorder(**recorder_config) as recorder:
        try:
            while True:
                recorder.text(process_text)
        except KeyboardInterrupt:
            print("\nExiting...")
            recorder.shutdown()

if __name__ == '__main__':
    main()
