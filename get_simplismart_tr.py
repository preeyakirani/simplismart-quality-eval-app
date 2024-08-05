import requests
import json
import difflib
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# from test_run import *
# from get_dataset_and_normalizer import *

def get_transcription_simplismart(audio_file_path, 
                                  task="transcribe", 
                                  language="en", 
                                  initial_prompt=None, 
                                  beam_size=5, 
                                  best_of=5, 
                                  patience=1.0, 
                                  vad_onset=0.5, 
                                  vad_offset=0.363, 
                                  min_duration_on=0, 
                                  min_duration_off=0, 
                                  pad_onset=0, 
                                  pad_offset=0):
    try:
        api_url = "https://demo.whisper.simplismart.ai/v3/models/api/predict"
        params = {
            "task": task,
            "initial_prompt": initial_prompt,
            "language": language,
            "beam_size": beam_size,  # Default value
            "best_of": best_of,  # Default value
            "patience": patience,  # Default value
            "word_timestamps": True,
            "diarize": False,
            "min_speakers": None,
            "max_speakers": None,
            "batch_size": 24,
            "vad_onset": vad_onset,  # Adjusted from speech_threshold
            "vad_offset": vad_offset,  # Adjusted from speech_threshold
            "min_duration_off": min_duration_off,  # Adjusted from min_silence_duration_ms
            "min_duration_on": min_duration_on,  # Adjusted from min_speech_duration_ms
            "max_duration": 30,  # Default value
            "pad_offset": pad_offset,  # Default value
            "pad_onset": pad_onset  # Default value
        }

        files = {
            'audio': (f'sample.mp3', open(audio_file_path, 'rb'), 'audio/mpeg'),
        }

        response = requests.post(api_url, params=params, files=files)
        response_json = response.json()
        return response_json, round(response_json["time_taken"],2)
    
    except Exception as e:
        print(f"Exception: {e}")
    


# def main():
#     res, time_taken = get_transcription_simplismart("audio_files/numbers_detection_1.mp3")
#     print(res)

# main()