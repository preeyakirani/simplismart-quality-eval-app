import openai
import time
import os
from dotenv import load_dotenv

# from test_run import *
# from get_dataset_and_normalizer import *

load_dotenv()

def get_transcription_whisper(audio_file_path, task, language):
    try:
        client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

        with open(audio_file_path, "rb") as file:
            if (task == "transcribe"):
                st = time.time()
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    language=language,
                    response_format="verbose_json",
                    file=file,
                    timestamp_granularities=['word']
                )
                et = time.time()
            elif (task == "translate"):
                st = time.time()
                response = client.audio.translations.create(
                    model="whisper-1",
                    response_format="verbose_json",
                    file=file,
                )
                et = time.time()

        return response, round((et-st),2)
    
    except Exception as e:
        print(f"Exception: {e}")


# def main():
#     res, time = get_transcription_whisper('audio_files/multiple_speakers.mp3', 'en')
#     print(res)

# main()