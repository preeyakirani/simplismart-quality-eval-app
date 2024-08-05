import assemblyai as aai
import time
import os
from dotenv import load_dotenv, dotenv_values

# from test_run import *
# from get_dataset_and_normalizer import *

load_dotenv()

def get_transcription_assemblyai(audio_file_path, language):
    try:
        aai.settings.api_key = os.getenv("ASSEMBLY_AI_API_KEY")
        transcriber = aai.Transcriber()

        if (language == None):
            config = aai.TranscriptionConfig(language_detection=True)
        else:
            config = aai.TranscriptionConfig(language_code=language)

        st = time.time()
        transcript = transcriber.transcribe(data=audio_file_path, config=config)
        et = time.time()
        return transcript, round((et-st),2)
    except Exception as e:
        print(f"Exception: {e}")


def main():
    res, time = get_transcription_assemblyai("audio_files/numbers_detection_1.mp3")
    print(res.words)
    print(res)


# main()
# def main():
#     transcript = get_transcription_assemblyai("/Users/preeyakirani/Downloads/audio.mp3")
#     print("got transcript")
#     with open("assemblyai.txt", "w") as file:
#         for utterance in transcript.utterances:
#             file.write(f"Speaker {utterance.speaker}: {utterance.text}")
#             file.write("\n")

# main()