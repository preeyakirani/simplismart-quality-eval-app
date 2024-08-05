from deepgram import (DeepgramClient, PrerecordedOptions, FileSource)
import time
import os
from dotenv import load_dotenv, dotenv_values

# from test_run import *
# from get_dataset_and_normalizer import *

load_dotenv()

def get_transcription_deepgram(audio_file_path, language):
    try:
        client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
        with open(audio_file_path, "rb") as file:
            buffer_data = file.read()
        payload: FileSource = {"buffer": buffer_data}

        if language == None:
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                detect_language=True
            )
        else:
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                language=language
            )

        st = time.time()
        response = client.listen.prerecorded.v("1").transcribe_file(payload, options)
        et = time.time()
        return response, round((et-st),2)

    except Exception as e:
        print(f"Exception: {e}")


def main():
    res = get_transcription_deepgram("audio_files/numbers_detection_1.mp3", "en")
    print(res)

# main()

# def main(x,y):

#     for i in range(x,y):
#         print("Iteration: " + str(i))

#         prediction, time_taken, truth = test_hindi2(i, get_transcription_deepgram, "hi")
#         evaluate_wer = get_evaluate_wer(prediction, truth, normalise_hindi_text)
#         werpy_wer = get_werpy_wer(prediction, truth, normalise_hindi_text)
#         add_to_csv("Deepgram", "hindi", prediction, truth, evaluate_wer, werpy_wer, time_taken, 
#                     normalise_hindi_text, "./collected_csv_files/compare-all-hindi-10samples-fleurs.csv")
        
#     print("done.")

# main(2,10)