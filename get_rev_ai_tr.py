from rev_ai import apiclient, language_identification_client
from rev_ai.models.asynchronous.translation_options import TranslationOptions
from rev_ai.models.asynchronous.translation_language_options import TranslationLanguageOptions
from rev_ai.models.asynchronous.translation_model import TranslationModel
import time
import re
import os
from dotenv import load_dotenv
import json
import requests

# from test_run import *
# from get_dataset_and_normalizer import *

load_dotenv()


def identify_language(audio_file_path):
    lang_client = language_identification_client.LanguageIdentificationClient(os.getenv("REV_AI_API_KEY"))
    lang_job = lang_client.submit_job_local_file(audio_file_path)

    while (lang_job.status.name == "IN_PROGRESS"):
        lang_details = lang_client.get_job_details(lang_job.id)

        if (lang_details.status.name == "COMPLETED"):
            lang_res = lang_client.get_result_object(lang_job.id)
            language = lang_res.top_language
            break
        
        elif (lang_details.status.name == "FAILED"):
            language = "en"
            break
    
    return language


def get_transcription_rev_ai(audio_file_path, task, language):
    try:
        client = apiclient.RevAiAPIClient(os.getenv("REV_AI_API_KEY"))
        st = time.time()

        # identify language if necessary
        if (language == None):
            language = identify_language(audio_file_path)

        # next task
        if (task == "translate"):
            job = client.submit_job_local_file(audio_file_path,
                                               language=language,
                                               skip_diarization=True, 
                                               skip_punctuation=False,
                                               translation_config=TranslationOptions(
                                                    target_languages=[TranslationLanguageOptions(
                                                        language="en",
                                                        model=TranslationModel.STANDARD
                                                    )]
                                                ))
            
        elif (task == "transcribe"):
            job = client.submit_job_local_file(audio_file_path, 
                                            language=language,
                                            skip_diarization=True, 
                                            skip_punctuation=False)

        while (job.status.name == 'IN_PROGRESS'):
            details = client.get_job_details(job.id)

            # if successful, print result
            if (details.status.name == 'TRANSCRIBED'):
                et = time.time()

                if (task == "transcribe"):
                    rev_res_text = client.get_transcript_text(job.id)
                    rev_res_words = client.get_transcript_json(job.id)
                    return rev_res_text, rev_res_words, round((et-st),2)
                elif (task == "translate"):
                     rev_translated_res_text = client.get_translated_transcript_text(job.id, "en")
                     rev_translated_res_words = client.get_translated_transcript_json(job.id, "en")
                     return rev_translated_res_text, rev_translated_res_words, round((et-st),2)

            # if unsuccessful, print error
            if (details.status.name == 'FAILED'):
                print("Job failed: " + details.failure_detail)
                return

    except Exception as e:
        print(f"Exception: {e}")
        return "ERROR", None, 0.00


# def main():
#     transcription, words, time = get_transcription_rev_ai('audio_files/hindi_translation.mp3', "translate", "hi")
#     # client = apiclient.RevAiAPIClient(os.getenv("REV_AI_API_KEY"))
#     # transcription = client.get_translated_transcript_text('paw3IwpWXyhfRQjs', 'en')
#     # words = client.get_translated_transcript_json('paw3IwpWXyhfRQjs', 'en')
#     print(transcription)
#     print(words)

# main()