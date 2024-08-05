import boto3
import os
import datetime
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()

s3_dict = {
    "simple_english.wav": "s3://simplismart-qual-eval-app/simple_english.mp3",
    "numbers_detection_1.wav": "s3://simplismart-qual-eval-app/numbers_detection_1.mp3",
    "numbers_detection_2.wav": "s3://simplismart-qual-eval-app/numbers_detection_2.mp3",
    "multiple_speakers.wav": "s3://simplismart-qual-eval-app/multiple_speakers.mp3",
    "hindi_transcription.wav": "s3://simplismart-qual-eval-app/hindi_transcription.mp3"
}

def start_transcription_job(job_name, media_uri, language_code, transcribe_client):
    try:
        if (language_code == None):
            job_args = {
                "TranscriptionJobName": job_name,
                "Media": {"MediaFileUri": media_uri},
                "MediaFormat": "mp3",
                "IdentifyLanguage": True
            }
        else:
            job_args = {
                "TranscriptionJobName": job_name,
                "Media": {"MediaFileUri": media_uri},
                "MediaFormat": "mp3",
                "LanguageCode": language_code,
            }

        response = transcribe_client.start_transcription_job(**job_args)
        job = response["TranscriptionJob"]

    except ClientError as e:
        print(e)
        print("Couldn't start transcription job %s.", job_name)
    else:
        return job



def get_transcription_job(job_name, transcribe_client):
    try:
        response = transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name
        )
        job = response["TranscriptionJob"]
    except ClientError as e:
        print(e)
        print("Couldn't get job %s.", job_name)
    else:
        return job


def get_transcription_aws(audio_file_path, language):

    file_name = audio_file_path.split("/")[-1]

    if file_name in s3_dict:
        file_uri = s3_dict[file_name]
    else:
        s3_client = boto3.client("s3",
                             aws_access_key_id = os.getenv('AWS_ACCESS_KEY'),
                             aws_secret_access_key = os.getenv('AWS_SECRET_KEY'),
                             region_name = "ap-southeast-2")
        s3_client.upload_file(audio_file_path, "simplismart-qual-eval-app", "file.mp3")
        file_uri = "s3://simplismart-qual-eval-app/file.mp3"
    
    job_name = "job_" + datetime.datetime.now().strftime("%d-%m-%y_%H.%M.%f")

    transcription_client = boto3.client('transcribe',
                                        aws_access_key_id = os.getenv('AWS_ACCESS_KEY'),
                                        aws_secret_access_key = os.getenv('AWS_SECRET_KEY'),
                                        region_name = "ap-southeast-2")
    st = time.time()
    job_sent = start_transcription_job(job_name, file_uri, language, transcription_client)

    while True:
        job_received = get_transcription_job(job_name, transcription_client)
        
        if (job_received["TranscriptionJobStatus"] == "COMPLETED"):
            et = time.time()
            response = requests.get(job_received["Transcript"]["TranscriptFileUri"])
            return response.json(), round((et-st), 2)
        elif (job_received["TranscriptionJobStatus"] == "IN_PROGRESS"):
            continue
        elif (job_received["TranscriptionJobStatus"] == "FAILED"):
            print("job" + job_name + "failed!")
            return {"status": "FAIL"}, 0


# def main():
    # transcribe_client = boto3.client('transcribe',
    #                                     aws_access_key_id = os.getenv('AWS_ACCESS_KEY'),
    #                                     aws_secret_access_key = os.getenv('AWS_SECRET_KEY'),
    #                                     region_name = "ap-southeast-2")
    # job_args = {
    #     "TranscriptionJobName": "testingMultipleSpeakers2",
    #     "Media": {"MediaFileUri": "s3://simplismart-qual-eval-app/multiple_speakers.mp3"},
    #     "MediaFormat": "mp3",
    #     "LanguageCode": "en-US"
    # }

    # job = transcribe_client.start_transcription_job(**job_args)
    # print(job)
    
    # print(transcribe_client.get_transcription_job(
    #         TranscriptionJobName="testingMultipleSpeakers2"
    #     ))
    
    # res, time = get_transcription_aws('audio_files/hindi_transcription.mp3', 'transcribe', 'en-US')
    # print(res)

# main()