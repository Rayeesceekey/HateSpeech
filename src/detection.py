from googletrans import Translator
import youtube_dl
import subprocess
import whisper
import getopt
import torch
import sys
import re
import math
import json
import moviepy.editor as mp
from pathlib import Path
from expertai.nlapi.cloud.client import ExpertAiClient
client = ExpertAiClient()

from whisper.model import Whisper, ModelDimensions

_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
}


# Select speech recognition model
##***** Model chosen as 'base' currently, there are options of (tiny, base, small, medium, large) *****
##***** Based on System config, better to chosse larger models for better accuracy

def load_model():
    my_file = Path("models/base.pt")
    if my_file.exists():
        global MODEL
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with open("models/base.pt", "rb") as fp:
            checkpoint = torch.load(fp, map_location=device)

        dims = ModelDimensions(**checkpoint["dims"])
        MODEL = Whisper(dims)
        MODEL.load_state_dict(checkpoint["model_state_dict"])

        MODEL = MODEL.to(device)
    else:
        load_other_model("base")


def load_other_model(model_name):
    global MODEL
    MODEL = whisper.load_model(model_name)

TRANSLATOR = Translator()
language= 'en'

def youtube_to_audio(youtube_url,audiofile_path):
    video_info = youtube_dl.YoutubeDL().extract_info(url=youtube_url,download=False)
    options={
        'format':'bestaudio/best',
        'keepvideo':False,
        'outtmpl':audiofile_path,
    }
    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])
        
def video_audio_convert(video_path,audio_path):
    my_clip = mp.VideoFileClip(video_path)
    my_clip.audio.write_audiofile(audio_path)
        
def get_transcribe(audio_file_path):
    # Transcribe the text
    result = MODEL.transcribe(audio_file_path)
    return result["text"]

def translate_lng(text):
    translation = TRANSLATOR.translate(text)
    return translation.text

def get_sentiment(input_text):
    output = client.specific_resource_analysis(
                                        body={"document": {"text": input_text}}, 
                                        params={'language': language, 'resource': 'sentiment'
                                    })
    sentiment_score = output.sentiment.overall
    ## Defining sentiment based on score (Identified from different examples)
    if math.isnan(sentiment_score) or 0 <= sentiment_score <= 10:
        return "NEUTRAL"
    elif sentiment_score > 10:
        return "POSITIVE"
    else:
        return "NEGATIVE"
    
def hate_speech_detection(input_text):
    ## split to seperate sentences
    text_arr = input_text.split(".")
    categories = []
    extracted_dict = {}
    for text in text_arr:
        output = client.detection(body={"document": {"text": text}}, params={'detector': 'hate-speech', 'language': language})
        for category in output.categories:
            categories.append(category.hierarchy)
        for idx,extraction in enumerate(output.extractions):
            extracted_dict[idx] = {}
            for field in extraction.fields:
                extracted_dict[idx][field.name] = field.value
    return categories,extracted_dict

def make_final_output(sentiment,categories,extracted_dict):
    out_dict = {}
    out_dict["Overall Sentiment of the Speech"] = sentiment
    out_dict["Hate Speech"] = {}
    tmp_arr = []
    if len(categories) > 0:
        out_dict["Hate Speech"]["Hate Speech Detected"] = "YES" 
        out_dict["Hate Speech"]["Categories"] = categories
        for _, value in extracted_dict.items():
            for k, v in value.items():
                tmp_arr.append(f"{k} : {v}")
        out_dict["Hate Speech"]["Details"] = tmp_arr
    else:
        out_dict["Hate Speech"]["Hate Speech Detected"] = "NO"
    return out_dict

def get_detection_data(audio_file_path,text=False,input_text = ""):
    if text == False:
        ## Fn to get prediction
        output_text = get_transcribe(audio_file_path)
    else:
        output_text = input_text

    ## Call translate fn if translation required(if text is some other language)
    #output_text = translate_lng(output_text)

    ## Call fn to get the sentiment for the identified doc
    sentiment = get_sentiment(output_text)
    ## Fn to detect hate speech and categorise
    categories,extracted_dict = hate_speech_detection(output_text)
    return sentiment,categories,extracted_dict

def yt_init(youtube_url,model_name):
    if model_name != "base":
        load_other_model(model_name)
    else:
        load_model()
    audiofile_path = "data/output_audio.mp3"
    # Convert Yoututbe video to mp3 audio file to transcribe to text
    youtube_to_audio(youtube_url,audiofile_path)
    ## Call detection fn
    sentiment,categories,extracted_dict = get_detection_data(audiofile_path)
    return make_final_output(sentiment,categories,extracted_dict)

def video_init(video_path,model_name):
    if model_name != "base":
        load_other_model(model_name)
    else:
        load_model()
    audiofile_path = "data/output_audio.mp3"
    ### If input is a video_path
    video_audio_convert(video_path,audiofile_path)
    ## Call detection fn
    sentiment,categories,extracted_dict = get_detection_data(audiofile_path)
    return make_final_output(sentiment,categories,extracted_dict)

def audio_init(audio_file_path,model_name):
    if model_name != "base":
        load_other_model(model_name)
    else:
        load_model()
    sentiment,categories,extracted_dict = get_detection_data(audio_file_path)
    return make_final_output(sentiment,categories,extracted_dict)

def text_init(input_text,model_name):
    if model_name != "base":
        load_other_model(model_name)
    else:
        load_model()
    sentiment,categories,extracted_dict = get_detection_data("",text=True,input_text = input_text)
    return make_final_output(sentiment,categories,extracted_dict)
    