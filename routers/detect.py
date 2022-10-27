from fastapi import APIRouter, Response
from schema import input_sm
from get_logger import _in_logger
from src import detection

router = APIRouter(
    prefix="/hate_speech_detection",
    tags=["hate speech"],
    responses={404: {"description": "Not found"}},
)   

@router.post("/text")
async def text_detection(input_data: input_sm.TEXT_INPUT):
    output_json = detection.text_init(input_text = input_data.input_text, model_name=input_data.model_size)
    _in_logger.info("Transcibe & Hate Speech Detection are Compeleted")
    return output_json

@router.post("/from_audio_path")
async def audio_detection(input_data: input_sm.AUDIO_INPUT):
    output_json = detection.audio_init(audio_path = input_data.audio_path, model_name=input_data.model_size)
    _in_logger.info("Transcibe & Hate Speech Detection are Compeleted")
    return output_json

@router.post("/from_youtube_url")
async def yt_detection(input_data: input_sm.YT_INPUT):
    output_json = detection.yt_init(youtube_url = input_data.url, model_name=input_data.model_size)
    _in_logger.info("Transcibe & Hate Speech Detection are Compeleted")
    return output_json

@router.post("/from_video_path")
async def video_detection(input_data: input_sm.VIDEO_INPUT):
    output_json = detection.video_init(video_path = input_data.video_path, model_name=input_data.model_size)
    _in_logger.info("Transcibe & Hate Speech Detection are Compeleted")
    return output_json