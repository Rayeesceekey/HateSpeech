from pydantic import BaseModel
from typing import Optional

class YT_INPUT(BaseModel):
    url : str
    model_size : Optional[str] = "base"

class VIDEO_INPUT(BaseModel):
    video_path : str
    model_size : Optional[str] = "base"

class AUDIO_INPUT(BaseModel):
    audio_path : str
    model_size : Optional[str] = "base"

class TEXT_INPUT(BaseModel):
    input_text : str
    model_size : Optional[str] = "base"