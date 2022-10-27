# Hate Speech Detection

This project is to detect hate speeches from Video/Youtube Url/Audio or direct text itself
This app fits into third category of the hackathon, which is "Hate Speech Detection" and used expert.ai NL API to detect sentiment and hate speech.
The objective of the project is to reduce hate speeches on social media.

## Summary

### Approach

1. Converting the video to audio (for youtube used youtube API, for video used moviepy)
2. Using whisper pre_trained model detect text from audio
  -  [Whisper Framework](https://github.com/openai/whisper)
3. Translating the converted text to english if required(Optional)
4. Finally detecting the sentiment & hate speech using expertai API's

#### Assumptions and Conclusions

* Using base model as default in whisper(can change it to any size-(tiny, base, small, medium, large)), as the model size increases accuracy increases for whisper
* By using a GPU system we can reduce a lot of latecy(for transcribing)
* For sentiment, if the value is between 0-10 used as neutral, >10 - positive, <0 - negative (After trying multiple examples)

### Future work

We can re-train the whisper model to get better accuracy for audio transcribing. Currently the app will work only on recorded videos or audios, in future we can integrate to live videos.

### Install and Run

- Buil the docker container
    ```
    sudo docker build -t "image_name" ./
    ```
- Run the docker container
    ```
    sudo docker run --env-file env.list --name "container_name" -p 80:80 "image_name"
    ```
- First time download the model from "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt" and put it under model directory otherwise it will take some time to run the api

- Then open any API client or you can open swagger UI by :- http://127.0.0.1/docs
    - Inference end point : **/hate_speech_detection/text**(Detection using text input)
        
        sample parameters:-

            {
                "input_text": "",
                "model_size": "base"
            }
    - Train end point :- **/hate_speech_detection/from_youtube_url**(Detection using youtube url)
        
        sample parameters:-

            {
                "url": "",
                "model_size": "yolov5l.pt"
            }
    - Inference end point : **/hate_speech_detection/from_video_path**(Detection using video path)
        
        sample parameters:-

            {
                "video_path": "",
                "model_size": "base"
            }
    - Inference end point : **/hate_speech_detection/from_audio_path**(Detection using audio path)
        
        sample parameters:-

            {
                "audio_path": "data/test_audio.mp3",
                "model_size": "base"
            }
