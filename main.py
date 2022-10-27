from chardet import detect
from fastapi import FastAPI
from routers import  detect

app = FastAPI()

# import os
# os.environ["EAI_USERNAME"] = 'YOUR_USERNAME'
# os.environ["EAI_PASSWORD"] = 'YOUR_PASSWORD'

app.include_router(detect.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Hate Speech Detection...!"}



