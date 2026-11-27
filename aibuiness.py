import os
import shutil
import subprocess
import sys
import tempfile
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse, Response, PlainTextResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

SECRET = os.environ.get("AI_SERVICE_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
TAPRA_KEY = os.environ.get("TAPRA_API_KEY")

# AI #1: Groq writes the call script
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None




def check_key(x_api_key):
    # Only your Node server may call this service
    if not SECRET or x_api_key != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


class CallRequest(BaseModel):
    phone: str
    prompt: str
    knowledge: str = ""


class SpeakRequest(BaseModel):
    text: str
    voice: str = "youssef"


@app.get("/health")
def health():
    return PlainTextResponse("ok")


@app.post("/generate")
def generate(req: CallRequest, x_api_key: str = Header(None)):
    check_key(x_api_key)
    if client is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")

    system = (
        "You are an AI phone assistant calling a client on behalf of a company.\n"
        "Follow the company's instructions below.\n"
        "Speak in short, natural sentences, as if talking on the phone.\n"
        "Answer questions only from the KNOWLEDGE section. If the answer is not "
        "there, say you will arrange a callback from a person. Never invent prices or facts.\n\n"
        f"COMPANY INSTRUCTIONS:\n{req.prompt}\n\n"
        f"KNOWLEDGE:\n{req.knowledge or '(no files were provided)'}"
    )

    def stream():
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Write what you say to the client at {req.phone}, from the greeting to the goodbye."},
            ],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


@app.post("/speak")
def speak(req: SpeakRequest, x_api_key: str = Header(None)):
    check_key(x_api_key)
    if tapra_client is None:
        raise HTTPException(status_code=500, detail="Tapra is not configured")

    # TODO: confirm the exact method name with: tapra speak "..." --dry-run
    audio = tapra_client.speech.create(text=req.text, voice=req.voice)
    return Response(content=audio, media_type="audio/wav")
