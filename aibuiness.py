import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq

import httpx
from fastapi.responses import Response

app = FastAPI()

SECRET = os.environ.get("AI_SERVICE_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
TAPRA_KEY = os.environ.get("TAPRA_API_KEY")
TAPRA_TTS_URL = os.environ.get("TAPRA_TTS_URL")  # from tapra.tn/docs

client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None


class SpeakRequest(BaseModel):
    text: str
    voice: str = "youssef"


@app.post("/speak")
def speak(req: SpeakRequest, x_api_key: str = Header(None)):
    if not SECRET or x_api_key != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not TAPRA_KEY or not TAPRA_TTS_URL:
        raise HTTPException(status_code=500, detail="Tapra is not configured")

    # Check the exact auth header and field names in tapra.tn/docs
    r = httpx.post(
        TAPRA_TTS_URL,
        headers={"Authorization": f"Bearer {TAPRA_KEY}"},
        json={"text": req.text, "voice": req.voice},
        timeout=60,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Tapra request failed")
    return Response(content=r.content, media_type="audio/wav")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
SECOND_KEY = os.environ.get("SECOND_AI_API_KEY")

# First AI (Groq)
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# Second AI: create its client here with SECOND_KEY.
# Example for an OpenAI-compatible provider:
# from openai import OpenAI
# client2 = OpenAI(api_key=SECOND_KEY, base_url="https://PROVIDER-URL/v1")


class CallRequest(BaseModel):
    phone: str
    prompt: str
    knowledge: str = ""


@app.get("/health")
def health():
    return "ok"


@app.post("/generate")
def generate(req: CallRequest, x_api_key: str = Header(None)):
    # Only your Node server may call this
    if not SECRET or x_api_key != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")
