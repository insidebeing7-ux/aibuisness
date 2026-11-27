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
        "Write ONLY the exact words to be spoken aloud: no titles, no markdown, "
        "no stage directions, no emojis, no brackets.\n"
        "Reply in the language of the company's instructions. For Tunisian Arabic, write "
        "Arabic script and keep French and English words in Latin letters.\n"
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


VOICES = {"youssef", "zeineb", "karim", "sarra", "anis", "rim",
          "hamza", "mariem", "mehdi", "ines", "walid"}


def tapra_cmd():
    # The `tapra` command is installed with the tapra package
    return shutil.which("tapra") or os.path.join(os.path.dirname(sys.executable), "tapra")


@app.post("/speak")
def speak(req: SpeakRequest, x_api_key: str = Header(None)):
    check_key(x_api_key)
    if not TAPRA_KEY:
        raise HTTPException(status_code=500, detail="TAPRA_API_KEY is not set")

    text = req.text.strip().lstrip("-–— ")[:1500]  # Tapra bills per character
    if not text:
        raise HTTPException(status_code=400, detail="No text")
    voice = req.voice if req.voice in VOICES else "youssef"

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "speech.wav")
        p = subprocess.run(
            [tapra_cmd(), "speak", text, "--voice", voice, "-o", out, "-y"],
            env={**os.environ, "TAPRA_API_KEY": TAPRA_KEY},
            capture_output=True, text=True, timeout=90,
        )
        if p.returncode != 0 or not os.path.exists(out):
            print("Tapra error:", p.stdout, p.stderr)
            raise HTTPException(status_code=502, detail="Tapra failed")
        with open(out, "rb") as f:
            data = f.read()

    return Response(content=data, media_type="audio/wav")
