from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(title="SignMeta Backend - GROUP B", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are an expert Sign Language interpreter.
You receive hand landmark coordinates (21 points per hand, x/y/z) from MediaPipe.
Your task:
1. Interpret the gesture pattern from the coordinates.
2. Map it to a Turkish Sign Language (TİD) or American Sign Language (ASL) sign.
3. Return a grammatically correct sentence in the target language.
4. Fix sign language grammar structure (topic-comment syntax).
5. Be concise. Return ONLY the interpreted text, nothing else.
If coordinates are unclear, return: "Anlaşılamadı / Not understood"
"""

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY ortam değişkeni ayarlanmamış. Lütfen Render panosundan API anahtarını ekleyin."
        )
    from groq import Groq
    return Groq(api_key=api_key)

class Landmark(BaseModel):
    x: float
    y: float
    z: float

class HandData(BaseModel):
    landmarks: List[Landmark]
    handedness: Optional[str] = "Right"
    timestamp: Optional[int] = 0

class TranslateRequest(BaseModel):
    hands: List[HandData]
    language: Optional[str] = "tr"
    session_id: Optional[str] = ""

class TranslateResponse(BaseModel):
    text: str
    confidence: float
    language: str
    served_by: str = "B"

class TextToSignRequest(BaseModel):
    text: str
    language: Optional[str] = "tr"

class TextToSignResponse(BaseModel):
    animation_sequence: List[str]
    description: str
    served_by: str = "B"

@app.get("/")
def root():
    return {"status": "SignMeta Backend GROUP-B Running", "group": "B", "version": "1.0.0"}

@app.get("/health")
def health():
    groq_configured = bool(os.environ.get("GROQ_API_KEY"))
    return {"status": "ok", "group": "B", "groq_configured": groq_configured}

@app.post("/translate", response_model=TranslateResponse)
async def translate_sign(request: TranslateRequest):
    client = get_groq_client()
    try:
        coord_text = ""
        for i, hand in enumerate(request.hands):
            coord_text += f"\nHand {i+1} ({hand.handedness}):\n"
            for j, lm in enumerate(hand.landmarks):
                coord_text += f"  Point {j}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}\n"

        user_message = f"""Interpret these hand coordinates as a sign language gesture.
Target language: {'Turkish' if request.language == 'tr' else 'English'}
Coordinates:
{coord_text}
Return ONLY the interpreted text."""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=256,
        )
        result = completion.choices[0].message.content.strip()
        return TranslateResponse(text=result, confidence=0.85, language=request.language, served_by="B")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API hatası: {str(e)}")

@app.post("/text-to-sign", response_model=TextToSignResponse)
async def text_to_sign(request: TextToSignRequest):
    client = get_groq_client()
    try:
        prompt = f"""Convert this text to sign language animation commands.
Text: "{request.text}"
Language: {'Turkish Sign Language (TİD)' if request.language == 'tr' else 'ASL'}
Format:
SIGNS: SIGN1, SIGN2, SIGN3
DESCRIPTION: brief description"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a sign language expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512,
        )
        content = completion.choices[0].message.content.strip()
        signs = []
        description = content
        for line in content.split("\n"):
            if line.startswith("SIGNS:"):
                signs = [s.strip() for s in line.replace("SIGNS:", "").split(",")]
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
        return TextToSignResponse(animation_sequence=signs, description=description, served_by="B")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API hatası: {str(e)}")
