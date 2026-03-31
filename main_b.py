from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import base64
import numpy as np
from PIL import Image
import io
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI(title="SignMeta Backend - GROUP B", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe Hand Landmarker - using bundled model
_base_options = python.BaseOptions(model_asset_path="assets/hand_landmarker.task")
_options = vision.HandLandmarkerOptions(base_options=_base_options, num_hands=2)
_hand_landmarker = None

def get_hand_landmarker():
    global _hand_landmarker
    if _hand_landmarker is None:
        _hand_landmarker = vision.HandLandmarker.create_from_options(_options)
    return _hand_landmarker

SYSTEM_PROMPT = """Sen uzman bir İşaret Dili yorumcususun.
Görevini:
1. El pozisyonu koordinatlarını (21 nokta, x/y/z) analiz et
2. Türk İşaret Dili (TİD) veya Amerikan İşaret Dili (ASL) olarak yorumla
3. Hedef dilde gramatikally doğru cümle döndür
4. Kısaca yanıt ver, sadece yorumlanmış metni döndür
Koordinatlar net değilse: "Anlaşılamadı" döndür
"""

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY ortam değişkeni ayarlanmamış."
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

class ImageRequest(BaseModel):
    image: str  # base64 encoded
    language: Optional[str] = "tr"

@app.get("/")
def root():
    return {"status": "SignMeta Backend GROUP-B Running", "group": "B", "version": "1.0.0", "mode": "server-side-mediapipe"}

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
            hand_type = "Sağ" if hand.handedness == "Right" else "Sol"
            coord_text += f"\nEl {i+1} ({hand_type}):\n"
            for j, lm in enumerate(hand.landmarks):
                coord_text += f"  Nokta {j}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}\n"

        user_message = f"""Bu el koordinatlarını işaret dili olarak yorumla.
Hedef dil: {'Türkçe' if request.language == 'tr' else 'İngilizce'}
Koordinatlar:
{coord_text}
Sadece yorumlanmış metni döndür."""

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

@app.post("/process-image", response_model=TranslateResponse)
async def process_image(request: ImageRequest):
    """Görüntüyü al, el landmarks tespit et, Groq ile yorumla"""
    try:
        # Base64 image to PIL Image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Convert to numpy array for MediaPipe
        np_image = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)
        
        # Detect hand landmarks
        detector = get_hand_landmarker()
        result = detector.detect(mp_image)
        
        if not result or not result.hand_landmarks:
            return TranslateResponse(
                text="El tespit edilemedi",
                confidence=0.0,
                language=request.language or "tr",
                served_by="B"
            )
        
        # Convert to HandData format
        hands = []
        for idx, landmarks in enumerate(result.hand_landmarks):
            hand_type = "Right" if result.handedness[idx][0].category_name == "Right" else "Left"
            landmark_list = [
                Landmark(x=lm.x, y=lm.y, z=lm.z) 
                for lm in landmarks
            ]
            hands.append(HandData(landmarks=landmark_list, handedness=hand_type))
        
        # Send to Groq for interpretation
        client = get_groq_client()
        coord_text = ""
        for i, hand in enumerate(hands):
            hand_type = "Sağ" if hand.handedness == "Right" else "Sol"
            coord_text += f"\nEl {i+1} ({hand_type}):\n"
            for j, lm in enumerate(hand.landmarks):
                coord_text += f"  Nokta {j}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}\n"
        
        user_message = f"""Bu el koordinatlarını işaret dili olarak yorumla.
Hedef dil: {'Türkçe' if request.language == 'tr' else 'İngilizce'}
Koordinatlar:
{coord_text}
Sadece yorumlanmış metni döndür."""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=256,
        )
        result_text = completion.choices[0].message.content.strip()
        
        return TranslateResponse(
            text=result_text,
            confidence=0.85,
            language=request.language or "tr",
            served_by="B"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Görüntü işleme hatası: {str(e)}")

@app.post("/text-to-sign", response_model=TextToSignResponse)
async def text_to_sign(request: TextToSignRequest):
    client = get_groq_client()
    try:
        prompt = f"""Bu metni işaret dili animasyon komutlarına çevir.
Metin: "{request.text}"
Dil: {'Türk İşaret Dili (TİD)' if request.language == 'tr' else 'ASL'}
Format:
İŞARETLER: İŞARET1, İŞARET2, İŞARET3
AÇIKLAMA: kısa açıklama"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Sen işaret dili uzmanısın."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512,
        )
        content = completion.choices[0].message.content.strip()
        signs = []
        description = content
        for line in content.split("\n"):
            if "İŞARETLER:" in line or "SIGNS:" in line.upper():
                signs = [s.strip() for s in line.replace("İŞARETLER:", "").replace("SIGNS:", "").split(",")]
            elif "AÇIKLAMA:" in line or "DESCRIPTION:" in line.upper():
                description = line.replace("AÇIKLAMA:", "").replace("DESCRIPTION:", "").strip()
        return TextToSignResponse(animation_sequence=signs, description=description, served_by="B")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API hatası: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)