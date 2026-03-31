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

# Konfigürasyon
USE_GROQ = os.environ.get("USE_GROQ", "true").lower() == "true"
USE_OLLAMA = os.environ.get("USE_OLLAMA", "true").lower() == "true"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

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

def call_ollama(prompt: str) -> str:
    import requests
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            },
            timeout=30
        )
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.text}")
        return response.json()["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama bağlantı hatası: {str(e)}")

def call_groq(prompt: str) -> str:
    client = get_groq_client()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=256,
    )
    return completion.choices[0].message.content.strip()

def translate_with_ai(coordinates_text: str) -> str:
    """Önce Ollama dene, başarısız olursa Groq'a geç"""
    
    if USE_OLLAMA:
        try:
            result = call_ollama(coordinates_text)
            if result and len(result) > 0:
                return result
        except Exception as e:
            print(f"Ollama failed: {e}")
    
    if USE_GROQ:
        try:
            result = call_groq(coordinates_text)
            if result and len(result) > 0:
                return result
        except Exception as e:
            print(f"Groq failed: {e}")
    
    return "Anlaşılamadı"

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
    return {
        "status": "SignMeta Backend GROUP-B Running",
        "group": "B",
        "version": "1.0.0",
        "ollama": USE_OLLAMA,
        "groq": USE_GROQ
    }

@app.get("/health")
def health():
    import requests
    ollama_ok = False
    if USE_OLLAMA:
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            ollama_ok = resp.status_code == 200
        except:
            pass
    
    groq_ok = False
    if USE_GROQ:
        groq_ok = bool(os.environ.get("GROQ_API_KEY"))
    
    return {
        "status": "ok",
        "group": "B",
        "ollama_available": ollama_ok,
        "groq_configured": groq_ok
    }

@app.post("/translate", response_model=TranslateResponse)
async def translate_sign(request: TranslateRequest):
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

        result = translate_with_ai(user_message)
        
        return TranslateResponse(
            text=result.strip(),
            confidence=0.85,
            language=request.language,
            served_by="B"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yorumlama hatası: {str(e)}")

@app.post("/text-to-sign", response_model=TextToSignResponse)
async def text_to_sign(request: TextToSignRequest):
    try:
        prompt = f"""Bu metni işaret dili animasyon komutlarına çevir.
Metin: "{request.text}"
Dil: {'Türk İşaret Dili (TİD)' if request.language == 'tr' else 'ASL'}
Format:
İŞARETLER: İŞARET1, İŞARET2, İŞARET3
AÇIKLAMA: kısa açıklama"""

        if USE_OLLAMA:
            try:
                content = call_ollama(prompt)
            except:
                if USE_GROQ:
                    content = call_groq(prompt)
                else:
                    content = "İŞARETLER: AÇIKLAMA: Metin dönüştürülemedi"
        elif USE_GROQ:
            content = call_groq(prompt)
        else:
            content = "İŞARETLER: AÇIKLAMA: Yapay zeka yapılandırılmamış"
        
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
        raise HTTPException(status_code=500, detail=f"Metin dönüşüm hatası: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)