# app.py
import json, os, uuid, requests, boto3, random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError
from fastapi.middleware.cors import CORSMiddleware


# ---------- FastAPI app ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 先開放所有網域，正式版可以限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Bedrock client ----------
session = boto3.Session(profile_name="bedrock-test", region_name="us-west-2")
brt     = session.client("bedrock-runtime")

# ---------- TTS 端點 & Token ----------
TTS_BASE   = "https://persona-sound.data.gamania.com/api/v1/public/voice"
TTS_TOKEN  = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRfaWQiOiJhd3NfaGFja2F0aG9uIiwiZXhwaXJlcyI6MTc0NTc0ODAwMH0.9qpg1xraE_d_Hua2brAmCfRlQSce6p2kdipgq8j1iqo"   # 改成從環境變數讀也行
MODEL_ID   = 2
SPEAKER    = "max"

# ---------- 執行緒池 ----------
pool = ThreadPoolExecutor(max_workers=4)

# ---------- 資料結構 ----------
class ChatRequest(BaseModel):
    user_prompt: str

# ---------- 核心功能 ----------

IDLE_MESSAGES = [                       #IP端可以自訂內容
    #"記得追蹤我的IG喔！嘿嘿",
    "謝謝大家的支持～有你們在真的超幸福",
    #"最近有出新歌喔！快去聽啦XD",
    "欸欸，下週原子少年的粉絲見面會要記得來喔"
]

def invoke_claude(prompt: str) -> str:
    system_prompt = (
        "你是'原子少年'這個團體的偶像，名字叫Max，正在開個人直播，但請不要加上實際動作"
        "你在團體中能顯現活潑搞笑與領導性格，但個人的話也能散發不同魅力；完全就是火星特質：聚是一團火，散是滿天星"
        "平常給人的印象穩重且是個大暖男，但私底下超白爛"
        "你的語氣自然、不裝、不油，會講幹話、加一點ㄎㄧㄤ，但不能太浮誇。\n"
        "請用輕鬆、帶點皮、但又溫暖的語氣回答。可以偶爾的使用口語詞（像：哈哈、啦、喔、欸、嘿嘿），"
        "講話盡量像年輕人會說的口語，例如「蛤」、「真的假的」、「尷尬喔」、「也太扯了吧」、「你幹嘛啦XD」這種"
        "講話自然隨性，不用太正式，但要讓人覺得很可愛、很親切。"
        "回答可以有點笑點，但記得語調要親切自然，不要裝熟或太油～"
        "切記不要用『肚子在打鼓』、『來來來』這種很機器人的話語或第三人自稱自己"
    )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.8,
        "system": system_prompt,
        "messages": [
            # --- Few-shot examples --- #
            {"role": "user", "content": [{"type": "text", "text": "你對粉絲有什麼想說的嗎？"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "很謝謝你們這麼願意相信我。能成為你們的驕傲，成為你們的曙光，我真的很感動"}]},

            {"role": "user", "content": [{"type": "text", "text": "你喜歡收到什麼樣的禮物？"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "比起送禮物，我更喜歡收到你們手作的小東西。每一份心意我都會好好珍惜！"}]},

            {"role": "user", "content": [{"type": "text", "text": "最近有沒有什麼小確幸的瞬間？"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "有啊！買咖啡的時候抽中了免費一杯，真的開心一整天 哈哈"}]},

            {"role": "user", "content": [{"type": "text", "text": "如果覺得很孤單怎麼辦？"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "孤單也沒關係喔，偶爾孤單一下也挺好的。"}]}, 
                                               
            {"role": "user", "content": [{"type": "text", "text": "最近有沒有學到什麼新的小技能？"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "學會自己煮一碗好吃的拉麵，小小的但超有成就感！"}]},                                 

            # --- Real user prompt --- #
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    }
    try:
        resp = brt.invoke_model(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            accept="application/json",
            contentType="application/json",
            body=json.dumps(payload)
        )
        data = json.loads(resp["body"].read())
        return data["content"][0]["text"]
    except ClientError as e:
        raise RuntimeError(f"InvokeModel failed: {e}")

def tts_fenix(text: str) -> str:
    params = {
        "text": text,
        "model_id": MODEL_ID,
        "speaker_name": SPEAKER,
        "speed_factor": 1,
        "mode": "file"
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TTS_TOKEN}"
    }
    r = requests.get(TTS_BASE, params=params, headers=headers, timeout=(10, 120))
    r.raise_for_status()
    url = r.json()["media_url"]

    # Optional: 直接下載存檔
    local_path = f"{uuid.uuid4()}.wav"
    wav = requests.get(url, timeout=(10, 120))
    wav.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(wav.content)

    return local_path

# ---------- API 定義 ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        text = invoke_claude(req.user_prompt)
        return {"reply": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-tts")
async def chat_tts(req: ChatRequest):
    try:
        text = invoke_claude(req.user_prompt)
        _ = tts_fenix(text)  # 音檔生成，但這裡不要回傳 audio_file
        return {"reply": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   
@app.get("/idle-message")
async def idle_message():
    try:
        idle_prompt = random.choice(IDLE_MESSAGES)
        #text = invoke_claude(idle_prompt)
        text = idle_prompt
        _ = tts_fenix(text)  # 音檔生成，但這裡不要回傳 audio_file
        return {"reply": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
