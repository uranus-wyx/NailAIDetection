# backend/app/worker_main.py
import base64
import json
from urllib.request import urlopen

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from .utils_hierarchical import init_models, hierarchical_predict

app = FastAPI(title="NailAI Worker")

@app.on_event("startup")
def _startup():
    # 跟主服務一樣，啟動時先把模型載好
    init_models()
    print("[worker] models initialized")

def _download_image_bytes(url: str) -> bytes:
    # 這裡用 urllib，避免多裝 requests
    with urlopen(url) as resp:
        return resp.read()

@app.post("/")
async def handle_pubsub(request: Request):
    """
    Pub/Sub push endpoint:
      - 解析 Pub/Sub envelope
      - 取出 data (base64) → decode 成我們當初丟的 JSON payload
      - 下載 GCS 圖片 → 跑 hierarchical_predict()
      - hierarchical_predict 本身會寫 BigQuery / 產生 heatmap
    """
    envelope = await request.json()
    if not envelope or "message" not in envelope:
        raise HTTPException(status_code=400, detail="No Pub/Sub message received")

    msg = envelope["message"]
    data_b64 = msg.get("data")
    if not data_b64:
        raise HTTPException(status_code=400, detail="Missing `data` in Pub/Sub message")

    try:
        decoded = base64.b64decode(data_b64).decode("utf-8")
        payload = json.loads(decoded)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Pub/Sub data: {e}")

    job_id = payload.get("job_id")
    image_url = payload.get("gcs_image_url") or payload.get("input_image_url")

    if not image_url:
        raise HTTPException(status_code=400, detail="No image URL in job payload")

    print(f"[worker] Received job_id={job_id}, image_url={image_url}")

    try:
        image_bytes = _download_image_bytes(image_url)
    except Exception as e:
        # 回 500 → Pub/Sub 會 retry
        raise HTTPException(status_code=500, detail=f"Failed to download image: {e}")

    try:
        # 跑推論（可以用預設 threshold，也可以複製你 main.py 的設定）
        result = hierarchical_predict(
            image_bytes=image_bytes,
            tau_coarse=0.55,
            delta_top2=0.08,
            tau_reject=0.20,
            second_opinion_eps=0.08,
            min_local_conf_for_second=0.80,
            tta=True,
            job_id=job_id,
        )
        # hierarchical_predict 會：
        # - 存 heatmap 到 GCS
        # - 寫 BigQuery log
        print(f"[worker] job_id={job_id} done: {result['predicted_class']} "
              f"({result['confidence']:.4f}) via {result['routed_via']}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # 回 200 → Pub/Sub 視為已成功處理並 ack
    return JSONResponse({"status": "OK", "job_id": job_id})
