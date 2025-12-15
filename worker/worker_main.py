# worker/worker_main.py

import os
import json
import base64
import logging
from datetime import datetime
from urllib.request import urlopen
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from google.cloud import storage, bigquery

# Import the same hierarchical inference used by the backend
from backend.app.utils_hierarchical import init_models, hierarchical_predict

# -----------------------------------------------------------------------------
# Global configuration (read from environment when possible)
# -----------------------------------------------------------------------------
# Cloud Run 預設有 GOOGLE_CLOUD_PROJECT，比你現在抓的變數更常見
PROJECT_ID = (
    os.getenv("GCP_PROJECT")
    or os.getenv("PROJECT_ID")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
)

BQ_DATASET = os.getenv("BQ_DATASET", "nailai_analytics")
BQ_TABLE = os.getenv("BQ_TABLE", "predictions")
HEATMAP_BUCKET = os.getenv("HEATMAP_BUCKET")  # optional, e.g. "nailai-demo-bucket"

# Initialize GCP clients once per container (reused across requests)
bq_client: Optional[bigquery.Client] = None
storage_client: Optional[storage.Client] = None

# Resolve backend root to help locate local heatmap files if needed
# worker_main.py 路徑大概是 /app/worker/worker_main.py
# parents[1] = /app  → /app/backend 才是 backend 根目錄
BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="NailAI Worker")
logger = logging.getLogger("worker")
logging.basicConfig(level=logging.INFO)


@app.get("/health")
def health():
    return {"status": "worker ok", "project_id": PROJECT_ID}


@app.on_event("startup")
def startup_event():
    """
    Called once when the worker container starts.

    - Loads all ML models into memory (coarse + fine heads).
    - Creates global BigQuery and Storage clients.
    """
    global bq_client, storage_client, PROJECT_ID

    logger.info("[worker] startup_event triggered")

    # Initialize models
    try:
        init_models()
        logger.info("[worker] Models initialized")
    except Exception as e:
        logger.exception(f"[worker] Failed to initialize models: {e}")

    # Initialize BigQuery client (will use service account attached to Cloud Run)
    try:
        bq_client = bigquery.Client(project=PROJECT_ID)
        logger.info(f"[worker] BigQuery client ready (project={bq_client.project})")
    except Exception as e:
        logger.exception(f"[worker] Failed to initialize BigQuery client: {e}")
        bq_client = None

    # Initialize Cloud Storage client (for optional heatmap uploads)
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        logger.info("[worker] Storage client ready")
    except Exception as e:
        logger.exception(f"[worker] Failed to initialize Storage client: {e}")
        storage_client = None


# -----------------------------------------------------------------------------
# Helper: download image bytes from GCS URL (or any HTTP URL)
# -----------------------------------------------------------------------------
def download_image_bytes(image_url: str) -> bytes:
    """
    Download the image bytes from a public HTTP(S) URL or GCS signed URL.
    """
    logger.info(f"[worker] Downloading image from {image_url}")
    with urlopen(image_url) as resp:
        return resp.read()


# -----------------------------------------------------------------------------
# Helper: upload local heatmap file to GCS (optional)
# -----------------------------------------------------------------------------
def maybe_upload_heatmap_to_gcs(local_heatmap_path: Path, job_id: str) -> Optional[str]:
    """
    If HEATMAP_BUCKET is set and the local file exists, upload the heatmap to GCS
    and return its public HTTPS URL. If anything fails, return None and log it.
    """
    if HEATMAP_BUCKET is None:
        logger.info("[worker] No HEATMAP_BUCKET set; skip heatmap upload.")
        return None

    if storage_client is None:
        logger.warning("[worker] Storage client is not initialized; cannot upload heatmap.")
        return None

    if not local_heatmap_path.exists():
        logger.warning(f"[worker] Heatmap file not found at {local_heatmap_path}")
        return None

    try:
        bucket = storage_client.bucket(HEATMAP_BUCKET)
        dest_blob_name = f"heatmaps/{job_id}_{local_heatmap_path.name}"
        blob = bucket.blob(dest_blob_name)
        blob.upload_from_filename(str(local_heatmap_path))

        public_url = f"https://storage.googleapis.com/{HEATMAP_BUCKET}/{dest_blob_name}"
        logger.info(f"[worker] Uploaded heatmap to {public_url}")
        return public_url
    except Exception as e:
        logger.exception(f"[worker] Failed to upload heatmap to GCS: {e}")
        return None


# -----------------------------------------------------------------------------
# Helper: insert inference result into BigQuery
# -----------------------------------------------------------------------------
def write_result_to_bigquery(job_id: str, image_url: str, result: Dict[str, Any]):
    """
    Insert one row into BigQuery with the inference result.
    """
    if bq_client is None:
        logger.warning("[worker] BigQuery client not initialized; skipping BQ insert.")
        return

    table_ref = bq_client.dataset(BQ_DATASET).table(BQ_TABLE)

    row = {
        "job_id": job_id,
        "image_url": image_url,
        "predicted_class": result.get("predicted_class"),
        "coarse_class": result.get("coarse_class"),
        "confidence": float(result.get("confidence") or 0.0),
        "routed_via": result.get("routed_via"),
        "heatmap_url": result.get("heatmap_url"),
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        errors = bq_client.insert_rows_json(table_ref, [row])
        if errors:
            logger.error(f"[worker] BigQuery insert errors for job_id={job_id}: {errors}")
        else:
            logger.info(f"[worker] BigQuery insert OK for job_id={job_id}")
    except Exception as e:
        logger.exception(f"[worker] Exception during BigQuery insert for job_id={job_id}: {e}")


# -----------------------------------------------------------------------------
# Pub/Sub push endpoint
# -----------------------------------------------------------------------------
@app.post("/")
async def handle_pubsub(request: Request):
    """
    Pub/Sub push endpoint for the worker (Cloud Run service).
    """
    envelope = await request.json()
    logger.info(f"[worker] Received envelope: keys={list(envelope.keys())}")

    if not envelope or "message" not in envelope:
        raise HTTPException(status_code=400, detail="No Pub/Sub message received")

    message = envelope["message"]
    data_b64 = message.get("data")
    if not data_b64:
        raise HTTPException(status_code=400, detail="Missing `data` in Pub/Sub message")

    # Decode Pub/Sub data payload
    try:
        decoded = base64.b64decode(data_b64).decode("utf-8")
        payload = json.loads(decoded)
    except Exception as e:
        logger.exception(f"[worker] Invalid Pub/Sub data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid Pub/Sub data: {e}")

    job_id = payload.get("job_id") or message.get("messageId")
    image_url = payload.get("gcs_image_url") or payload.get("input_image_url")

    if not image_url:
        raise HTTPException(status_code=400, detail="No image URL in job payload")

    logger.info(f"[worker] Received job_id={job_id}, image_url={image_url}")

    # 1) Download image bytes
    try:
        image_bytes = download_image_bytes(image_url)
    except Exception as e:
        logger.exception(f"[worker] Failed to download image for job_id={job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download image: {e}")

    # 2) Run inference
    try:
        result = hierarchical_predict(
            image_bytes=image_bytes,
            tau_coarse=0.55,
            delta_top2=0.08,
            tau_reject=0.20,
            second_opinion_eps=0.08,
            min_local_conf_for_second=0.80,
            tta=True,
        )
    except Exception as e:
        logger.exception(f"[worker] Inference failed for job_id={job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    logger.info(
        f"[worker] job_id={job_id} done: "
        f"{result.get('predicted_class')} ({result.get('confidence')}) via {result.get('routed_via')}"
    )

    # 3) If heatmap_url is a local path, optionally upload it to GCS
    raw_heatmap_url = result.get("heatmap_url")
    if raw_heatmap_url and not str(raw_heatmap_url).startswith("http"):
        local_heatmap_path = (BACKEND_ROOT / raw_heatmap_url).resolve()
        gcs_heatmap_url = maybe_upload_heatmap_to_gcs(local_heatmap_path, job_id)
        if gcs_heatmap_url:
            result["heatmap_url"] = gcs_heatmap_url

    # 4) Write result to BigQuery
    write_result_to_bigquery(job_id, image_url, result)

    # 5) Return 200 OK so Pub/Sub marks the message as ACKed
    return JSONResponse({"status": "OK", "job_id": job_id})
