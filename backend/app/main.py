# main.py
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uuid
import json
from datetime import datetime, timezone
from google.cloud import bigquery

from backend.app.utils_hierarchical import (
    init_models,
    hierarchical_predict,
    publish_inference_job,
    get_bq_client,
    BQ_DATASET,
    BQ_TABLE,
)

# ---------------- Paths ----------------
# Resolve all paths relative to this file instead of the working directory.
BASE = Path(__file__).parent.resolve()          # backend/app
FRONTEND_DIR = (BASE / "../../frontend").resolve()
PREDICT_DIR  = (BASE / "../../predict_data").resolve()

# Ensure static directories exist (StaticFiles requires them to exist).
PREDICT_DIR.mkdir(parents=True, exist_ok=True)
(FRONTEND_DIR / "static").mkdir(parents=True, exist_ok=True)

# ---------------- App ----------------
app = FastAPI(title="Virtual Nail Doctor")

# CORS: wide-open for development. Restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static routes
app.mount("/predict_data", StaticFiles(directory=str(PREDICT_DIR)), name="predict_data")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

# ---------------- Lifecycle ----------------
@app.on_event("startup")
def _startup():
    """
    Load all models into memory at process startup.
    utils_hierarchical.init_models() handles device selection and caching.
    """
    init_models()

# ---------------- Health ----------------
@app.get("/healthz")
async def healthz():
    return {"ok": True}

# ---------------- Frontend ----------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve the SPA entrypoint. Uses absolute path to avoid CWD issues.
    """
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h1>Frontend not found</h1><p>Expected at ./frontend/index.html</p>",
            status_code=404,
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

# ---------------- Inference ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded image and return hierarchical prediction results in JSON:
    (coarse route, final label, confidence, top3, heatmap URL, etc.)
    """
    try:
        # Basic content-type sanity check (non-fatal — PIL will validate later)
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            pass  # allow anyway

        # Read raw bytes and feed directly to hierarchical_predict()
        image_bytes = await file.read()

        result = hierarchical_predict(
            image_bytes=image_bytes,
            # thresholds (tune as needed)
            tau_coarse=0.55,
            delta_top2=0.08,
            tau_reject=0.20,
            # second-opinion behavior
            second_opinion_eps=0.08,
            min_local_conf_for_second=0.80,
            # simple TTA
            tta=True,
        )
        return JSONResponse(result)

    except Exception as e:
        # Error payload with stable structure for frontend
        return JSONResponse(
            {"error": str(e), "predicted_class": None, "coarse_class": None, "routed_via": "exception"},
            status_code=500,
        )


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Query asynchronous inference status:
      - If no record exists for this job_id → PENDING
      - If a record exists → DONE + return the inference result
    """
    try:
        client = get_bq_client()
        table = f"{client.project}.{BQ_DATASET}.{BQ_TABLE}"

        query = f"""
        SELECT
          predicted_at,
          coarse_class,
          predicted_class,
          confidence,
          routed_via,
          heatmap_url,
          input_image_url,
          top3_json,
          fine_candidates_json
        FROM `{table}`
        WHERE job_id = @job_id
        ORDER BY predicted_at DESC
        LIMIT 1
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("job_id", "STRING", job_id)
            ]
        )

        rows = list(client.query(query, job_config=job_config))

        if not rows:
            # Worker has not written the result yet
            return JSONResponse(
                {
                    "job_id": job_id,
                    "status": "PENDING",
                    "message": "Job not finished or not found yet.",
                }
            )

        row = rows[0]

        # Parse JSON fields
        try:
            top3 = json.loads(row.top3_json or "{}")
        except Exception:
            top3 = {}
        try:
            fine_candidates = json.loads(row.fine_candidates_json or "[]")
        except Exception:
            fine_candidates = []

        return JSONResponse(
            {
                "job_id": job_id,
                "status": "DONE",
                "predicted_at": row.predicted_at.isoformat() if row.predicted_at else None,
                "coarse_class": row.coarse_class,
                "predicted_class": row.predicted_class,
                "confidence": row.confidence,
                "routed_via": row.routed_via,
                "heatmap_url": row.heatmap_url,
                "input_image_url": row.input_image_url,
                "top3_probabilities": top3,
                "fine_candidates": fine_candidates,
            }
        )

    except Exception as e:
        # Query failure should not crash; return ERROR state
        return JSONResponse(
            {
                "job_id": job_id,
                "status": "ERROR",
                "message": str(e),
            },
            status_code=500,
        )


@app.post("/submit")
async def submit_job(file: UploadFile = File(...)):
    """
    Asynchronous inference entrypoint:
      1) Receive image → upload to GCS
      2) Create job_id
      3) Publish a Pub/Sub message (metadata only, no raw bytes)
      4) Return { job_id, input_image_url } to the frontend
    """
    try:
        image_bytes = await file.read()

        # 1) Upload original image to GCS and obtain URL
        input_image_url = None
        try:
            from backend.app.utils_hierarchical import save_input_image
            input_image_url = save_input_image(image_bytes)
        except Exception as e:
            print(f"[submit] save_input_image failed: {e}")
            raise

        # 2) Create job_id (UUID)
        job_id = str(uuid.uuid4())
        requested_at = datetime.now(timezone.utc).isoformat()

        # 3) Build Pub/Sub job payload
        job_payload = {
            "job_id": job_id,
            "gcs_image_url": input_image_url,
            "requested_at": requested_at,
        }

        # 4) Publish to Pub/Sub
        publish_inference_job(job_payload)

        # 5) Return to frontend (job accepted; result will be queried later)
        return JSONResponse(
            {
                "job_id": job_id,
                "input_image_url": input_image_url,
                "status": "QUEUED",
                "message": "Job accepted; processing asynchronously.",
            }
        )

    except Exception as e:
        return JSONResponse(
            {"error": str(e), "status": "FAILED"},
            status_code=500,
        )
