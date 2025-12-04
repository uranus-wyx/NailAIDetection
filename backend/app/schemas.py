# app/schemas.py
from pydantic import BaseModel, HttpUrl
from typing import Optional

class PredictionResult(BaseModel):
    disease_label: str
    confidence: float
    gradcam_url: Optional[HttpUrl] = None

class PredictResponse(BaseModel):
    request_id: str
    status: str  # "success" / "processing" / "error"
    result: Optional[PredictionResult] = None
    message: Optional[str] = None
