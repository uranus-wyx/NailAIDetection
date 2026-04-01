import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

from google.cloud import storage


APP_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = (APP_DIR / "..").resolve()
PROJECT_ROOT = (APP_DIR / "../..").resolve()

MODELS_DIR = Path(os.getenv("MODELS_DIR", str((BACKEND_DIR / "models").resolve()))).resolve()
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str((BACKEND_DIR / "artifacts").resolve()))).resolve()
FRONTEND_MODELS_DIR = Path(
    os.getenv("FRONTEND_MODELS_DIR", str((PROJECT_ROOT / "frontend/static/models").resolve()))
).resolve()

RUNTIME_ASSET_MODE = os.getenv("RUNTIME_ASSET_MODE", "local").strip().lower()
RUNTIME_ASSET_BUCKET = os.getenv("RUNTIME_ASSET_BUCKET", "").strip()
RUNTIME_ASSET_PREFIX = os.getenv("RUNTIME_ASSET_PREFIX", "").strip().strip("/")

_storage_client = None

BACKEND_MODEL_FILES = [
    "resnet18.pth",
    "resnet18_ps_vs_def.pth",
    "fine_color.pt",
    "fine_color.labels.json",
    "fine_cyanosis.pt",
    "fine_cyanosis.labels.json",
    "fine_eczema.pt",
    "fine_eczema.labels.json",
    "fine_lines.pt",
    "fine_lines.labels.json",
    "fine_shape.pt",
    "fine_shape.labels.json",
    "fine_whitening.pt",
    "fine_whitening.labels.json",
]

BACKEND_ARTIFACT_FILES = [
    "coarse_labels.json",
    "true_to_pred_idx.json",
    "svp_order.json",
]

FRONTEND_MODEL_FILES = [
    "hand_landmarker.task",
]

HAND_LANDMARKER_SOURCE_URL = os.getenv("HAND_LANDMARKER_MODEL_URL", "").strip()


def _get_storage_client():
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def _blob_name(*parts: str) -> str:
    trimmed = [p.strip("/") for p in parts if p and p.strip("/")]
    return "/".join(trimmed)


def _download_files(target_dir: Path, filenames: Iterable[str], gcs_subdir: str) -> None:
    if not RUNTIME_ASSET_BUCKET:
        raise RuntimeError("RUNTIME_ASSET_BUCKET is required when RUNTIME_ASSET_MODE=gcs")

    client = _get_storage_client()
    bucket = client.bucket(RUNTIME_ASSET_BUCKET)
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in filenames:
        local_path = target_dir / name
        if local_path.exists():
            continue
        blob = bucket.blob(_blob_name(RUNTIME_ASSET_PREFIX, gcs_subdir, name))
        if not blob.exists(client):
            raise FileNotFoundError(f"Missing runtime asset in GCS: gs://{RUNTIME_ASSET_BUCKET}/{blob.name}")
        blob.download_to_filename(str(local_path))


def _download_from_url(source_url: str, dest_path: Path) -> None:
    parsed = urlparse(source_url)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if parsed.scheme == "gs":
        client = _get_storage_client()
        bucket = client.bucket(parsed.netloc)
        blob_name = parsed.path.lstrip("/")
        blob = bucket.blob(blob_name)
        if not blob.exists(client):
            raise FileNotFoundError(f"Missing runtime asset in GCS: {source_url}")
        blob.download_to_filename(str(dest_path))
        return

    if parsed.scheme in {"http", "https"}:
        with urlopen(source_url) as resp, dest_path.open("wb") as out:
            out.write(resp.read())
        return

    raise RuntimeError(f"Unsupported HAND_LANDMARKER_MODEL_URL scheme: {source_url!r}")


def _ensure_frontend_hand_landmarker() -> None:
    target = FRONTEND_MODELS_DIR / "hand_landmarker.task"
    if target.exists():
        return
    if not HAND_LANDMARKER_SOURCE_URL:
        return
    _download_from_url(HAND_LANDMARKER_SOURCE_URL, target)


def ensure_runtime_assets(include_frontend: bool = False) -> None:
    """
    Make private runtime assets available for a clean clone.

    Modes:
    - local: assume assets already exist on disk
    - gcs: download missing assets from GCS at startup/runtime
    """
    if RUNTIME_ASSET_MODE not in {"local", "gcs"}:
        raise RuntimeError(f"Unsupported RUNTIME_ASSET_MODE={RUNTIME_ASSET_MODE!r}")

    if RUNTIME_ASSET_MODE == "gcs":
        _download_files(MODELS_DIR, BACKEND_MODEL_FILES, "backend/models")
        _download_files(ARTIFACTS_DIR, BACKEND_ARTIFACT_FILES, "backend/artifacts")

    if include_frontend:
        if RUNTIME_ASSET_MODE == "gcs" and not HAND_LANDMARKER_SOURCE_URL:
            _download_files(FRONTEND_MODELS_DIR, FRONTEND_MODEL_FILES, "frontend/static/models")
        _ensure_frontend_hand_landmarker()


def frontend_hand_landmarker_url() -> str:
    return "/static/models/hand_landmarker.task"
