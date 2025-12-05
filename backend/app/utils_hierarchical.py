# utils_hierarchical.py
import io, os, time, json, re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import pubsub_v1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ---- Grad-CAM ----
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ----------------- Device -----------------
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ----------------- Paths -----------------
# __file__ = backend/app/utils_hierarchical.py
APP_DIR = Path(__file__).parent.resolve()          # backend/app
BACKEND_DIR = (APP_DIR / "..").resolve()           # backend
PROJECT_ROOT = (APP_DIR / "../..").resolve()       # root

MODELS_DIR = (BACKEND_DIR / "models").resolve()    # coarse / binary / fine_* checkpoints
ARTIFACTS = (BACKEND_DIR / "artifacts").resolve()  # coarse_labels.json / true_to_pred_idx.json / svp_order.json
PREDICT_DIR = (PROJECT_ROOT / "predict_data").resolve()

# ----------------- Defaults -----------------
_DEFAULT_COARSE = [
    "Color abnormalities", "Cyanosis signs", "Darier's disease", "Eczema-related",
    "Healthy", "Line abnormalities", "Melanoma", "Psoriasis",
    "Shape deformities", "Whitening disorders"
]
PS_VS_DEF_CLASSES = ["Psoriasis", "Shape deformities"]

# ----------------- Global state -----------------
__LOADED = False

coarse_class_names: List[str] = []
true_to_pred_idx: Optional[List[int]] = None
svp_order: Tuple[str, str] = ("Psoriasis", "Shape deformities")

coarse_model: Optional[nn.Module] = None
ps_vs_def_model: Optional[nn.Module] = None
fine_models_dict: Dict[str, nn.Module] = {}
fine_label_names_dict: Dict[str, List[str]] = {}

# Fine head packages (one per coarse class with >1 fine labels)
FINE_PACKAGES: Dict[str, Path] = {
    "Whitening disorders": MODELS_DIR / "fine_whitening.pt",
    "Cyanosis signs":      MODELS_DIR / "fine_cyanosis.pt",
    "Color abnormalities": MODELS_DIR / "fine_color.pt",
    "Line abnormalities":  MODELS_DIR / "fine_lines.pt",
    "Shape deformities":   MODELS_DIR / "fine_shape.pt",
    "Eczema-related":      MODELS_DIR / "fine_eczema.pt",
    # Single-fine classes do not need a head: Healthy / Psoriasis / Darier's disease / Melanoma
}

# ----------------- Google Cloud Storage -----------------
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME") 
_gcs_client = None

def get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client

# ----------------- Fine package loader -----------------
def _load_resnet18_package(pkg_path: Path, device: torch.device):
    """
    Supported formats:
      1) {'state_dict': ..., 'class_names': [...]}  (or sidecar .labels.json)
      2) torch.save(model)  (class names read from sidecar if available)
    """
    obj = torch.load(str(pkg_path), map_location=device)
    class_names = None

    if isinstance(obj, dict) and "state_dict" in obj:
        class_names = obj.get("class_names")
        labels_sidecar = pkg_path.with_suffix(".labels.json")
        if class_names is None and labels_sidecar.exists():
            class_names = json.loads(labels_sidecar.read_text(encoding="utf-8"))
        if class_names is None:
            raise RuntimeError(f"{pkg_path} missing class_names; provide in package or {labels_sidecar.name}.")
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(class_names))
        m.load_state_dict(obj["state_dict"])
        m.to(device).eval()
        return m, class_names

    if hasattr(obj, "eval"):
        m = obj.to(device).eval()
        labels_sidecar = pkg_path.with_suffix(".labels.json")
        if labels_sidecar.exists():
            class_names = json.loads(labels_sidecar.read_text(encoding="utf-8"))
        return m, class_names

    raise RuntimeError(f"Unsupported package format: {pkg_path}")

# ----------------- Init (call once on startup) -----------------
def init_models():
    """
    Loads:
      - artifacts/coarse_labels.json (falls back to default order)
      - artifacts/true_to_pred_idx.json (optional)
      - artifacts/svp_order.json (optional)
      - models/resnet18.pth (coarse)
      - models/resnet18_ps_vs_def.pth (binary Psoriasis vs Shape)
      - models/fine_*.pt (fine heads)
    """
    global __LOADED
    global coarse_class_names, true_to_pred_idx, svp_order
    global coarse_model, ps_vs_def_model, fine_models_dict, fine_label_names_dict

    if __LOADED:
        return

    # 1) Labels and mappings
    cl_path = ARTIFACTS / "coarse_labels.json"
    if cl_path.exists():
        coarse_class_names = json.loads(cl_path.read_text(encoding="utf-8"))
    else:
        # Use default but ensure it matches training order
        coarse_class_names = _DEFAULT_COARSE[:]

    t2p_path = ARTIFACTS / "true_to_pred_idx.json"
    true_to_pred_idx = None
    if t2p_path.exists():
        true_to_pred_idx = json.loads(t2p_path.read_text(encoding="utf-8"))

    svp_path = ARTIFACTS / "svp_order.json"
    if svp_path.exists():
        s = json.loads(svp_path.read_text(encoding="utf-8"))
        svp_order = (s[0], s[1])
    else:
        svp_order = ("Psoriasis", "Shape deformities")

    # 2) Coarse model
    cm = models.resnet18(weights=None)
    cm.fc = nn.Linear(cm.fc.in_features, len(coarse_class_names))
    cm.load_state_dict(torch.load(str(MODELS_DIR / "resnet18.pth"), map_location=DEVICE))
    cm.to(DEVICE).eval()
    coarse_model = cm

    # 3) Binary model (Psoriasis vs Shape deformities)
    bm = models.resnet18(weights=None)
    bm.fc = nn.Linear(bm.fc.in_features, 2)
    bm.load_state_dict(torch.load(str(MODELS_DIR / "resnet18_ps_vs_def.pth"), map_location=DEVICE))
    bm.to(DEVICE).eval()
    ps_vs_def_model = bm

    # 4) Fine heads
    fine_models_dict = {}
    fine_label_names_dict = {}
    for cname, path in FINE_PACKAGES.items():
        if not path.exists():
            continue
        fm, fnames = _load_resnet18_package(path, DEVICE)
        if fnames is None:
            raise RuntimeError(f"{path} lacks fine class names (add .labels.json or embed in package).")
        fine_models_dict[cname] = fm
        fine_label_names_dict[cname] = fnames

    # Add singleton fine classes so routing can still return a label
    for single in ["Healthy","Psoriasis","Darier's disease","Melanoma"]:
        if single not in fine_label_names_dict:
            fine_label_names_dict[single] = [single]

    __LOADED = True
    print(f"[init_models] device={DEVICE} | coarse={len(coarse_class_names)} "
          f"| fine_heads={len(fine_models_dict)} | svp_order={svp_order}")

# ----------------- CAM / Transform -----------------
def _pick_target_layers_for_resnet18(model: nn.Module):
    """Pick a suitable last conv layer for Grad-CAM (supports custom heads)."""
    if hasattr(model, "layer4"):
        last = model.layer4[-1]
        return [last] if isinstance(last, nn.Module) else [model.layer4]
    # Fallback: last Conv2d in the module tree
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No suitable convolutional layer found for Grad-CAM target.")
    return [last_conv]

def generate_heatmap(image_tensor, model, target_class_idx):
    """Generate a Grad-CAM heatmap PIL image for the given model and class index."""
    model.eval()
    # Ensure tensor is on the same device as the model
    image_tensor = image_tensor.to(next(model.parameters()).device)

    target_layers = _pick_target_layers_for_resnet18(model)

    # Grad-CAM requires gradients enabled
    with torch.enable_grad():
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=[ClassifierOutputTarget(int(target_class_idx))]
        )[0]

    # Recover to [0,1] image and overlay CAM
    img_np = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
    cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return Image.fromarray(cam_img)

def transform_image(image_bytes):
    """Load bytes → RGB PIL → 224×224 tensor → normalize; returns 1×3×224×224 tensor."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

# ----------------- Predict helpers -----------------
@torch.no_grad()
def _softmax_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over a 1D logit vector (dim=0)."""
    return F.softmax(logits, dim=0)

@torch.no_grad()
def _forward_logits(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass a BCHW tensor through model on DEVICE, return logits [C]."""
    return m(x.to(DEVICE))[0]  # [C]

def top3_probs(prob_array, class_names):
    """Return a dict of top-3 probabilities > 0 as {class_name: prob}."""
    filtered = [(i, p) for i, p in enumerate(prob_array) if p > 0]
    top = sorted(filtered, key=lambda x: x[1], reverse=True)[:3]
    return {class_names[i]: float(p) for i, p in top}

# Original 22→10 fine-to-coarse mapping (kept for fine candidate list in responses)
label_map_22_to_10 = {
    "healthy": "Healthy",
    "white nail": "Whitening disorders",
    "leukonychia": "Whitening disorders",
    "pale nail": "Whitening disorders",
    "terry_s nail": "Whitening disorders",
    "bluish nail": "Cyanosis signs",
    "blue_finger": "Cyanosis signs",
    "red lunula": "Color abnormalities",
    "yellow nails": "Color abnormalities",
    "beau_s lines": "Line abnormalities",
    "Muehrck-e_s lines": "Line abnormalities",
    "splinter hemmorrage": "Line abnormalities",
    "half and half nailes (Lindsay_s nails)": "Line abnormalities",
    "koilonychia": "Shape deformities",
    "clubbing": "Shape deformities",
    "onycholycis": "Shape deformities",
    "pitting": "Shape deformities",
    "psoriasis": "Psoriasis",
    "eczema": "Eczema-related",
    "aloperia areata": "Eczema-related",
    "Darier_s disease": "Darier’s disease",
    "Acral_Lentiginous_Melanoma": "Melanoma"
}
coarse_to_fine_map = defaultdict(list)
for fine, coarse in label_map_22_to_10.items():
    coarse_to_fine_map[coarse].append(fine)

def save_heatmap(pil_image: Image.Image):
    """
    If GCS_BUCKET_NAME is set → upload to Cloud Storage and return public URL.
    Otherwise, fallback to saving in local predict_data/ and return a relative path (for local debugging).
    """
    timestamp = int(time.time())
    filename = f"hint_{timestamp}.jpg"

    if GCS_BUCKET_NAME:
        # Upload to GCS
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob_path = f"heatmaps/{filename}"
        blob = bucket.blob(blob_path)

        # Save PIL image into memory before uploading
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        buf.seek(0)

        blob.upload_from_file(buf, content_type="image/jpeg")

        # For demo purposes: make file public; production can switch to signed URLs
        blob.make_public()
        return blob.public_url

    # Without GCS_BUCKET_NAME → keep original local behavior (for local development)
    PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PREDICT_DIR / filename
    pil_image.save(save_path)
    return f"predict_data/{filename}"

def save_input_image(image_bytes: bytes):
    """
    Save the original input image to GCS (if GCS_BUCKET_NAME is set),
    return its image URL; if GCS is not configured, save locally under predict_data/inputs.
    """

    timestamp = int(time.time())
    filename = f"input_{timestamp}.jpg"

    if GCS_BUCKET_NAME:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob_path = f"inputs/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_file(io.BytesIO(image_bytes), content_type="image/jpeg")
        blob.make_public()
        return blob.public_url

    # Without GCS, save locally for debugging
    save_dir = os.path.join(os.path.dirname(__file__), "../predict_data/inputs")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    # When frontend runs under same origin, it uses /predict_data/...
    return f"predict_data/inputs/{filename}"

# ------------- BigQuery logging -------------
BQ_DATASET = os.getenv("BQ_DATASET", "nailai_analytics")
BQ_TABLE   = os.getenv("BQ_TABLE",   "predictions")
_bq_client = None

def get_bq_client():
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client()
    return _bq_client

def log_prediction_to_bigquery(payload: dict, job_id: str | None = None):
    """
    Best-effort logging: failures are printed but do not affect the API response.
    payload = the dict returned by hierarchical_predict() to the frontend.
    job_id   = UUID for asynchronous processing; synchronous /predict can omit this.
    """
    try:
        client = get_bq_client()
        table_id = f"{client.project}.{BQ_DATASET}.{BQ_TABLE}"

        row = {
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "coarse_class": payload.get("coarse_class"),
            "predicted_class": payload.get("predicted_class"),
            "confidence": float(payload.get("confidence") or 0.0),
            "routed_via": payload.get("routed_via"),
            "heatmap_url": payload.get("heatmap_url"),
            "top3_json": json.dumps(payload.get("top3_probabilities") or {}),
            "fine_candidates_json": json.dumps(payload.get("fine_candidates") or []),
            "input_image_url": payload.get("input_image_url"),
            "job_id": job_id or payload.get("job_id"),
        }

        errors = client.insert_rows_json(table_id, [row])
        if errors:
            print(f"[BigQuery] insert error: {errors}")
    except Exception as e:
        print(f"[BigQuery] logging failed: {e}")

# ------------- Pub/Sub -------------
PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC", "projects/nailai-demo/topics/nailai-infer")
_pub_publisher = None

def get_pub_publisher():
    global _pub_publisher
    if _pub_publisher is None:
        _pub_publisher = pubsub_v1.PublisherClient()
    return _pub_publisher

def publish_inference_job(job_payload: dict):
    """
    Send the inference job into Pub/Sub.
    It is recommended that job_payload includes: job_id, gcs_image_url, requested_at, etc.
    """
    try:
        publisher = get_pub_publisher()
        topic_path = PUBSUB_TOPIC
        data = json.dumps(job_payload).encode("utf-8")
        future = publisher.publish(topic_path, data=data)
        future.add_done_callback(
            lambda f: print(f"[PubSub] published job_id={job_payload.get('job_id')} err={f.exception()}"))
    except Exception as e:
        print(f"[PubSub] publish failed: {e}")
        raise

# ----------------- Hierarchical Predict -----------------
def hierarchical_predict(
    image_bytes: bytes,
    # —— thresholds —— #
    tau_coarse: float = 0.55,
    delta_top2: float = 0.08,
    tau_reject: float = 0.20,
    # —— TTA —— #
    tta: bool = True,
    # —— Second-Opinion (ambiguous pairs; default Color↔Line) —— #
    second_opinion_pairs: Set[Tuple[str, str]] = {("Color abnormalities", "Line abnormalities")},
    second_opinion_eps: float = 0.08,
    min_local_conf_for_second: float = 0.80,
    # —— job id for async logging —— #
    job_id: str | None = None,
    ):
    """
    image bytes → dict (directly FastAPI-serializable)

    Steps:
      1) Coarse prediction (optionally realign logits to true order via true_to_pred_idx).
      2) If top-2 includes both Psoriasis & Shape AND low confidence/close margin → binary arbitration.
      3) If the routed coarse has a fine head (>1 class) → run fine; else return singleton label.
      4) Generate Grad-CAM from the final decision model & class.

    Returns:
      {
        "predicted_class": str | None,
        "coarse_class": str | None,
        "confidence": float,                       # final probability
        "top3_probabilities": {cls: prob, ...},    # from final head/model
        "fine_candidates": list[str],              # 22→10 lookup
        "heatmap_url": str | None,                 # relative path for frontend
        "routed_via": str                          # e.g., "coarse-top1+fine", "svp-binary", "second-opinion(...)"
      }
    """
    global __LOADED
    if not __LOADED:
        init_models()

            # Store raw input images（best-effort）

    input_image_url = None
    try:
        input_image_url = save_input_image(image_bytes)
    except Exception as e:
        print(f"❌ Input image save failed: {e}")

    # Preprocess input once; produce TTA variants if enabled
    x = transform_image(image_bytes)  # [1,3,224,224]
    xs = [x]
    if tta:
        xs.append(torch.flip(x, dims=[3]))  # horizontal flip TTA

    def avg_logits(model: nn.Module) -> torch.Tensor:
        """Average logits across TTA variants. Returns [C]."""
        with torch.no_grad():
            outs = [model(z.to(DEVICE)) for z in xs]   # each [1,C]
        return torch.stack([o.squeeze(0) for o in outs], dim=0).mean(dim=0)

    # === (1) coarse ===
    logits_c = avg_logits(coarse_model)            # [C(pred_order)]
    if true_to_pred_idx is not None:
        # Reorder model outputs to match coarse_class_names
        idx = torch.tensor(true_to_pred_idx, device=logits_c.device, dtype=torch.long)
        logits_c = logits_c.index_select(0, idx)

    probs_c_t = F.softmax(logits_c, dim=0)         # [C]
    probs_c = probs_c_t.cpu().numpy()
    topk = torch.topk(probs_c_t, k=min(3, len(probs_c_t)))
    p1, i1 = float(topk.values[0]), int(topk.indices[0])
    p2, i2 = (float(topk.values[1]), int(topk.indices[1])) if topk.values.numel() > 1 else (0.0, i1)
    margin = p1 - p2
    c1, c2 = coarse_class_names[i1], coarse_class_names[i2]

    # Reject if the best coarse probability is too low
    if p1 < tau_reject:
        return {
            "predicted_class": None,
            "coarse_class": None,
            "confidence": float(p1),
            "top3_probabilities": top3_probs(probs_c, coarse_class_names),
            "fine_candidates": [],
            "heatmap_url": None,
            "routed_via": "reject"
        }

    routed_coarse = c1
    routed_via = "coarse-top1"

    # === (2) Psoriasis ↔ Shape arbitration (only when ambiguous) ===
    if ({"Psoriasis", "Shape deformities"} <= set([c1, c2])) and (p1 < tau_coarse or margin < delta_top2):
        logits_b = avg_logits(ps_vs_def_model)     # [2]
        probs_b = F.softmax(logits_b, dim=0).cpu().numpy()
        ps_first = (svp_order[0] == "Psoriasis")
        ps_idx, sh_idx = (0, 1) if ps_first else (1, 0)
        routed_coarse = "Psoriasis" if probs_b[ps_idx] >= probs_b[sh_idx] else "Shape deformities"
        routed_via = "svp-binary"

    # === (3) fine head (if exists and has >1 classes) ===
    used_model = None
    used_class_names = None
    final_label = None
    final_probs = None
    final_idx = None

    def run_fine_for(coarse_name: str):
        """Run the fine head for this coarse group; return (names, probs, argmax_idx)."""
        m = fine_models_dict[coarse_name]
        names = fine_label_names_dict[coarse_name]
        logits_f = avg_logits(m)
        probs_f_t = F.softmax(logits_f, dim=0)
        probs_f = probs_f_t.cpu().numpy()
        f_idx = int(probs_f_t.argmax())
        return names, probs_f, f_idx

    has_fine = (routed_coarse in fine_models_dict) and (len(fine_label_names_dict.get(routed_coarse, [])) > 1)
    if has_fine:
        names_main, probs_main, i_main = run_fine_for(routed_coarse)
        used_model = fine_models_dict[routed_coarse]
        used_class_names = names_main
        final_label, final_probs, final_idx = names_main[i_main], probs_main, i_main
        routed_via = routed_via + "+fine"

        # === (4) Second-opinion (e.g., Color ↔ Line) ===
        # Conditions: (coarse top-2 are close) AND (main fine local confidence is low)
        # Score fusion: w = (max fine prob) × (coarse prob)
        for competitor in [c2]:  # consider 2nd best coarse as competitor
            if competitor is None:
                continue
            pair = (routed_coarse, competitor)
            if not (pair in second_opinion_pairs or pair[::-1] in second_opinion_pairs):
                continue
            p_routed = float(probs_c_t[coarse_class_names.index(routed_coarse)])
            p_comp = float(probs_c_t[coarse_class_names.index(competitor)])
            if abs(p_routed - p_comp) > float(second_opinion_eps):
                continue
            if float(max(final_probs)) >= float(min_local_conf_for_second):
                continue  # already confident; do not flip

            if (competitor in fine_models_dict) and (len(fine_label_names_dict.get(competitor, [])) > 1):
                names_comp, probs_comp, i_comp = run_fine_for(competitor)
                w_main = float(max(final_probs)) * p_routed
                w_comp = float(max(probs_comp)) * p_comp
                if w_comp > w_main:
                    routed_via = f"second-opinion({competitor})"
                    routed_coarse = competitor
                    used_model = fine_models_dict[competitor]
                    used_class_names = names_comp
                    final_label, final_probs, final_idx = names_comp[i_comp], probs_comp, i_comp
                    break
    else:
        # Singleton fine case: Healthy / Psoriasis / Darier's / Melanoma
        final_label = routed_coarse
        final_probs = probs_c
        final_idx = coarse_class_names.index(routed_coarse)
        used_model = coarse_model
        used_class_names = coarse_class_names

    # === (5) Grad-CAM for the final decision ===
    try:
        with torch.enable_grad():
            heatmap_image = generate_heatmap(x, used_model, final_idx)
        heatmap_url = save_heatmap(heatmap_image)
    except Exception as e:
        print(f"❌ Heatmap generation failed: {e}")
        heatmap_url = None

    result = {
        "predicted_class": final_label,
        "coarse_class": routed_coarse,
        "confidence": float(final_probs[final_idx]),
        "top3_probabilities": top3_probs(final_probs, used_class_names),
        "fine_candidates": coarse_to_fine_map.get(routed_coarse, []),
        "heatmap_url": heatmap_url,
        "input_image_url": input_image_url,
        "routed_via": routed_via,
        "job_id": job_id,
    }

    # BigQuery log（best-effort）
    if os.getenv("ENABLE_BQ_LOGGING", "true").lower() in {"1", "true", "yes"}:
        log_prediction_to_bigquery(result)

    return result