# Nail Detection

Deployment-focused GCP repository for serving a nail image inference application.

This repository is intended to stay GitHub-safe:
- application code stays here
- infrastructure templates stay here
- private datasets, notebooks, checkpoints, local environments, and scratch outputs stay out

## Repository Layout

```text
Nail_Detection/
├── backend/
│   ├── app/
│   ├── artifacts/              # Runtime metadata; review before publishing
│   ├── models/                 # Runtime model files; review before publishing
│   └── requirements.txt
├── worker/
│   ├── worker_main.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   └── static/
├── infra/
│   ├── cloud-run/
│   │   ├── backend.yaml
│   │   └── worker.yaml
│   └── cloudbuild/
│       ├── backend.yaml
│       └── worker.yaml
├── scripts/
│   └── loadtest/
│       └── locustfile.py
├── docs/
│   └── assets/
├── Dockerfile.backend
├── Dockerfile.worker
└── .gitignore
```

## What Belongs Here

- Backend inference API code
- Worker code
- Frontend assets
- Dockerfiles
- GCP deployment templates
- Build configuration
- Load testing scripts
- Documentation assets

## What Should Stay Out

- Training datasets
- Jupyter notebooks
- Experiment outputs
- Local virtual environments
- Temporary prediction images
- Logs and caches
- Private checkpoints not required for deployed runtime

## Deployment Notes

The templates under `infra/` are sanitized examples. Replace placeholders before deployment:
- `<PROJECT_ID>`
- `<REGION>`
- `<ARTIFACT_REGISTRY_REPO>`
- `<BACKEND_IMAGE>`
- `<WORKER_IMAGE>`
- `<BUCKET_NAME>`
- `<PUBSUB_TOPIC>`
- `<BQ_DATASET>`
- `<BQ_TABLE>`
- `<SERVICE_ACCOUNT_EMAIL>`

## Runtime Asset Review

The following paths were intentionally left in place because they may be required at runtime and should be reviewed manually before publishing:
- `backend/models/`
- `backend/artifacts/`
- `frontend/static/models/`

If you want a fully code-only public repository, move those runtime assets to private storage and load or download them during build or deploy.

## Private Runtime Assets

A clean public clone does not need to store private runtime assets in git.

This repository includes minimal scaffolding for external asset loading:
- Backend and worker can download missing private backend assets from GCS at startup/runtime.
- Frontend camera runtime keeps using `/static/models/hand_landmarker.task`.
- If `HAND_LANDMARKER_MODEL_URL` is set on the backend service, the backend downloads that file at startup into the local static path.

### Backend and Worker Asset Variables

- `RUNTIME_ASSET_MODE=local|gcs`
- `RUNTIME_ASSET_BUCKET=<private-bucket>`
- `RUNTIME_ASSET_PREFIX=<optional-prefix>`
- `MODELS_DIR=<optional-local-model-dir>`
- `ARTIFACTS_DIR=<optional-local-artifact-dir>`

When `RUNTIME_ASSET_MODE=gcs`, the app expects these objects:

```text
gs://<RUNTIME_ASSET_BUCKET>/<RUNTIME_ASSET_PREFIX>/backend/models/...
gs://<RUNTIME_ASSET_BUCKET>/<RUNTIME_ASSET_PREFIX>/backend/artifacts/...
```

### Frontend Asset Variable

- `HAND_LANDMARKER_MODEL_URL=<gs://... or https://... source for hand_landmarker.task>`

Recommended choices:
- Backend models and artifacts: private GCS download at startup
- Frontend hand landmarker: backend startup download to `/static/models/hand_landmarker.task`

### Cloud Run Environment Variables

Backend service:
- `LOAD_MODELS=1`
- `RUNTIME_ASSET_MODE=gcs`
- `RUNTIME_ASSET_BUCKET=<private-runtime-asset-bucket>`
- `RUNTIME_ASSET_PREFIX=<optional-prefix>`
- `HAND_LANDMARKER_MODEL_URL=gs://<private-runtime-asset-bucket>/<optional-prefix>/frontend/static/models/hand_landmarker.task`
- `GCS_BUCKET_NAME=<app-bucket>`
- `BQ_DATASET=<dataset>`
- `BQ_TABLE=<table>`
- `PUBSUB_TOPIC=projects/<PROJECT_ID>/topics/<TOPIC_NAME>`

Worker service:
- `RUNTIME_ASSET_MODE=gcs`
- `RUNTIME_ASSET_BUCKET=<private-runtime-asset-bucket>`
- `RUNTIME_ASSET_PREFIX=<optional-prefix>`
- `BQ_DATASET=<dataset>`
- `BQ_TABLE=<table>`
- `HEATMAP_BUCKET=<app-bucket>`

Expected private bucket layout:

```text
gs://<RUNTIME_ASSET_BUCKET>/<RUNTIME_ASSET_PREFIX>/backend/models/...
gs://<RUNTIME_ASSET_BUCKET>/<RUNTIME_ASSET_PREFIX>/backend/artifacts/...
gs://<RUNTIME_ASSET_BUCKET>/<RUNTIME_ASSET_PREFIX>/frontend/static/models/hand_landmarker.task
```

This keeps the public repository code-focused while allowing private runtime assets to be provided outside git.
