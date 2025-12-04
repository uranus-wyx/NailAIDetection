# NailAI: Cloud-Powered Fine-Grained Nail Disease Detection System 🩺

https://nailai-backend-299381123286.us-central1.run.app/

NailAI is a full-stack cloud-hosted AI diagnosis system that detects **22 fine-grained nail diseases** using a ResNet-18 model with Grad-CAM explainability.

The system demonstrates a complete **serverless AI microservice architecture**, including:

- **Cloud Run** (frontend + backend)
- **Pub/Sub** (job queue)
- **Cloud Run Worker** (async inference)
- **Cloud Storage** (image storage + Grad-CAM)
- **BigQuery** (analytics log)
- **JS Web UI** (upload + camera mode)
- **Fine-grained hierarchical classifier**

This project serves as the final project for **DTSA 5503 – Cloud & Big Data Computing**.

---

# ✨ Features

✅ Fine-grained ML classifier (22 categories)  
✅ Async AI pipeline using Pub/Sub  
✅ Cloud Run scalable backend  
✅ Background worker container for inference  
✅ Grad-CAM heatmap generation  
✅ BigQuery logging  
✅ Fully responsive web UI  
✅ Camera capture → ROI extraction → inference  
✅ Local browser history viewer  
✅ 100% serverless, auto-scaling  

---

# 🏗️ System Architecture


---

# 📁 Repository Structure

```

NailAI/
│
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI backend: /submit, /status
│   │   ├── utils.py              # Image loading, Grad-CAM helpers
│   │   ├── model_loader.py       # Load hierarchical ResNet18
│   │   ├── inference.py          # Logic shared with worker
│   └── requirements.txt
│
├── worker/
│   ├── worker.py                 # Pub/Sub consumer + inference
│   └── Dockerfile
│
├── frontend/
│   ├── index.html
│   ├── static/
│   │   ├── js/
│   │   │   ├── main.js           # async submit + polling
│   │   │   └── static_frame.js
│   │   ├── css/style.css
│   │   └── favicon.ico
│
├── model/
│   └── nail_model.pth            # Trained model file
│
└── README.md

```

---

# 🚀 Deployment Guide (Cloud Run + Cloud Build)

## **1. Enable Required GCP Services**
```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  pubsub.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com
````

---

## **2. Create Storage Bucket**

```bash
gsutil mb -l us-central1 gs://nailai-demo-bucket/
```

---

## **3. Create Pub/Sub Topic**

```bash
gcloud pubsub topics create nailai-jobs
```

---

## **4. Create BigQuery Dataset + Table**

### Dataset:

```bash
bq --location=US mk nailai_analytics
```

### Table:

```bash
bq mk \
--table \
nailai_analytics.inference_log \
schema.json
```

Example schema:

```json
[
  {"name": "job_id", "type": "STRING"},
  {"name": "predicted_class", "type": "STRING"},
  {"name": "confidence", "type": "FLOAT"},
  {"name": "image_path", "type": "STRING"},
  {"name": "heatmap_path", "type": "STRING"},
  {"name": "timestamp", "type": "TIMESTAMP"}
]
```

---

# 🐳 5. Deploy Backend (Cloud Run)

From repo root:

```bash
gcloud builds submit --tag gcr.io/<PROJECT_ID>/nailai-backend
gcloud run deploy nailai-backend \
  --image gcr.io/<PROJECT_ID>/nailai-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

# 🐳 6. Deploy Worker (Cloud Run)

```bash
gcloud builds submit worker/ --tag gcr.io/<PROJECT_ID>/nailai-worker

gcloud run deploy nailai-worker \
  --image gcr.io/<PROJECT_ID>/nailai-worker \
  --platform managed \
  --region us-central1 \
  --max-instances=5 \
  --allow-unauthenticated
```

Bind worker to the Pub/Sub trigger:

```bash
gcloud run services add-iam-policy-binding nailai-worker \
  --member=serviceAccount:PROJECT_NUM-compute@developer.gserviceaccount.com \
  --role=roles/run.invoker
```

```bash
gcloud pubsub subscriptions create nailai-sub \
  --topic nailai-jobs \
  --push-endpoint=https://nailai-worker-xxxxxx.run.app/ \
  --push-auth-service-account=PROJECT_NUM-compute@developer.gserviceaccount.com
```

---

# 🧪 Local Development

### Install dependencies:

```bash
pip install -r backend/requirements.txt
```

### Run:

```bash
cd backend
uvicorn app.main:app --reload --port 8080
```

---

# 🌐 Frontend Usage

Open:

```
https://<CLOUD_RUN_BACKEND_URL>
```

Features:

* 📤 Upload image
* 📸 Camera mode with ROI capture
* 🔄 `/submit` async inference
* 🔍 `/status/{job_id}` polling
* 🔥 Grad-CAM heatmap
* 🕘 Local history viewer (browser only)

---

# 🧠 ML Model

* ResNet-18 backbone
* Fine-grained classification: 22 nail diseases
* Softmax probability
* Grad-CAM explanation
* Hierarchical coarse → fine routing

---

# 🔍 Demo Flow

1. User uploads image or captures via camera
2. Frontend sends **POST /submit**
3. Backend:

   * Stores image
   * Publishes Pub/Sub message
4. Worker:

   * Runs inference
   * Generates heatmap
   * Writes to BigQuery
5. Frontend:

   * Polls /status
   * Displays results + heatmap

---

# ⚠️ Troubleshooting

| Issue                         | Fix                                      |
| ----------------------------- | ---------------------------------------- |
| 404 on heatmap                | Check Cloud Storage file path            |
| Pub/Sub not triggering worker | Verify subscription push URL             |
| Worker returning 500          | Check Cloud Logging                      |
| CORS issues                   | Deploy frontend & backend to same origin |
| BigQuery insert failed        | Check schema mismatch                    |

