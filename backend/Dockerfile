# backend/Dockerfile

FROM python:3.11-slim

# Working directory inside the container
WORKDIR /app

# ==== System deps (for OpenCV / torch, etc.) ====
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# ==== Python deps ====
# 現在 build context 就是 backend/，所以這裡用 requirements.txt 就好
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==== App code ====
# 把整個 backend 專案內容複製到 /app/backend 裡
# 這樣原本的匯入路徑 backend.app.main 仍然成立
COPY . ./backend

# 建 predict_data 目錄（給 Grad-CAM 存 local 檔用，如果有）
RUN mkdir -p /app/predict_data

# ==== Env ====
ENV PORT=8080
ENV PYTHONPATH="/app"

# ==== Entrypoint ====
# 注意：這裡仍然用 backend.app.main:app
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}

