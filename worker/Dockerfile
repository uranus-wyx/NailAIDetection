# Use slim Python image
FROM python:3.11-slim

# Install system dependencies (avoid cv2 unless needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend + worker code into container
COPY backend ./backend
COPY worker ./worker

# Let Python find /app as root package
ENV PYTHONPATH=/app

# Cloud Run 會設定環境變數 PORT，我們用它來啟動 uvicorn
# 注意這裡用的是 worker.worker_main:app
CMD ["sh", "-c", "uvicorn worker.worker_main:app --host 0.0.0.0 --port ${PORT:-8080}"]