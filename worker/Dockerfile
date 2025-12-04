# Use slim Python image
FROM python:3.11-slim

# Install system dependencies (avoid cv2 unless needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY worker ./worker

# Cloud Run 使用 PORT，但 worker 不需要 HTTP，仍保留避免錯誤
ENV PYTHONPATH=/app

# Start the worker
CMD ["python", "worker_main.py"]
