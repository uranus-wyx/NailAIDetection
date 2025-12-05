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

# Copy only requirements first for better Docker layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy ONLY the needed worker code
COPY . ./worker
COPY backend/app/utils_hierarchical.py ./backend/app/
COPY backend/models ./backend/models

# Let Python treat /app as the root package
ENV PYTHONPATH=/app

# Cloud Run provides the PORT environment variable; use it to start uvicorn
# Note: we are serving the FastAPI app from worker.worker_main:app
CMD ["sh", "-c", "uvicorn worker.worker_main:app --host 0.0.0.0 --port ${PORT:-8080}"]
