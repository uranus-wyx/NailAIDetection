# backend/Dockerfile

FROM python:3.11-slim

# Working directory inside the container
WORKDIR /app

# ==== System dependencies (for OpenCV, PyTorch, etc.) ====
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ==== Python dependencies ====
# Build context is backend/, so requirements.txt can be copied directly
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==== App code ====
# Copy the entire backend project into /app/backend
# This preserves the original import path: backend.app.main
COPY . ./backend

# Create predict_data directory (for saving local Grad-CAM outputs if needed)
RUN mkdir -p /app/predict_data

# ==== Environment ====
ENV PYTHONPATH="/app"

# ==== Entrypoint ====
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
