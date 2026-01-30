# VSS PoC - Video Search and Summarization
# Multi-stage Dockerfile for Python application with YOLO and FFmpeg

# Build stage
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --target=/build/deps -r requirements.txt

# Runtime stage
FROM python:3.12-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/deps \
    # Gradio settings
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    # Default device for YOLO (CPU)
    YOLO_DEVICE=cpu

WORKDIR /app

# Install runtime dependencies including FFmpeg and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /build/deps /app/deps

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/
COPY run_poc.py /app/
COPY pyproject.toml /app/

# Create directories for video storage and output
RUN mkdir -p /app/videos /app/output /app/temp

# Download YOLO model during build (optional, can be done at runtime)
# This downloads the default yolov8n-seg model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-seg')" || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Expose Gradio port
EXPOSE 7860

# Default entrypoint - can be overridden
ENTRYPOINT ["python"]

# Default command - run Gradio UI
# Override with: docker run ... python run_poc.py --video /app/videos/video.mp4
CMD ["-m", "src.ui.gradio_app"]
