# Configuration and Deployment Specification

This document contains all configuration and deployment specifications for the Video Search and Summarization (VSS) PoC.

---

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Configuration File (config.yaml)](#configuration-file-configyaml)
3. [Docker Compose Configuration](#docker-compose-configuration)
4. [Environment Example (.env.example)](#environment-example-envexample)
5. [Dependencies (requirements.txt)](#dependencies-requirementstxt)
6. [Performance Considerations and API Limits](#performance-considerations-and-api-limits)
7. [Optimization Strategies](#optimization-strategies)

---

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google AI Studio API key for Gemini models | `your_api_key_here` |
| `MILVUS_HOST` | Hostname for Milvus vector database | `localhost` |
| `MILVUS_PORT` | Port for Milvus gRPC connection | `19530` |
| `NEO4J_HOST` | Hostname for Neo4j graph database | `localhost` |
| `NEO4J_BOLT_PORT` | Port for Neo4j Bolt protocol | `7687` |
| `NEO4J_USERNAME` | Neo4j authentication username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j authentication password | `your_password` |

### Optional Gemini Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_MODEL` | Model for VLM and LLM tasks | `gemini-3.0-pro` |
| `GEMINI_EMBEDDING_MODEL` | Model for embedding generation | `text-embedding-004` |
| `GEMINI_MAX_VIDEO_DURATION` | Maximum seconds per video chunk | `300` |
| `GEMINI_SAFETY_THRESHOLD` | Safety filter level | `BLOCK_ONLY_HIGH` |

### Optional YOLO Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `YOLO_MODEL` | YOLO model variant (n, s, m, l, x) | `yolov26n-seg` |
| `YOLO_CONFIDENCE` | Detection confidence threshold | `0.5` |
| `YOLO_IOU_THRESHOLD` | NMS IoU threshold | `0.45` |
| `YOLO_DEVICE` | Device for inference | `cuda:0` |

### Optional Processing Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `CHUNK_DURATION` | Video chunk duration in seconds | `60` |
| `MAX_CONCURRENT_UPLOADS` | Parallel Gemini uploads | `3` |
| `ENABLE_CV_PIPELINE` | Enable YOLO detection | `true` |
| `ENABLE_TRACKING` | Enable object tracking | `true` |

### RAG Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_TOP_K` | Number of results to retrieve | `5` |
| `RAG_BATCH_SIZE` | Batch size for summarization | `6` |
| `RETRIEVAL_MODE` | Retrieval strategy | `hybrid` |

### Complete Environment Variables Block

```bash
# Required - Gemini API
GEMINI_API_KEY=your_api_key_here

# Optional - Gemini Settings
GEMINI_MODEL=gemini-3.0-pro           # Model for VLM and LLM
GEMINI_EMBEDDING_MODEL=text-embedding-004
GEMINI_MAX_VIDEO_DURATION=300         # Max seconds per chunk (default: 300)
GEMINI_SAFETY_THRESHOLD=BLOCK_ONLY_HIGH  # Safety filter level

# Optional - YOLO Settings
YOLO_MODEL=yolov26n-seg               # n, s, m, l, x variants
YOLO_CONFIDENCE=0.5                   # Detection threshold
YOLO_IOU_THRESHOLD=0.45               # NMS IoU threshold
YOLO_DEVICE=cuda:0                    # Device for inference

# Optional - Processing
CHUNK_DURATION=60                     # Video chunk duration in seconds
MAX_CONCURRENT_UPLOADS=3              # Parallel Gemini uploads
ENABLE_CV_PIPELINE=true               # Enable YOLO detection
ENABLE_TRACKING=true                  # Enable object tracking

# Vector DB (Milvus) - Required
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Graph DB (Neo4j) - Required
NEO4J_HOST=localhost
NEO4J_BOLT_PORT=7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# RAG Configuration
RAG_TOP_K=5
RAG_BATCH_SIZE=6
RETRIEVAL_MODE=hybrid
```

---

## Configuration File (config.yaml)

The main configuration file uses YAML format with environment variable substitution support via `!ENV` tags.

```yaml
# Gemini Configuration
gemini:
  api_key: !ENV ${GEMINI_API_KEY}

  # VLM (Video Understanding)
  vlm:
    model: "gemini-3.0-pro"
    generation_config:
      temperature: 0.2
      top_p: 0.8
      top_k: 40
      max_output_tokens: 2048
    safety_settings:
      harassment: BLOCK_ONLY_HIGH
      hate_speech: BLOCK_ONLY_HIGH
      sexually_explicit: BLOCK_ONLY_HIGH
      dangerous_content: BLOCK_ONLY_HIGH
    video_config:
      max_duration_seconds: 300       # 5 minutes max per chunk
      supported_formats:
        - video/mp4
        - video/mpeg
        - video/quicktime
        - video/webm

  # LLM (Text Generation for RAG)
  llm:
    model: "gemini-3.0-pro"
    generation_config:
      temperature: 0.3
      top_p: 0.9
      max_output_tokens: 4096

  # Embeddings
  embeddings:
    model: "text-embedding-004"
    task_type: "RETRIEVAL_DOCUMENT"   # or RETRIEVAL_QUERY for queries
    dimensions: 768

# CV Pipeline Configuration
cv_pipeline:
  enabled: true

  yolo:
    model: "yolov26n-seg"             # Model variant
    confidence: 0.5
    iou_threshold: 0.45
    device: "cuda:0"
    half_precision: true              # FP16 inference
    target_classes:                   # Filter to specific classes (null = all)
      - person
      - car
      - truck
      - bicycle
      - motorcycle

  tracker:
    algorithm: "bytetrack"
    track_thresh: 0.5
    track_buffer: 30
    match_thresh: 0.8

# Video Processing
processing:
  chunk_duration: 60                  # seconds
  chunk_overlap: 2                    # seconds overlap between chunks
  max_concurrent_chunks: 4

# RAG Configuration
rag:
  enabled: true
  max_context_tokens: 100000

  # Vector DB (Milvus) - Required
  vector_db:
    enabled: true
    type: milvus
    host: !ENV ${MILVUS_HOST:localhost}
    port: !ENV ${MILVUS_PORT:19530}
    collection_name: "vss_poc_captions"

  # Graph DB (Neo4j) - Required
  graph_db:
    enabled: true
    type: neo4j
    host: !ENV ${NEO4J_HOST:localhost}
    port: !ENV ${NEO4J_BOLT_PORT:7687}
    username: !ENV ${NEO4J_USERNAME:neo4j}
    password: !ENV ${NEO4J_PASSWORD}

  # Hybrid Retrieval Configuration
  retrieval:
    mode: hybrid  # vector_only, graph_only, hybrid
    vector_weight: 0.6
    graph_weight: 0.4
    top_k: 5
    rerank: true
    temporal_boost: 1.2

  # Function Configuration
  functions:
    summarization:
      batch_size: 6
    ingestion:
      extract_entities: true
      extract_events: true
      link_entities: true
    retrieval:
      chat_history: true

  prompts:
    caption: |
      Analyze this video segment and provide detailed captions of all events.
      For each distinct event or action, provide:
      - Start and end timestamps in format <HH:MM:SS>
      - Detailed description of what is happening
      - Any relevant objects, people, or text visible
      Focus on actions, interactions, and notable occurrences.

    summarization: |
      Given the following timestamped captions from a video, create a structured summary.
      Group related events together and identify:
      - Key activities and their timeframes
      - Notable incidents or anomalies
      - Patterns or recurring events
      Format as bullet points with time ranges.

    chat_system: |
      You are a video analysis assistant. You have access to detailed captions,
      object detection data, and a knowledge graph of entities and events from a video.
      Answer questions accurately based on the provided context.
      If information is not available in the context, say so clearly.
      Always reference specific timestamps when relevant.

    entity_extraction: |
      Extract all entities from the video caption. Include persons, vehicles,
      objects, and locations. Provide name, type, and attributes for each.

    event_extraction: |
      Extract all events and actions from the video caption. Include description,
      type, severity, participants, and timestamps for each event.

# Logging
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Tools and Functions Configuration

The following extended configuration shows the tools and functions setup for the CA-RAG system:

```yaml
# config/config.yaml

tools:
  # Graph Database
  graph_db:
    type: neo4j
    params:
      host: !ENV ${NEO4J_HOST:localhost}
      port: !ENV ${NEO4J_BOLT_PORT:7687}
      username: !ENV ${NEO4J_USERNAME:neo4j}
      password: !ENV ${NEO4J_PASSWORD}
    tools:
      embedding: gemini_embedding

  # Vector Database
  vector_db:
    type: milvus
    params:
      host: !ENV ${MILVUS_HOST:localhost}
      port: !ENV ${MILVUS_PORT:19530}
      collection_name: "vss_poc_captions"
    tools:
      embedding: gemini_embedding

  # Gemini VLM
  gemini_vlm:
    type: vlm
    params:
      model: "gemini-2.0-flash"
      api_key: !ENV ${GEMINI_API_KEY}

  # Gemini LLM
  gemini_llm:
    type: llm
    params:
      model: "gemini-2.0-flash"
      api_key: !ENV ${GEMINI_API_KEY}
      max_tokens: 4096
      temperature: 0.3

  # Gemini Embeddings
  gemini_embedding:
    type: embedding
    params:
      model: "text-embedding-004"
      api_key: !ENV ${GEMINI_API_KEY}

functions:
  # Batch Summarization
  summarization:
    type: batch_summarization
    params:
      batch_size: 6
      prompts:
        caption_summarization: |
          Summarize the following timestamped video captions.
          Group related events and identify key activities.
          Format: start_time:end_time: description
        summary_aggregation: |
          Aggregate these summaries into a final report.
          Cluster by category: Safety Issues, Operations, Notable Events
    tools:
      llm: gemini_llm
      db: vector_db

  # Graph Ingestion
  ingestion_function:
    type: graph_ingestion
    params:
      batch_size: 1
      extract_entities: true
      extract_events: true
      link_entities: true
    tools:
      llm: gemini_llm
      db: graph_db

  # Graph Retrieval
  retriever_function:
    type: graph_retrieval
    params:
      top_k: 5
      retrieval_mode: hybrid
      chat_history: true
    tools:
      llm: gemini_llm
      db: graph_db

context_manager:
  functions:
    - summarization
    - ingestion_function
    - retriever_function
```

---

## Docker Compose Configuration

### Complete Docker Compose File

```yaml
# docker-compose.yaml

version: '3.8'

services:
  # Milvus Vector Database
  milvus-standalone:
    image: milvusdb/milvus:v2.5.4
    container_name: vss-milvus
    environment:
      ETCD_USE_EMBED: "true"
      ETCD_DATA_DIR: /var/lib/milvus/etcd
      COMMON_STORAGETYPE: local
    ports:
      - "19530:19530"  # gRPC
      - "9091:9091"    # HTTP
    volumes:
      - milvus_data:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Neo4j Graph Database
  neo4j:
    image: neo4j:5.26.4
    container_name: vss-neo4j
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:-password}
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: apoc.*
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 30s
      timeout: 10s
      retries: 5

  # VSS PoC Application
  vss-poc:
    build:
      context: ./poc
      dockerfile: Dockerfile
    container_name: vss-poc
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      MILVUS_HOST: milvus-standalone
      MILVUS_PORT: 19530
      NEO4J_HOST: neo4j
      NEO4J_BOLT_PORT: 7687
      NEO4J_USERNAME: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD:-password}
    ports:
      - "7860:7860"  # Gradio UI
    volumes:
      - ./videos:/app/videos
      - ./output:/app/output
    depends_on:
      milvus-standalone:
        condition: service_healthy
      neo4j:
        condition: service_healthy

volumes:
  milvus_data:
  neo4j_data:
  neo4j_logs:
```

### Service Details

#### Milvus Vector Database

| Property | Value |
|----------|-------|
| Image | `milvusdb/milvus:v2.5.4` |
| Container Name | `vss-milvus` |
| gRPC Port | `19530` |
| HTTP Port | `9091` |
| Storage Type | Local (embedded etcd) |
| Health Check | HTTP healthz endpoint |

#### Neo4j Graph Database

| Property | Value |
|----------|-------|
| Image | `neo4j:5.26.4` |
| Container Name | `vss-neo4j` |
| HTTP Port | `7474` |
| Bolt Port | `7687` |
| Plugins | APOC |
| Health Check | HTTP endpoint |

#### VSS PoC Application

| Property | Value |
|----------|-------|
| Container Name | `vss-poc` |
| Gradio UI Port | `7860` |
| Video Mount | `./videos:/app/videos` |
| Output Mount | `./output:/app/output` |
| Dependencies | Milvus (healthy), Neo4j (healthy) |

---

## Environment Example (.env.example)

```bash
# .env.example

# Required - Gemini API
GEMINI_API_KEY=your_gemini_api_key

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Neo4j Configuration
NEO4J_HOST=localhost
NEO4J_BOLT_PORT=7687
NEO4J_HTTP_PORT=7474
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Processing Configuration
CHUNK_DURATION=60
MAX_CONCURRENT_UPLOADS=3
ENABLE_CV_PIPELINE=true

# RAG Configuration
RAG_TOP_K=5
RAG_BATCH_SIZE=6
RETRIEVAL_MODE=hybrid  # vector_only, graph_only, hybrid

# YOLO Configuration
YOLO_MODEL=yolov8n-seg
YOLO_CONFIDENCE=0.5
YOLO_DEVICE=cuda:0
```

---

## Dependencies (requirements.txt)

```
# Core Framework
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
pydantic>=2.0.0

# Gemini SDK
google-generativeai>=0.8.0
google-auth>=2.0.0

# CV Pipeline
ultralytics>=8.3.0            # YOLOv8/v26
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0

# Video Processing
av>=11.0.0                    # PyAV for video decoding
ffmpeg-python>=0.2.0          # FFmpeg wrapper

# Tracking
supervision>=0.20.0           # ByteTrack integration

# Vector DB (Milvus) - Required
pymilvus>=2.5.0

# Graph DB (Neo4j) - Required
neo4j>=5.0.0

# Utilities
pillow>=10.0.0
aiofiles>=23.0.0
tenacity>=8.0.0               # Retry logic
pyyaml>=6.0.0
python-dotenv>=1.0.0
pyaml-env>=1.2.0              # YAML env var substitution

# UI (optional)
gradio>=4.0.0

# Development
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

### Dependency Categories

| Category | Packages | Purpose |
|----------|----------|---------|
| **Core Framework** | fastapi, uvicorn, pydantic | REST API server |
| **Gemini SDK** | google-generativeai, google-auth | Google AI integration |
| **CV Pipeline** | ultralytics, opencv-python, torch | Object detection |
| **Video Processing** | av, ffmpeg-python | Video decoding/encoding |
| **Tracking** | supervision | ByteTrack object tracking |
| **Vector DB** | pymilvus | Milvus client |
| **Graph DB** | neo4j | Neo4j client |
| **Utilities** | pillow, aiofiles, tenacity, pyyaml | General utilities |
| **UI** | gradio | Web interface |
| **Development** | pytest, pytest-asyncio | Testing |

---

## Performance Considerations and API Limits

### Gemini API Limits

| Limit | Value | Mitigation Strategy |
|-------|-------|---------------------|
| Video duration | 1 hour max | Chunk long videos into segments |
| File size | 2GB max | Compress or split large files |
| RPM (requests/min) | 60 (free tier) | Queue requests + rate limiting |
| TPM (tokens/min) | 1M (free tier) | Batch efficiently |
| Concurrent uploads | 10 | Use semaphore limiting |

### YOLOv26 Model Variants

| Variant | Parameters | mAP@50-95 | Speed (ms) | Use Case |
|---------|------------|-----------|------------|----------|
| yolov26n-seg | 3.2M | 37.2 | 1.2 | Real-time, edge devices |
| yolov26s-seg | 11.2M | 44.8 | 2.1 | Balanced performance |
| yolov26m-seg | 25.9M | 50.1 | 4.3 | Accuracy focused |
| yolov26l-seg | 43.7M | 52.8 | 6.8 | High accuracy |
| yolov26x-seg | 68.2M | 54.3 | 10.2 | Maximum accuracy |

### Error Handling

#### Gemini-Specific Errors

| Error | Cause | Handling |
|-------|-------|----------|
| `SAFETY_BLOCKED` | Content filtered | Log warning, return partial results |
| `QUOTA_EXCEEDED` | Rate limit hit | Exponential backoff, retry |
| `FILE_TOO_LARGE` | Video > 2GB | Split into smaller chunks |
| `UNSUPPORTED_FORMAT` | Bad video codec | Transcode to MP4/H.264 |
| `PROCESSING_FAILED` | File API error | Retry upload, fallback to frames |
| `CONTEXT_LENGTH_EXCEEDED` | Too many tokens | Truncate context, summarize first |

#### Retry Configuration

```python
RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 1.0,       # seconds
    "max_delay": 60.0,
    "exponential_base": 2,
    "retryable_errors": [
        "QUOTA_EXCEEDED",
        "INTERNAL_ERROR",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED"
    ]
}
```

#### Fallback Behavior

When native video upload fails:

1. Log error with details
2. Attempt frame-based fallback:
   - Extract key frames (1 fps)
   - Send as image sequence
   - Reduced quality but functional
3. If fallback fails: mark chunk as failed, continue with others

---

## Optimization Strategies

### 1. Parallel Processing

- **Video Upload**: Upload multiple chunks concurrently (max 3-5)
- **Pipeline Parallelism**: Run CV pipeline in parallel with VLM
- **Batch Operations**: Batch embedding generation for efficiency

### 2. Caching

| Cache Type | Description | TTL |
|------------|-------------|-----|
| Gemini File URIs | Cache uploaded file URIs | 48 hours (until expiration) |
| Embeddings | Cache embeddings for repeated queries | Session-based |
| Chat Contexts | LRU cache for frequent chat contexts | Configurable |

### 3. Resource Management

- **File Cleanup**: Delete uploaded files after processing
- **Streaming**: Stream large video files during upload
- **Precision**: Use FP16 for YOLO inference (half precision)

### 4. Database Optimization

#### Milvus Index Configuration

```python
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
```

#### Neo4j Indexes

Create indexes for frequently queried properties:

```cypher
CREATE INDEX chunk_stream_idx FOR (c:VideoChunk) ON (c.stream_id);
CREATE INDEX entity_name_idx FOR (e:Entity) ON (e.name);
CREATE INDEX event_type_idx FOR (e:Event) ON (e.event_type);
```

### 5. Hybrid Retrieval Configuration

```yaml
retrieval:
  mode: hybrid              # vector_only, graph_only, hybrid
  vector_weight: 0.6        # Weight for vector results
  graph_weight: 0.4         # Weight for graph results
  top_k: 5                  # Number of results
  rerank: true              # Enable reranking
  temporal_boost: 1.2       # Boost for adjacent chunks
```

### 6. Processing Configuration

```yaml
processing:
  chunk_duration: 60        # seconds per chunk
  chunk_overlap: 2          # seconds overlap between chunks
  max_concurrent_chunks: 4  # parallel chunk processing
```

### 7. Memory Management

- Use batch processing for large video files
- Implement streaming for video uploads
- Clear intermediate results after processing
- Use generators for large result sets

---

## Quick Start

1. **Copy environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials**:
   ```bash
   GEMINI_API_KEY=your_actual_api_key
   NEO4J_PASSWORD=your_secure_password
   ```

3. **Start services**:
   ```bash
   docker-compose up -d
   ```

4. **Verify services are healthy**:
   ```bash
   docker-compose ps
   ```

5. **Access the application**:
   - Gradio UI: http://localhost:7860
   - Neo4j Browser: http://localhost:7474
   - Milvus Health: http://localhost:9091/healthz
