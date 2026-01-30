# PoC Overview and Gap Analysis

## Executive Summary

This document provides an overview of the Video Search and Summarization (VSS) PoC that replaces NVIDIA's local models with Google's Gemini 2.0 Flash and Ultralytics YOLOv8-seg. The PoC simplifies the original architecture while maintaining core functionality.

## Implementation Status

> **Last Updated**: 2026-01-30

### End-to-End Test Readiness

| Mode | Status | Command |
|------|--------|---------|
| Basic (VLM only) | ✅ **Ready** | `python run_poc.py --video video.mp4` |
| With Summarization | ✅ **Ready** | `python run_poc.py --video video.mp4 --summarize` |
| With Chat | ✅ **Ready** | `python run_poc.py --video video.mp4 --chat` |
| With CV Pipeline | ✅ **Ready** | `python run_poc.py --video video.mp4 --enable-cv` |
| With Milvus | ✅ **Ready** | `python run_poc.py --video video.mp4 --enable-milvus` |
| With Neo4j | ✅ **Ready** | `python run_poc.py --video video.mp4 --enable-neo4j` |

### Test Video

- **File**: `sample_videos/car_accident.mp4` (130.65 seconds)
- **Expected Output**: `validation/car_accident.md`
- **Chunks**: 3 (at 60s default duration)

## Architecture Comparison

### Original VSS Engine

```
┌─────────────────────────────────────────────────────────────────┐
│                        VSS Engine                                │
├─────────────────────────────────────────────────────────────────┤
│  VLM Models:                                                     │
│  - Cosmos Reason1 7B (TensorRT/VLLM)                            │
│  - NVILA (TensorRT)                                              │
│  - VILA 1.5 (TensorRT)                                          │
│  - OpenAI Compatible (API)                                       │
├─────────────────────────────────────────────────────────────────┤
│  LLM:                                                            │
│  - Llama 3.1 70B via NVIDIA NIM                                 │
├─────────────────────────────────────────────────────────────────┤
│  Embeddings:                                                     │
│  - NVIDIA llama-3.2-nv-embedqa-1b-v2                            │
├─────────────────────────────────────────────────────────────────┤
│  CV Pipeline:                                                    │
│  - Grounding DINO (TensorRT)                                    │
│  - SAM (TensorRT)                                                │
│  - DeepStream integration                                        │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure:                                                 │
│  - Multi-GPU support                                             │
│  - Live RTSP streaming                                           │
│  - Neo4j graph DB                                                │
│  - Milvus vector DB                                              │
│  - Prometheus metrics                                            │
└─────────────────────────────────────────────────────────────────┘
```

### PoC Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          PoC Engine                              │
├─────────────────────────────────────────────────────────────────┤
│  VLM:                                                            │
│  - Gemini 2.0 Flash (Native Video Upload)                       │
├─────────────────────────────────────────────────────────────────┤
│  LLM:                                                            │
│  - Gemini 2.0 Flash                                              │
├─────────────────────────────────────────────────────────────────┤
│  Embeddings:                                                     │
│  - Gemini text-embedding-004 (768-dim)                          │
├─────────────────────────────────────────────────────────────────┤
│  CV Pipeline:                                                    │
│  - YOLOv8-seg (Ultralytics) - placeholder for YOLOv26           │
│  - ByteTrack (supervision)                                       │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure:                                                 │
│  - Single process                                                │
│  - File-based processing only                                    │
│  - Milvus 2.5.4 (optional, in-memory fallback)                  │
│  - Neo4j 5.26.4 (optional, for Graph RAG)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Component Mapping

| Original Component | PoC Replacement | Spec Document |
|-------------------|-----------------|---------------|
| Frame extraction + embedding | Native video upload | [01-gemini-file-manager.md](01-gemini-file-manager.md) |
| Cosmos/NVILA/VILA VLM | Gemini VLM | [02-gemini-vlm.md](02-gemini-vlm.md) |
| Llama 3.1 70B NIM | Gemini LLM | [03-gemini-llm.md](03-gemini-llm.md) |
| NVIDIA embeddings | Gemini embeddings | [04-gemini-embeddings.md](04-gemini-embeddings.md) |
| Grounding DINO + SAM | YOLOv26-seg | [05-yolo-pipeline.md](05-yolo-pipeline.md) |
| ViaStreamHandler | Simplified handler | [06-stream-handler.md](06-stream-handler.md) |
| CA-RAG Context Manager | Simple RAG | [07-rag-context-manager.md](07-rag-context-manager.md) |

**Note:** No REST API layer needed - the PoC processes local files directly. Gradio UI (if used) reads files from their uploaded location.

## Key Differences

### Video Processing

| Aspect | Original | PoC |
|--------|----------|-----|
| Input | Frame extraction | Native video upload |
| Processing | Local GPU inference | Cloud API |
| Timestamps | Overlay on frames | Extracted from video |
| Batching | Local batch | Concurrent API calls |

### Model Inference

| Aspect | Original | PoC |
|--------|----------|-----|
| VLM | Local TensorRT/VLLM | Gemini API |
| LLM | NVIDIA NIM API | Gemini API |
| Embeddings | NVIDIA API | Gemini API |
| CV | Local TensorRT | Local PyTorch |

### Infrastructure

| Aspect | Original | PoC |
|--------|----------|-----|
| GPU | Required (8 GPUs optimal) | Optional (for YOLO) |
| Setup | Complex (NGC, TRT build) | Simple (pip install) |
| Dependencies | NVIDIA stack | Google SDK + Ultralytics |
| Deployment | Docker + K8s | Single container |

## Out of Scope

The following features from the original VSS engine are **not** included in the PoC:

1. **Live RTSP stream processing** - File-based only
2. **Kubernetes deployment** - Single container (Docker Compose only)
3. **Prometheus metrics** - Basic health check only
4. **Alert callbacks** - No notification system
5. **NeMo Guardrails** - No content filtering
6. **Multi-GPU distribution** - Single process
7. **TensorRT optimization** - Standard PyTorch

> **Note**: Neo4j graph database IS included in the PoC for Graph RAG functionality.

## Directory Structure

```
./
├── src/
│   ├── pipeline.py                # Main processing pipeline
│   ├── chunk_info.py              # Chunk metadata
│   ├── file_splitter.py           # Video chunking
│   ├── utils.py                   # Utilities
│   │
│   ├── models/
│   │   └── gemini/
│   │       ├── __init__.py
│   │       ├── gemini_vlm.py
│   │       ├── gemini_llm.py
│   │       ├── gemini_embeddings.py
│   │       └── gemini_file_manager.py
│   │
│   ├── cv_pipeline/
│   │   ├── yolo_pipeline.py       # YOLOv26 wrapper
│   │   ├── tracker.py             # ByteTrack
│   │   └── cv_metadata_fuser.py   # Metadata fusion
│   │
│   ├── rag/
│   │   ├── context_manager.py
│   │   └── summarization.py
│   │
│   └── ui/
│       └── gradio_app.py          # Gradio UI (optional)
│
├── config/
│   ├── config.yaml
│   └── prompts/
│       ├── caption.txt
│       ├── summarization.txt
│       └── chat.txt
│
├── tests/
│   ├── test_gemini_file_manager.py
│   ├── test_gemini_vlm.py
│   ├── test_gemini_llm.py
│   ├── test_gemini_embeddings.py
│   ├── test_yolo_pipeline.py
│   ├── test_pipeline.py
│   └── test_context_manager.py
│
├── run_poc.py                     # Entry point (CLI)
├── requirements.txt
└── README.md
```

## Dependencies

```
# Core
pydantic>=2.0.0
numpy>=1.24.0

# Gemini SDK
google-generativeai>=0.8.0

# CV Pipeline
ultralytics>=8.3.0
opencv-python>=4.8.0
torch>=2.0.0

# Video Processing
av>=11.0.0

# Tracking
supervision>=0.20.0

# Vector DB (optional)
pymilvus>=2.3.0

# Utilities
pillow>=10.0.0
aiofiles>=23.0.0
tenacity>=8.0.0
pyyaml>=6.0.0
python-dotenv>=1.0.0

# UI (optional)
gradio>=4.0.0

# Development
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

## Environment Variables

```bash
# Required
GEMINI_API_KEY=your_api_key_here

# Optional - Gemini Settings
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004
GEMINI_MAX_VIDEO_DURATION=300

# Optional - YOLO Settings
YOLO_MODEL=yolov8n-seg
YOLO_CONFIDENCE=0.5
YOLO_DEVICE=cuda:0

# Optional - Processing
CHUNK_DURATION=60
MAX_CONCURRENT_UPLOADS=3
ENABLE_CV_PIPELINE=true

# Optional - Vector DB
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

## Implementation Order

Recommended order for implementing the PoC:

1. **Gemini Models**
   - `gemini_file_manager.py` - File upload
   - `gemini_vlm.py` - Video analysis
   - `gemini_llm.py` - Text generation
   - `gemini_embeddings.py` - Embeddings

2. **CV Pipeline**
   - `yolo_pipeline.py` - Object detection
   - `tracker.py` - Object tracking
   - `cv_metadata_fuser.py` - Metadata fusion

3. **Core Infrastructure**
   - `file_splitter.py` - Video chunking
   - `chunk_info.py` - Chunk metadata

4. **RAG Pipeline**
   - `context_manager.py` - Context storage
   - `summarization.py` - Batch summarization

5. **Main Pipeline**
   - `pipeline.py` - Processing orchestrator
   - `run_poc.py` - CLI entry point

6. **UI & Testing** (Optional)
   - Gradio UI
   - Integration tests

## Success Criteria

The PoC is considered successful if it can:

1. ✅ Upload and process video files up to 1 hour
2. ✅ Generate timestamped captions using Gemini VLM
3. ✅ Detect objects using YOLOv26
4. ✅ Generate video summaries
5. ✅ Answer questions about video content
6. ✅ Run without NVIDIA GPUs (CPU mode for YOLO)
7. ✅ Deploy with a single Docker container

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Gemini API rate limits | Implement retry with backoff |
| Video upload failures | Fallback to frame extraction |
| Large video files | Chunk into smaller segments |
| API costs | Monitor usage, implement caching |
| Model availability | Use stable model versions |
