# Video Search and Summarization (VSS) PoC

Video RAG system using Gemini VLM/LLM, YOLOv26, Milvus vector DB, and Neo4j graph DB.

## Build & Run

### Setup
```bash
# Install dependencies (use uv, already installed)
uv pip install -r requirements.txt

# Or with pyproject.toml
uv pip install -e .
```

### Run Infrastructure
```bash
# Start databases (Milvus + Neo4j)
docker-compose up -d milvus-standalone neo4j

# Verify health
docker-compose ps
```

### Run Application
```bash
# CLI mode
python run_poc.py --video /path/to/video.mp4 --enable-cv --summarize

# Interactive chat mode
python run_poc.py --video /path/to/video.mp4 --chat

# Gradio UI
python -m src.ui.gradio_app
# Access at http://localhost:7860
```

### Docker (Full Stack)
```bash
docker-compose up -d
```

## Validation

Run these after implementing to get immediate feedback:

- Tests: `pytest tests/ -v`
- Tests (async): `pytest tests/ -v --asyncio-mode=auto`
- Integration tests: `pytest tests/integration/ -v` (requires running DBs)
- Typecheck: `pyrefly check src/`
- Lint: `ruff check src/`

## Operational Notes

### Environment Variables (Required)
```bash
GEMINI_API_KEY=...          # Google AI Studio API key
MILVUS_HOST=localhost       # Milvus vector DB
MILVUS_PORT=19530
NEO4J_HOST=localhost        # Neo4j graph DB
NEO4J_BOLT_PORT=7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

### Key Configuration (config/config.yaml)
- `gemini.vlm.model`: Gemini model for video understanding (default: gemini-2.0-flash)
- `cv_pipeline.enabled`: Toggle YOLO object detection
- `rag.retrieval.mode`: vector_only | graph_only | hybrid
- `processing.chunk_duration`: Video chunk size in seconds (default: 60)

### API Rate Limits
- Gemini free tier: 60 RPM, 1M TPM
- Max video duration per chunk: 5 minutes
- Max concurrent uploads: 3-5 (use semaphore)

### Retry on Errors
Retryable Gemini errors: QUOTA_EXCEEDED, INTERNAL_ERROR, UNAVAILABLE, DEADLINE_EXCEEDED

## Codebase Patterns

### Directory Structure
```
src/
├── models/gemini/          # Gemini API wrappers (VLM, LLM, embeddings, file manager)
├── cv_pipeline/            # YOLO detection + ByteTrack tracking
├── db/                     # Milvus + Neo4j clients, graph ingestion/retrieval
├── rag/                    # Context manager, hybrid retrieval, RAG functions
└── ui/                     # Gradio interface
config/
├── config.yaml             # Main configuration
└── prompts/                # LLM prompt templates
```

### Key Classes
- `ViaStreamHandler`: Main orchestrator for video processing pipeline
- `GeminiFileManager`: Upload videos to Gemini File API
- `GeminiVLM`: Video understanding and captioning
- `GeminiLLM`: Text generation, summarization, chat
- `GeminiEmbeddings`: 768-dim embeddings via text-embedding-004
- `YOLOPipeline`: Object detection with YOLOv26-seg
- `HybridRetriever`: Combined vector + graph retrieval
- `ContextManager`: CA-RAG context management

### Data Flow
1. Video -> FileSplitter (chunks) -> GeminiFileManager (upload)
2. Parallel: GeminiVLM (captions) + YOLOPipeline (detections)
3. CVMetadataFuser (merge results)
4. Ingest to Milvus (vectors) + Neo4j (entities/events graph)
5. Query via HybridRetriever -> GeminiLLM (answer generation)

### Dataclasses
- `ChunkInfo`: Video chunk metadata (timestamps, PTS, stream_id)
- `VideoEvent`: Timestamped event from VLM caption
- `DetectionResult`: YOLO detection with bbox, class, confidence
- `Document`: RAG document with embedding and metadata

### Async Patterns
- Use `asyncio` for concurrent Gemini API calls
- Semaphore for rate limiting uploads (MAX_CONCURRENT_UPLOADS)
- `tenacity` for retry with exponential backoff

### Testing
- Mock Gemini API responses in unit tests
- Use pytest-asyncio for async tests
- Integration tests require running Milvus + Neo4j containers
