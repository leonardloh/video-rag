# Video Search and Summarization (VSS) PoC

A video analysis pipeline using Google Gemini APIs (VLM, LLM, embeddings) and YOLOv26 for object detection. Processes video files to generate timestamped captions, extract entities/events, and enable semantic search via hybrid RAG (Milvus vector DB + Neo4j graph DB).

## Features

- **Video Understanding**: Native video upload to Gemini 2.0 Flash for VLM analysis
- **Object Detection**: YOLOv26-seg with ByteTrack tracking
- **Semantic Search**: 768-dimensional embeddings via Gemini text-embedding-004
- **Hybrid RAG**: Combined vector (Milvus) and graph (Neo4j) retrieval
- **Batch Summarization**: Hierarchical summarization of video captions
- **Interactive Chat**: Q&A about video content with context retrieval
- **Gradio UI**: Web interface for video upload, processing, and chat

## Architecture

```
                                    ┌─────────────────────────────────────┐
                                    │           VSS PoC Engine            │
                                    ├─────────────────────────────────────┤
┌──────────────┐                    │                                     │
│  Video File  │───────────────────►│  FileSplitter (chunks)              │
└──────────────┘                    │         │                           │
                                    │         ▼                           │
                                    │  ┌─────────────────────────────┐    │
                                    │  │    ViaStreamHandler         │    │
                                    │  │    (Main Orchestrator)      │    │
                                    │  └──────────┬──────────────────┘    │
                                    │             │                       │
                                    │     ┌───────┴───────┐               │
                                    │     ▼               ▼               │
                                    │ ┌────────┐    ┌──────────┐          │
                                    │ │GeminiVLM│    │YOLOPipeline│       │
                                    │ │(captions)│   │(detections)│       │
                                    │ └────┬────┘    └─────┬─────┘        │
                                    │      │              │               │
                                    │      └──────┬───────┘               │
                                    │             ▼                       │
                                    │    ┌────────────────┐               │
                                    │    │CVMetadataFuser │               │
                                    │    └───────┬────────┘               │
                                    │            │                        │
                                    │     ┌──────┴──────┐                 │
                                    │     ▼             ▼                 │
                                    │ ┌────────┐   ┌─────────┐            │
                                    │ │ Milvus │   │  Neo4j  │            │
                                    │ │(vectors)│  │ (graph) │            │
                                    │ └────────┘   └─────────┘            │
                                    │                                     │
                                    │         HybridRetriever             │
                                    │              │                      │
                                    │              ▼                      │
                                    │         GeminiLLM                   │
                                    │     (summarize/chat)                │
                                    └─────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for databases)
- Google AI Studio API key ([Get one here](https://aistudio.google.com/))

### Installation

```bash
# Clone the repository
git clone https://github.com/leonardloh/video-rag.git
cd video-rag

# Install dependencies using uv (recommended)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# Required: GEMINI_API_KEY
# Optional: MILVUS_HOST, NEO4J_HOST, etc.
```

### Start Databases

```bash
# Start Milvus and Neo4j
docker-compose up -d milvus-standalone neo4j

# Verify services are healthy
docker-compose ps
```

### Run the Application

#### CLI Mode

```bash
# Basic video processing
python run_poc.py --video /path/to/video.mp4

# With CV pipeline (object detection)
python run_poc.py --video /path/to/video.mp4 --enable-cv

# Generate summary after processing
python run_poc.py --video /path/to/video.mp4 --summarize

# Interactive chat mode
python run_poc.py --video /path/to/video.mp4 --chat

# Full pipeline with all features
python run_poc.py --video /path/to/video.mp4 --enable-cv --enable-milvus --enable-neo4j --summarize --chat
```

#### Gradio UI

```bash
# Start the web interface
python -m src.ui.gradio_app

# Access at http://localhost:7860
```

#### Docker Deployment

```bash
# Build and run full stack
docker-compose up -d

# Access Gradio UI at http://localhost:7860
# Access Neo4j Browser at http://localhost:7474
```

## CLI Reference

```
usage: run_poc.py [-h] --video VIDEO [--output OUTPUT] [--config CONFIG]
                  [--prompt PROMPT] [--enable-cv] [--enable-milvus]
                  [--enable-neo4j] [--summarize] [--chat]
                  [--log-level {DEBUG,INFO,WARNING,ERROR}]

VSS PoC - Video Search and Summarization

options:
  -h, --help            show this help message and exit
  --video VIDEO, -v VIDEO
                        Path to input video file
  --output OUTPUT, -o OUTPUT
                        Output directory for results
  --config CONFIG, -c CONFIG
                        Path to configuration file (default: config/config.yaml)
  --prompt PROMPT, -p PROMPT
                        Custom VLM prompt (or path to prompt file)
  --enable-cv           Enable CV pipeline (YOLO object detection)
  --enable-milvus       Enable Milvus vector database
  --enable-neo4j        Enable Neo4j graph database
  --summarize, -s       Generate video summary after processing
  --chat                Enter interactive chat mode after processing
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Logging level (default: INFO)
```

## Configuration

The main configuration file is `config/config.yaml`. Key sections:

### Gemini Settings

```yaml
gemini:
  api_key: !ENV ${GEMINI_API_KEY}
  vlm:
    model: "gemini-2.0-flash"
    generation_config:
      temperature: 0.2
      max_output_tokens: 2048
  llm:
    model: "gemini-2.0-flash"
  embeddings:
    model: "text-embedding-004"
    dimensions: 768
```

### CV Pipeline

```yaml
cv_pipeline:
  enabled: true
  yolo:
    model: "yolov8n-seg"  # n, s, m, l, x variants
    confidence: 0.5
    device: "cpu"  # or "cuda:0"
```

### RAG Settings

```yaml
rag:
  retrieval:
    mode: hybrid  # vector_only, graph_only, hybrid
    vector_weight: 0.6
    graph_weight: 0.4
    top_k: 5
```

### Processing

```yaml
processing:
  chunk_duration: 60  # seconds per chunk
  chunk_overlap: 2    # overlap between chunks
  max_concurrent_chunks: 4
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | - | Google AI Studio API key |
| `MILVUS_HOST` | No | localhost | Milvus vector DB host |
| `MILVUS_PORT` | No | 19530 | Milvus gRPC port |
| `NEO4J_HOST` | No | localhost | Neo4j graph DB host |
| `NEO4J_BOLT_PORT` | No | 7687 | Neo4j Bolt port |
| `NEO4J_USERNAME` | No | neo4j | Neo4j username |
| `NEO4J_PASSWORD` | No | - | Neo4j password |
| `YOLO_MODEL` | No | yolov8n-seg | YOLO model variant |
| `YOLO_DEVICE` | No | cpu | Device for YOLO inference |
| `CHUNK_DURATION` | No | 60 | Video chunk duration (seconds) |

## Project Structure

```
video-rag/
├── src/
│   ├── models/gemini/          # Gemini API wrappers
│   │   ├── gemini_vlm.py       # Video understanding
│   │   ├── gemini_llm.py       # Text generation
│   │   ├── gemini_embeddings.py # Embeddings
│   │   └── gemini_file_manager.py # File upload
│   ├── cv_pipeline/            # Computer vision
│   │   ├── yolo_pipeline.py    # YOLOv26 detection
│   │   ├── tracker.py          # ByteTrack tracking
│   │   └── cv_metadata_fuser.py # Metadata fusion
│   ├── db/                     # Database clients
│   │   ├── milvus_client.py    # Vector DB
│   │   ├── neo4j_client.py     # Graph DB
│   │   ├── graph_ingestion.py  # Entity/event extraction
│   │   └── graph_retrieval.py  # Graph queries
│   ├── rag/                    # RAG components
│   │   ├── context_manager.py  # Context management
│   │   ├── hybrid_retrieval.py # Vector + graph retrieval
│   │   └── functions/          # RAG functions
│   ├── ui/                     # Gradio interface
│   │   └── gradio_app.py
│   ├── via_stream_handler.py   # Main orchestrator
│   ├── file_splitter.py        # Video chunking
│   └── asset_manager.py        # Asset management
├── config/
│   ├── config.yaml             # Main configuration
│   └── prompts/                # LLM prompt templates
├── tests/                      # Unit tests
├── run_poc.py                  # CLI entry point
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

## API Rate Limits

Gemini API has the following limits (free tier):

| Limit | Value |
|-------|-------|
| Requests per minute | 60 |
| Tokens per minute | 1,000,000 |
| Max video duration | 1 hour |
| Max file size | 2 GB |
| Max concurrent uploads | 10 |

The application implements automatic retry with exponential backoff for rate limit errors.

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with async support
pytest tests/ -v --asyncio-mode=auto

# Run specific test file
pytest tests/test_gemini_vlm.py -v
```

### Type Checking

```bash
pyrefly check src/
```

### Linting

```bash
ruff check src/
```

## Troubleshooting

### Common Issues

1. **GEMINI_API_KEY not set**
   ```
   ValueError: GEMINI_API_KEY environment variable is required
   ```
   Solution: Set the environment variable or add it to `.env`

2. **Milvus connection failed**
   ```
   Failed to initialize Milvus client
   ```
   Solution: Ensure Milvus is running (`docker-compose up -d milvus-standalone`)

3. **YOLO model download fails**
   ```
   Failed to initialize YOLO pipeline
   ```
   Solution: Check internet connection; model downloads on first use

4. **Video processing timeout**
   ```
   ProcessingTimeoutError
   ```
   Solution: Reduce chunk duration or check Gemini API status

### Logs

Enable debug logging for more details:

```bash
python run_poc.py --video video.mp4 --log-level DEBUG
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Google Gemini](https://ai.google.dev/) - VLM, LLM, and embeddings
- [Ultralytics](https://ultralytics.com/) - YOLOv8/v26
- [Milvus](https://milvus.io/) - Vector database
- [Neo4j](https://neo4j.com/) - Graph database
- [Gradio](https://gradio.app/) - Web interface
