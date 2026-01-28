# Video Search and Summarization - PoC Requirements

## Purpose

This document serves as the **top-level requirements index** for the Video Search and Summarization (VSS) Proof-of-Concept. It defines the overarching Jobs to Be Done (JTBD) and maps each to the detailed component specifications in `specs/*.md`.

---

## Jobs to Be Done (JTBD)

### JTBD-1: Understand Video Content at Scale

**User Need:** "I need to automatically understand what's happening in hours of video footage without watching it manually."

**Solution:** Native video upload to Gemini VLM for automatic captioning with temporal awareness.

| Requirement | Spec Reference |
|-------------|----------------|
| Upload videos to Gemini File API | [01-gemini-file-manager.md](specs/01-gemini-file-manager.md) |
| Generate timestamped captions from video | [02-gemini-vlm.md](specs/02-gemini-vlm.md) |
| Detect and track objects in video | [05-yolo-pipeline.md](specs/05-yolo-pipeline.md) |
| Orchestrate video processing pipeline | [06-stream-handler.md](specs/06-stream-handler.md) |

---

### JTBD-2: Search and Query Video Content

**User Need:** "I need to ask questions about video content and get accurate answers with timestamps."

**Solution:** Hybrid RAG combining vector similarity search (Milvus) with graph-based entity/event retrieval (Neo4j).

| Requirement | Spec Reference |
|-------------|----------------|
| Store and search caption embeddings | [08-milvus.md](specs/08-milvus.md) |
| Track entities and events in knowledge graph | [09-neo4j.md](specs/09-neo4j.md) |
| Hybrid retrieval (vector + graph) | [07-rag-context-manager.md](specs/07-rag-context-manager.md) |
| Generate answers from retrieved context | [03-gemini-llm.md](specs/03-gemini-llm.md) |
| Generate embeddings for search | [04-gemini-embeddings.md](specs/04-gemini-embeddings.md) |

---

### JTBD-3: Summarize Video Content

**User Need:** "I need concise summaries of video content organized by activity, incident, or time period."

**Solution:** Batch summarization using LLM with context from both vector and graph databases.

| Requirement | Spec Reference |
|-------------|----------------|
| Batch caption summarization | [07-rag-context-manager.md](specs/07-rag-context-manager.md#summarizationfunction) |
| LLM-based summary generation | [03-gemini-llm.md](specs/03-gemini-llm.md) |
| Temporal context retrieval | [09-neo4j.md](specs/09-neo4j.md#graphretrieval-api) |

---

### JTBD-4: Extract Structured Knowledge

**User Need:** "I need to extract entities (people, vehicles, objects) and events (actions, incidents) from video for analysis."

**Solution:** LLM-based entity/event extraction with graph storage for relationship tracking.

| Requirement | Spec Reference |
|-------------|----------------|
| Entity extraction from captions | [09-neo4j.md](specs/09-neo4j.md#graphingestion-api) |
| Event extraction and classification | [09-neo4j.md](specs/09-neo4j.md#graphingestion-api) |
| Object detection enrichment | [05-yolo-pipeline.md](specs/05-yolo-pipeline.md) |
| Knowledge graph storage | [09-neo4j.md](specs/09-neo4j.md#graph-schema) |

---

### JTBD-5: Deploy and Operate the System

**User Need:** "I need to deploy and configure the system with appropriate infrastructure."

**Solution:** Containerized deployment with configurable components.

| Requirement | Spec Reference |
|-------------|----------------|
| Environment configuration | [10-config-deployment.md](specs/10-config-deployment.md#environment-variables) |
| Docker deployment | [10-config-deployment.md](specs/10-config-deployment.md#docker-compose-configuration) |
| System architecture overview | [00-overview.md](specs/00-overview.md) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                      (CLI / Gradio UI)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Stream Handler [06]                          │
│              Orchestrates all video processing                   │
└─────────────────────────────────────────────────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Gemini VLM     │ │   CV Pipeline   │ │  RAG Context    │
│  [01] [02]      │ │      [05]       │ │  Manager [07]   │
│                 │ │                 │ │                 │
│ - File Upload   │ │ - YOLO Detect   │ │ - Ingestion     │
│ - Captioning    │ │ - Tracking      │ │ - Retrieval     │
└─────────────────┘ └─────────────────┘ │ - Summarization │
          │                   │         └─────────────────┘
          │                   │                   │
          ▼                   │           ┌──────┴──────┐
┌─────────────────┐           │           ▼             ▼
│  Gemini LLM     │           │    ┌───────────┐ ┌───────────┐
│     [03]        │◄──────────┴────│  Milvus   │ │  Neo4j    │
│                 │                │   [08]    │ │   [09]    │
│ - Summarization │                │           │ │           │
│ - Chat/Q&A      │                │ - Vectors │ │ - Entities│
│ - Extraction    │                │ - Search  │ │ - Events  │
└─────────────────┘                └───────────┘ │ - Graph   │
          │                                      └───────────┘
          ▼
┌─────────────────┐
│ Gemini Embed    │
│     [04]        │
│                 │
│ - 768-dim vecs  │
└─────────────────┘
```

---

## Specification Index

| # | Spec File | Component | JTBD Coverage |
|---|-----------|-----------|---------------|
| 00 | [00-overview.md](specs/00-overview.md) | Architecture & Gap Analysis | All |
| 01 | [01-gemini-file-manager.md](specs/01-gemini-file-manager.md) | Video Upload to Gemini | JTBD-1 |
| 02 | [02-gemini-vlm.md](specs/02-gemini-vlm.md) | Video Understanding | JTBD-1 |
| 03 | [03-gemini-llm.md](specs/03-gemini-llm.md) | Text Generation | JTBD-2, JTBD-3 |
| 04 | [04-gemini-embeddings.md](specs/04-gemini-embeddings.md) | Embedding Generation | JTBD-2 |
| 05 | [05-yolo-pipeline.md](specs/05-yolo-pipeline.md) | Object Detection & Tracking | JTBD-1, JTBD-4 |
| 06 | [06-stream-handler.md](specs/06-stream-handler.md) | Processing Orchestration | JTBD-1 |
| 07 | [07-rag-context-manager.md](specs/07-rag-context-manager.md) | RAG Functions & Context | JTBD-2, JTBD-3, JTBD-4 |
| 08 | [08-milvus.md](specs/08-milvus.md) | Vector Database | JTBD-2 |
| 09 | [09-neo4j.md](specs/09-neo4j.md) | Graph Database | JTBD-2, JTBD-4 |
| 10 | [10-config-deployment.md](specs/10-config-deployment.md) | Configuration & Deployment | JTBD-5 |

---

## Technology Stack

| Layer | Original (NVIDIA) | PoC (Google/Ultralytics) |
|-------|-------------------|--------------------------|
| **VLM** | Cosmos Reason1 / NVILA / VILA 1.5 | Gemini 3.0 Pro |
| **LLM** | Llama 3.1 70B (NIM) | Gemini 3.0 Pro |
| **Embeddings** | NVIDIA llama-3.2-nv-embedqa-1b-v2 | Gemini text-embedding-004 |
| **Object Detection** | Grounding DINO + SAM (TensorRT) | YOLOv26-seg (Ultralytics) |
| **Vector DB** | Milvus | Milvus |
| **Graph DB** | Neo4j | Neo4j |
| **API Provider** | Local NIMs / NVIDIA Cloud | Google AI Studio API |

---

## Key Design Decisions

### 1. Native Video Upload
The PoC uses **native video upload** to Gemini rather than frame extraction. This leverages Gemini's built-in video understanding for more efficient processing and better temporal coherence.

**Trade-offs:**
- (+) Better temporal understanding
- (+) Simpler pipeline (no frame extraction for VLM)
- (-) Dependent on Gemini File API availability
- (-) 48-hour file expiration

### 2. Hybrid RAG Architecture
Combines **vector search** (semantic similarity) with **graph traversal** (entity/event relationships) for comprehensive context retrieval.

**Trade-offs:**
- (+) Better recall for complex queries
- (+) Entity tracking across video
- (-) Increased complexity
- (-) Two databases to maintain

### 3. Parallel CV Pipeline
Object detection runs **in parallel** with VLM captioning, then results are fused.

**Trade-offs:**
- (+) Faster processing
- (+) Enriched captions with detection counts
- (-) Potential inconsistencies between VLM and CV results

---

## Scope

### In Scope
- Full pipeline: Upload -> Caption -> Extract -> Store -> Query
- Gemini 3.0 Pro for all AI tasks
- YOLOv26-seg for object detection
- Milvus for vector search
- Neo4j for knowledge graph
- Hybrid retrieval (vector + graph)
- Gradio UI
- Docker deployment

### Out of Scope
- Live RTSP stream processing
- Kubernetes deployment
- Prometheus metrics
- Alert system with callbacks
- NeMo Guardrails integration
- Multi-GPU distribution
- TensorRT optimization

---

## Implementation Phases

### Phase 1: Core Infrastructure
1. [01-gemini-file-manager.md](specs/01-gemini-file-manager.md) - File upload foundation
2. [08-milvus.md](specs/08-milvus.md) - Vector database setup
3. [09-neo4j.md](specs/09-neo4j.md) - Graph database setup

### Phase 2: AI Integration
4. [02-gemini-vlm.md](specs/02-gemini-vlm.md) - Video understanding
5. [03-gemini-llm.md](specs/03-gemini-llm.md) - Text generation
6. [04-gemini-embeddings.md](specs/04-gemini-embeddings.md) - Embedding generation

### Phase 3: Processing Pipeline
7. [05-yolo-pipeline.md](specs/05-yolo-pipeline.md) - Object detection
8. [06-stream-handler.md](specs/06-stream-handler.md) - Orchestration

### Phase 4: RAG System
9. [07-rag-context-manager.md](specs/07-rag-context-manager.md) - Context management

### Phase 5: Deployment
10. [10-config-deployment.md](specs/10-config-deployment.md) - Configuration and Docker

---

## Success Criteria

| JTBD | Metric | Target |
|------|--------|--------|
| JTBD-1 | Caption accuracy | >85% event coverage |
| JTBD-1 | Processing throughput | 1 hour video in <10 min |
| JTBD-2 | Query relevance | >80% relevant results in top-5 |
| JTBD-3 | Summary quality | Captures all critical events |
| JTBD-4 | Entity extraction | >90% precision on key entities |
| JTBD-5 | Deployment | Single `docker-compose up` |

---

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# 2. Start infrastructure
docker-compose up -d milvus-standalone neo4j

# 3. Run the application
docker-compose up vss-poc

# 4. Access UI
open http://localhost:7860
```

See [10-config-deployment.md](specs/10-config-deployment.md) for detailed setup instructions.
