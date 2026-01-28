# Video Search and Summarization - PoC Specifications

This directory contains detailed component specifications for the VSS PoC implementation.

## Specification Index

| # | File | Description |
|---|------|-------------|
| 00 | [00-overview.md](./00-overview.md) | Project overview, architecture comparison, gap analysis |
| 01 | [01-gemini-file-manager.md](./01-gemini-file-manager.md) | Gemini File API wrapper for video uploads |
| 02 | [02-gemini-vlm.md](./02-gemini-vlm.md) | Gemini VLM for native video understanding |
| 03 | [03-gemini-llm.md](./03-gemini-llm.md) | Gemini LLM for summarization and chat |
| 04 | [04-gemini-embeddings.md](./04-gemini-embeddings.md) | Gemini text embeddings for vector search |
| 05 | [05-yolo-pipeline.md](./05-yolo-pipeline.md) | YOLOv26 object detection and ByteTrack tracking |
| 06 | [06-stream-handler.md](./06-stream-handler.md) | Stream handler orchestrating video processing |
| 07 | [07-rag-context-manager.md](./07-rag-context-manager.md) | CA-RAG context manager and functions |
| 08 | [08-milvus.md](./08-milvus.md) | Milvus vector database client and schema |
| 09 | [09-neo4j.md](./09-neo4j.md) | Neo4j graph database, ingestion, and retrieval |
| 10 | [10-config-deployment.md](./10-config-deployment.md) | Configuration, Docker, and deployment |

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│                    CLI / Gradio UI                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stream Handler (06)                           │
│              Orchestrates video processing                       │
└─────────────────────────────────────────────────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Gemini File Mgr │ │   YOLO Pipeline │ │  RAG Context    │
│      (01)       │ │      (05)       │ │  Manager (07)   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          ▼                   │           ┌──────┴──────┐
┌─────────────────┐           │           ▼             ▼
│   Gemini VLM    │           │    ┌───────────┐ ┌───────────┐
│      (02)       │           │    │  Milvus   │ │  Neo4j    │
└─────────────────┘           │    │   (08)    │ │   (09)    │
          │                   │    └───────────┘ └───────────┘
          ▼                   │           │             │
┌─────────────────┐           │           └──────┬──────┘
│   Gemini LLM    │◄──────────┴──────────────────┘
│      (03)       │
└─────────────────┘
          │
          ▼
┌─────────────────┐
│ Gemini Embed    │
│      (04)       │
└─────────────────┘
```

## Quick Links

### Gemini Integration
- [File Manager API](./01-gemini-file-manager.md#geminifilemanager-api)
- [VLM API](./02-gemini-vlm.md#geminivlm-api)
- [LLM API](./03-gemini-llm.md#geminillm-api)
- [Embeddings API](./04-gemini-embeddings.md#geminiembeddings-api)

### CV Pipeline
- [YOLOPipeline API](./05-yolo-pipeline.md#yolopipeline-api)
- [ObjectTracker API](./05-yolo-pipeline.md#objecttracker-api)

### RAG System
- [ContextManager API](./07-rag-context-manager.md#contextmanager-api)
- [HybridRetriever API](./07-rag-context-manager.md#hybridretriever-api)

### Databases
- [MilvusClient API](./08-milvus.md#milvusclient-api)
- [Neo4jClient API](./09-neo4j.md#neo4jclient-api)
- [GraphIngestion API](./09-neo4j.md#graphingestion-api)
- [GraphRetrieval API](./09-neo4j.md#graphretrieval-api)

### Configuration
- [Environment Variables](./10-config-deployment.md#environment-variables)
- [config.yaml](./10-config-deployment.md#full-configyaml-example)
- [Docker Compose](./10-config-deployment.md#docker-compose-configuration)

## Implementation Order

Recommended implementation sequence:

1. **Phase 1: Core Infrastructure**
   - `01-gemini-file-manager.md` - File upload foundation
   - `08-milvus.md` - Vector database setup
   - `09-neo4j.md` - Graph database setup

2. **Phase 2: Gemini Integration**
   - `02-gemini-vlm.md` - Video understanding
   - `03-gemini-llm.md` - Text generation
   - `04-gemini-embeddings.md` - Embedding generation

3. **Phase 3: Processing Pipeline**
   - `05-yolo-pipeline.md` - Object detection
   - `06-stream-handler.md` - Orchestration

4. **Phase 4: RAG System**
   - `07-rag-context-manager.md` - Context management and retrieval

5. **Phase 5: Deployment**
   - `10-config-deployment.md` - Configuration and Docker setup
