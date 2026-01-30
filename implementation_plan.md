# VSS PoC Implementation Plan

## Overview
Implementation checklist for the Video Search and Summarization Proof-of-Concept using Gemini 2.0 Flash and YOLOv26.

**Note:** This PoC processes local files directly without a REST API layer. Gradio UI (if used) reads files from their uploaded location.

**Key Features:**
- Gemini 2.0 Flash for VLM (native video upload) and LLM (RAG/summarization)
- Gemini text-embedding-004 for 768-dimensional embeddings
- YOLOv26-seg for object detection with ByteTrack tracking
- **Milvus** vector database for semantic search (required)
- **Neo4j** graph database for temporal/entity relationships (Graph RAG)
- Context-Aware RAG (CA-RAG) combining vector and graph retrieval via HybridRetriever

---

## Gap Analysis (Specs vs Implementation Plan)

The following gaps were identified by comparing `specs/*.md` with this implementation plan and have been addressed:

| Gap | Resolution |
|-----|------------|
| Missing `AssetManager` component (06-stream-handler.md) | Added Phase 4.3 |
| Incomplete Milvus schema - missing `start_pts`, `end_pts`, `cv_meta` fields (08-milvus.md) | Updated Phase 5.2 |
| Missing `HybridRetriever` configuration details (08-milvus.md) | Added `RetrievalMode`, `HybridConfig` in Phase 7.8 |
| Missing `ContextStore` abstraction (07-rag-context-manager.md) | Added Phase 7.5-7.6 with `InMemoryContextStore` and `MilvusContextStore` |
| Pipeline renamed to `ViaStreamHandler` (06-stream-handler.md) | Updated Phase 8.1 |
| Missing prompt files: `caption_warehouse.txt`, `aggregation.txt` | Added to Phase 9.2 |
| Missing dependencies: `pyaml-env`, `ffmpeg-python` (10-config-deployment.md) | Added to Phase 1 |
| Missing test files for new components | Added to Phase 13.1 |
| Configuration example incomplete | Updated with full `rag`, `cv_pipeline`, `processing` sections |

---

## Phase 1: Project Setup

- [x] Create `src` directory structure
- [x] Create `./pyproject.toml` with project metadata and dependencies
- [x] Create `./requirements.txt` with all dependencies:
  - Core: `pydantic`, `numpy`
  - Gemini: `google-generativeai`, `google-auth`
  - CV: `ultralytics`, `opencv-python`, `torch`
  - Video: `av`, `ffmpeg-python`
  - Tracking: `supervision`
  - Vector DB: `pymilvus`
  - Graph DB: `neo4j`
  - Utils: `pillow`, `aiofiles`, `tenacity`, `pyyaml`, `python-dotenv`, `pyaml-env`
  - UI: `gradio`
  - Dev: `pytest`, `pytest-asyncio`
- [x] Create `./config/config.yaml` with default configuration (tools, functions, context_manager sections)
- [x] Create `./.env.example` with required environment variables
- [x] Create `./src/__init__.py` and package structure

---

## Phase 2: Gemini Integration (Core Models)

### 2.1 Gemini File Manager (`./src/models/gemini/gemini_file_manager.py`)

- [x] Define `FileState` enum (PROCESSING, ACTIVE, FAILED)
- [x] Define `FileUploadResult` dataclass
- [x] Define `FileStatus` dataclass
- [x] Define custom exceptions (`VideoTooLongError`, `UnsupportedFormatError`, `UploadError`, `ProcessingTimeoutError`)
- [x] Implement `GeminiFileManager.__init__()` - initialize with API key
- [x] Implement `GeminiFileManager._validate_video()` - validate file format and size
- [x] Implement `GeminiFileManager.upload_video()` - upload video to Gemini File API
- [x] Implement `GeminiFileManager.wait_for_processing()` - poll until ACTIVE status
- [x] Implement `GeminiFileManager.upload_and_wait()` - convenience method combining upload + wait
- [x] Implement `GeminiFileManager.delete_file()` - delete uploaded file
- [x] Implement `GeminiFileManager.list_files()` - list all uploaded files
- [x] Implement `GeminiFileManager.get_file_status()` - get current file status
- [x] Implement `GeminiFileManager.upload_multiple()` - concurrent uploads with semaphore
- [x] Implement `GeminiFileManager.cleanup_expired_files()` - cleanup tracking of expired files
- [x] Write unit tests for `GeminiFileManager`

### 2.2 Gemini VLM (`./src/models/gemini/gemini_vlm.py`)

- [x] Define `VideoEvent` dataclass
- [x] Define `TokenUsage` dataclass
- [x] Define `SafetyRating` dataclass
- [x] Define `VideoAnalysisResult` dataclass
- [x] Define `GenerationConfig` dataclass
- [x] Define `SafetySettings` dataclass
- [x] Define custom exceptions (`VLMError`, `SafetyBlockedError`, `ContextLengthExceededError`, `GenerationError`)
- [x] Implement `GeminiVLM.__init__()` - initialize with API key and model config
- [x] Implement `GeminiVLM._build_generation_config()` - build config dict for API
- [x] Implement `GeminiVLM._build_safety_settings()` - build safety settings list
- [x] Implement `GeminiVLM._extract_usage()` - extract token usage from response
- [x] Implement `GeminiVLM._extract_safety_ratings()` - extract safety ratings
- [x] Implement `GeminiVLM.parse_events()` - parse timestamped events from caption text
- [x] Implement `GeminiVLM.analyze_video()` - analyze video with prompt
- [x] Implement `GeminiVLM.analyze_video_with_context()` - analyze with previous chunk context
- [x] Implement `GeminiVLM.analyze_video_batch()` - analyze multiple videos concurrently
- [x] Write unit tests for `GeminiVLM`

### 2.3 Gemini LLM (`./src/models/gemini/gemini_llm.py`)

- [x] Define `TokenUsage` dataclass (or reuse from VLM)
- [x] Define `SafetyRating` dataclass (or reuse from VLM)
- [x] Define `GenerationResult` dataclass
- [x] Define `Message` dataclass
- [x] Define `LLMGenerationConfig` dataclass
- [x] Implement `GeminiLLM.__init__()` - initialize with API key and model config
- [x] Implement `GeminiLLM._build_generation_config()` - build config dict
- [x] Implement `GeminiLLM._convert_messages()` - convert Message objects to Gemini format
- [x] Implement `GeminiLLM.generate()` - single-turn text generation
- [x] Implement `GeminiLLM.generate_stream()` - streaming text generation
- [x] Implement `GeminiLLM.chat()` - multi-turn chat
- [x] Implement `GeminiLLM.chat_stream()` - streaming multi-turn chat
- [x] Implement `GeminiLLM.summarize_captions()` - summarize batch of captions
- [x] Implement `GeminiLLM.aggregate_summaries()` - aggregate multiple summaries
- [x] Implement `GeminiLLM.check_notification()` - check caption for notification events
- [x] Write unit tests for `GeminiLLM`

### 2.4 Gemini Embeddings (`./src/models/gemini/gemini_embeddings.py`)

- [x] Define `TaskType` enum (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
- [x] Define `EmbeddingResult` dataclass
- [x] Define `BatchEmbeddingResult` dataclass
- [x] Implement `GeminiEmbeddings.__init__()` - initialize with API key
- [x] Implement `GeminiEmbeddings._truncate_text()` - truncate text to max length
- [x] Implement `GeminiEmbeddings._chunk_batch()` - split texts into batches
- [x] Implement `GeminiEmbeddings._embed_sync()` - synchronous embedding (for executor)
- [x] Implement `GeminiEmbeddings.embed_text()` - embed single text
- [x] Implement `GeminiEmbeddings.embed_batch()` - embed multiple texts
- [x] Implement `GeminiEmbeddings.embed_query()` - convenience for query embedding
- [x] Implement `GeminiEmbeddings.embed_document()` - convenience for document embedding
- [x] Implement `GeminiEmbeddings.embed_documents()` - batch document embedding
- [x] Implement `GeminiEmbeddings.cosine_similarity()` - static similarity calculation
- [x] Write unit tests for `GeminiEmbeddings`

### 2.5 Gemini Module Init (`./src/models/gemini/__init__.py`)

- [x] Export all Gemini classes
- [x] Export all dataclasses and enums
- [x] Export all exceptions

---

## Phase 3: CV Pipeline

### 3.1 YOLO Pipeline (`./src/cv_pipeline/yolo_pipeline.py`)

- [x] Define `DetectionResult` dataclass with filter methods
- [x] Define `FrameDetection` dataclass
- [x] Define `TrackedObject` dataclass
- [x] Define `TrackingResult` dataclass
- [x] Implement `YOLOPipeline.__init__()` - load model and configure
- [x] Implement `YOLOPipeline._filter_results()` - convert YOLO results to DetectionResult
- [x] Implement `YOLOPipeline._draw_annotations()` - draw boxes/masks on frame
- [x] Implement `YOLOPipeline.detect()` - single frame detection
- [x] Implement `YOLOPipeline.detect_batch()` - batch frame detection
- [x] Implement `YOLOPipeline.detect_video()` - process video file
- [x] Implement `YOLOPipeline.detect_video_stream()` - streaming video detection
- [x] Implement `YOLOPipeline.class_names` property
- [x] Write unit tests for `YOLOPipeline`

### 3.2 Object Tracker (`./src/cv_pipeline/tracker.py`)

- [x] Implement `ObjectTracker.__init__()` - initialize ByteTrack
- [x] Implement `ObjectTracker.update()` - update tracker with detections
- [x] Implement `ObjectTracker.reset()` - reset tracker state
- [x] Write unit tests for `ObjectTracker`

### 3.3 CV Metadata Fuser (`./src/cv_pipeline/cv_metadata_fuser.py`)

- [x] Implement `CVMetadataFuser.fuse()` - fuse VLM captions with CV metadata
- [x] Implement object counting and aggregation logic
- [x] Write unit tests for `CVMetadataFuser`

### 3.4 CV Pipeline Module Init (`./src/cv_pipeline/__init__.py`)

- [x] Export all CV pipeline classes

---

## Phase 4: Core Infrastructure

### 4.1 Chunk Info (`./src/chunk_info.py`)

- [x] Define `ChunkInfo` dataclass with all metadata fields:
  - `chunkIdx`, `streamId`, `file`, `start_ntp`, `end_ntp`
  - `start_pts`, `end_pts` (presentation timestamps in nanoseconds)
- [x] Implement timestamp formatting helpers
- [x] Implement chunk duration calculation

### 4.2 File Splitter (`./src/file_splitter.py`)

- [x] Implement `FileSplitter.__init__()` - configure chunk duration and overlap
- [x] Implement `FileSplitter.split()` - split video into chunks
- [x] Implement `FileSplitter._get_video_duration()` - get video duration using ffprobe
- [x] Implement `FileSplitter._extract_chunk()` - extract chunk using ffmpeg
- [x] Implement chunk overlap handling
- [ ] Write unit tests for `FileSplitter`

### 4.3 Asset Manager (`./src/asset_manager.py`)

- [x] Define `AssetManager` class for managing video assets and temporary files
- [x] Implement `AssetManager.__init__()` - configure storage paths
- [x] Implement `AssetManager.store_video()` - store uploaded video
- [x] Implement `AssetManager.get_chunk_path()` - get path for chunk storage
- [x] Implement `AssetManager.cleanup()` - cleanup temporary files
- [x] Implement `AssetManager.get_output_path()` - get output directory for results
- [ ] Write unit tests for `AssetManager`

### 4.4 Utilities (`./src/utils.py`)

- [x] Implement timestamp parsing utilities (HH:MM:SS to seconds)
- [x] Implement timestamp formatting utilities (seconds to HH:MM:SS)
- [x] Implement PTS conversion utilities (nanoseconds)
- [x] Implement file size formatting
- [x] Implement duration formatting
- [x] Implement retry decorator with exponential backoff

---

## Phase 5: Vector Database (Milvus)

### 5.1 Milvus Client (`./src/db/milvus_client.py`)

- [x] Define `MilvusConfig` dataclass (host, port, collection_name)
- [x] Define `VectorDocument` dataclass (id, text, embedding, metadata)
- [x] Define `SearchResult` dataclass (document, score)
- [x] Implement `MilvusClient.__init__()` - connect to Milvus
- [x] Implement `MilvusClient._ensure_collection()` - create collection with schema
- [x] Implement `MilvusClient._create_index()` - create vector index (IVF_FLAT, COSINE)
- [x] Implement `MilvusClient.insert()` - insert single document with embedding
- [x] Implement `MilvusClient.insert_batch()` - batch insert documents
- [x] Implement `MilvusClient.search()` - vector similarity search
- [x] Implement `MilvusClient.search_with_filter()` - search with metadata filter
- [x] Implement `MilvusClient.get_by_id()` - retrieve document by ID
- [x] Implement `MilvusClient.get_by_metadata()` - retrieve by metadata filter
- [x] Implement `MilvusClient.delete()` - delete document by ID
- [x] Implement `MilvusClient.delete_by_metadata()` - delete by metadata filter
- [x] Implement `MilvusClient.drop_collection()` - drop entire collection
- [x] Implement `MilvusClient.get_collection_stats()` - get collection statistics
- [x] Write unit tests for `MilvusClient`

### 5.2 Milvus Collection Schema

- [x] Define collection schema (`vss_poc_captions`):
  - `id` (VARCHAR, max 64, primary key) - unique document identifier
  - `text` (VARCHAR, max 65535) - VLM-generated caption text
  - `embedding` (FLOAT_VECTOR, dim=768) - Gemini text-embedding-004 vector
  - `stream_id` (VARCHAR, max 64) - video stream identifier
  - `chunk_idx` (INT64) - chunk index within video
  - `start_time` (VARCHAR, max 32) - chunk start timestamp (HH:MM:SS)
  - `end_time` (VARCHAR, max 32) - chunk end timestamp (HH:MM:SS)
  - `start_pts` (INT64) - presentation timestamp start (nanoseconds)
  - `end_pts` (INT64) - presentation timestamp end (nanoseconds)
  - `cv_meta` (VARCHAR, max 65535) - JSON-encoded CV detection metadata
  - `created_at` (INT64) - Unix timestamp
- [x] Create index on `embedding` field (IVF_FLAT, nlist=128, metric=COSINE)

---

## Phase 6: Graph Database (Neo4j)

### 6.1 Neo4j Client (`./src/db/neo4j_client.py`)

- [x] Define `Neo4jConfig` dataclass (host, port, username, password)
- [x] Define `GraphNode` dataclass (id, labels, properties)
- [x] Define `GraphRelationship` dataclass (start_node, end_node, type, properties)
- [x] Implement `Neo4jClient.__init__()` - connect to Neo4j
- [x] Implement `Neo4jClient.close()` - close connection
- [x] Implement `Neo4jClient.execute_query()` - execute Cypher query
- [x] Implement `Neo4jClient.create_node()` - create node with labels and properties
- [x] Implement `Neo4jClient.create_relationship()` - create relationship between nodes
- [x] Implement `Neo4jClient.get_node_by_id()` - retrieve node by ID
- [x] Implement `Neo4jClient.find_nodes()` - find nodes by label and properties
- [x] Implement `Neo4jClient.get_neighbors()` - get connected nodes
- [x] Implement `Neo4jClient.delete_node()` - delete node and relationships
- [x] Implement `Neo4jClient.clear_database()` - clear all nodes and relationships
- [x] Write unit tests for `Neo4jClient`

### 6.2 Graph Schema

- [x] Define node types:
  - `VideoChunk` - represents a video chunk
    - Properties: `chunk_id`, `stream_id`, `chunk_idx`, `start_time`, `end_time`, `caption`, `embedding_id`
  - `Entity` - extracted entity from caption (person, object, location)
    - Properties: `entity_id`, `name`, `type`, `first_seen`, `last_seen`
  - `Event` - detected event/action
    - Properties: `event_id`, `description`, `start_time`, `end_time`, `severity`
- [x] Define relationship types:
  - `FOLLOWS` - temporal sequence (VideoChunk → VideoChunk)
  - `CONTAINS` - chunk contains entity (VideoChunk → Entity)
  - `PARTICIPATES_IN` - entity participates in event (Entity → Event)
  - `OCCURS_IN` - event occurs in chunk (Event → VideoChunk)
  - `INTERACTS_WITH` - entity interaction (Entity → Entity)
- [x] Create indexes on frequently queried properties (via Neo4jClient.create_indexes())

### 6.3 Graph Ingestion (`./src/db/graph_ingestion.py`)

- [x] Implement `GraphIngestion.__init__()` - initialize with Neo4j client and LLM
- [x] Implement `GraphIngestion.extract_entities()` - extract entities from caption using LLM
- [x] Implement `GraphIngestion.extract_events()` - extract events from caption using LLM
- [x] Implement `GraphIngestion.ingest_chunk()` - ingest single chunk to graph
- [x] Implement `GraphIngestion.create_temporal_links()` - create FOLLOWS relationships
- [x] Implement `GraphIngestion.link_entities_across_chunks()` - link entities across chunks
- [x] Implement `GraphIngestion.ingest_batch()` - batch ingest multiple chunks
- [x] Write unit tests for `GraphIngestion`

### 6.4 Graph Retrieval (`./src/db/graph_retrieval.py`)

- [x] Implement `GraphRetrieval.__init__()` - initialize with Neo4j client and embeddings
- [x] Implement `GraphRetrieval.get_temporal_context()` - get chunks before/after a timestamp
- [x] Implement `GraphRetrieval.get_entity_timeline()` - get all chunks containing an entity
- [x] Implement `GraphRetrieval.get_related_events()` - get events related to a query
- [x] Implement `GraphRetrieval.traverse_from_chunk()` - traverse graph from a starting chunk
- [x] Implement `GraphRetrieval.find_entity_interactions()` - find entity-entity interactions
- [x] Implement `GraphRetrieval.get_event_context()` - get context around an event
- [x] Write unit tests for `GraphRetrieval`

---

## Phase 7: Context-Aware RAG (CA-RAG)

### 7.1 RAG Functions Base (`./src/rag/functions/base.py`)

- [ ] Define `RAGFunction` abstract base class
- [ ] Define `FunctionConfig` dataclass
- [ ] Define `FunctionResult` dataclass
- [ ] Implement abstract methods: `configure()`, `execute()`, `reset()`

### 7.2 Graph Ingestion Function (`./src/rag/functions/graph_ingestion.py`)

- [ ] Implement `GraphIngestionFunction.__init__()` - initialize with graph client
- [ ] Implement `GraphIngestionFunction.configure()` - configure batch size, params
- [ ] Implement `GraphIngestionFunction.execute()` - ingest document to graph
- [ ] Implement `GraphIngestionFunction.reset()` - clear graph for stream
- [ ] Write unit tests for `GraphIngestionFunction`

### 7.3 Graph Retrieval Function (`./src/rag/functions/graph_retrieval.py`)

- [ ] Implement `GraphRetrievalFunction.__init__()` - initialize with graph client and LLM
- [ ] Implement `GraphRetrievalFunction.configure()` - configure top_k, params
- [ ] Implement `GraphRetrievalFunction.execute()` - retrieve context and generate response
- [ ] Implement `GraphRetrievalFunction._build_context()` - combine vector and graph results
- [ ] Implement `GraphRetrievalFunction.reset()` - reset retrieval state
- [ ] Write unit tests for `GraphRetrievalFunction`

### 7.4 Batch Summarization Function (`./src/rag/functions/summarization.py`)

- [ ] Implement `SummarizationFunction.__init__()` - initialize with LLM
- [ ] Implement `SummarizationFunction.configure()` - configure batch size, prompts
- [ ] Implement `SummarizationFunction.execute()` - batch summarize captions
- [ ] Implement `SummarizationFunction._summarize_batch()` - summarize single batch
- [ ] Implement `SummarizationFunction._aggregate_summaries()` - aggregate batch summaries
- [ ] Implement `SummarizationFunction.reset()` - reset summarization state
- [ ] Write unit tests for `SummarizationFunction`

### 7.5 Context Store (`./src/rag/context_store.py`)

- [ ] Define `Document` dataclass (id, text, embedding, metadata, created_at)
- [ ] Define `RetrievalResult` dataclass (documents, scores, query, top_k)
- [ ] Define `ContextWindow` dataclass (documents, total_tokens, truncated)
- [ ] Define `ContextStore` abstract base class with methods:
  - `add_document()`, `get_document()`, `search()`, `get_all_documents()`, `delete_document()`, `clear()`
- [ ] Implement `InMemoryContextStore` - simple in-memory implementation with cosine similarity
- [ ] Write unit tests for `InMemoryContextStore`

### 7.6 Milvus Context Store (`./src/rag/milvus_store.py`)

- [ ] Implement `MilvusContextStore` extending `ContextStore`
- [ ] Implement `MilvusContextStore.__init__()` - connect to Milvus, ensure collection
- [ ] Implement `MilvusContextStore._ensure_collection()` - create collection with schema
- [ ] Implement all abstract methods using Milvus operations
- [ ] Write unit tests for `MilvusContextStore`

### 7.7 Context Manager (`./src/rag/context_manager.py`)

- [ ] Define `ContextManagerConfig` dataclass
- [ ] Implement `ContextManager.__init__()` - initialize with embeddings, store, max_context_tokens
- [ ] Implement `ContextManager.add_document()` - add document with optional embedding generation
- [ ] Implement `ContextManager.add_documents_batch()` - batch add documents
- [ ] Implement `ContextManager.retrieve()` - retrieve relevant documents for query
- [ ] Implement `ContextManager.get_context_window()` - get context window within token limit
- [ ] Implement `ContextManager.get_all_captions()` - get all captions for stream
- [ ] Implement `ContextManager.get_captions_by_time_range()` - filter by time range
- [ ] Implement `ContextManager.clear_context()` - clear context for stream
- [ ] Implement `ContextManager._estimate_tokens()` - estimate token count
- [ ] Implement `ContextManager._register_functions()` - register RAG functions
- [ ] Implement `ContextManager.call()` - call registered functions
- [ ] Implement `ContextManager.get_function()` - get function by name
- [ ] Write unit tests for `ContextManager`

### 7.8 Hybrid Retrieval (`./src/rag/hybrid_retrieval.py`)

- [ ] Define `RetrievalMode` enum (VECTOR_ONLY, GRAPH_ONLY, HYBRID)
- [ ] Define `HybridConfig` dataclass:
  - `mode` - retrieval strategy
  - `vector_weight` (default 0.6) - weight for vector results
  - `graph_weight` (default 0.4) - weight for graph results
  - `top_k` (default 5) - number of results
  - `rerank` (default True) - enable reranking
  - `temporal_boost` (default 1.2) - boost for adjacent chunks
- [ ] Implement `HybridRetriever.__init__()` - initialize with Milvus client, Neo4j client, embeddings, config
- [ ] Implement `HybridRetriever.retrieve()` - combined vector + graph retrieval
- [ ] Implement `HybridRetriever._vector_search()` - search Milvus with query embedding
- [ ] Implement `HybridRetriever._graph_search()` - search Neo4j for entities/events
- [ ] Implement `HybridRetriever._merge_results()` - merge and rank results using weights
- [ ] Implement `HybridRetriever._apply_temporal_boost()` - boost scores for temporally adjacent chunks
- [ ] Implement `HybridRetriever._rerank()` - optional LLM-based reranking step
- [ ] Write unit tests for `HybridRetriever`

### 7.9 RAG Module Init (`./src/rag/__init__.py`)

- [ ] Export all RAG classes: `ContextManager`, `ContextStore`, `InMemoryContextStore`, `MilvusContextStore`
- [ ] Export `HybridRetriever`, `HybridConfig`, `RetrievalMode`
- [ ] Export all function classes and dataclasses

---

## Phase 8: Main Pipeline

### 8.1 Stream Handler (`./src/via_stream_handler.py`)

- [ ] Define `RequestStatus` enum (QUEUED, PROCESSING, SUCCESSFUL, FAILED)
- [ ] Define `VlmRequestParams` dataclass
- [ ] Define `ProcessingResponse` dataclass (start_timestamp, end_timestamp, response, reasoning_description)
- [ ] Define `ChunkResult` dataclass (chunk, vlm_response, cv_metadata, frame_times, error, processing_time)
- [ ] Define `RequestInfo` dataclass with all processing state fields
- [ ] Implement `ViaStreamHandler.__init__()` - initialize with:
  - `AssetManager`
  - `GeminiFileManager`
  - `GeminiVLM`
  - `GeminiLLM`
  - `GeminiEmbeddings`
  - `YOLOPipeline` (optional)
  - `MilvusClient`
  - `Neo4jClient`
  - config dict
- [ ] Implement `ViaStreamHandler._init_context_manager()` - initialize CA-RAG context manager
- [ ] Implement `ViaStreamHandler._split_video()` - split video into chunks using FileSplitter
- [ ] Implement `ViaStreamHandler._run_vlm_pipeline()` - upload chunk to Gemini and analyze
- [ ] Implement `ViaStreamHandler._run_cv_pipeline()` - run YOLO detection on chunk
- [ ] Implement `ViaStreamHandler._fuse_metadata()` - fuse VLM + CV results using CVMetadataFuser
- [ ] Implement `ViaStreamHandler._ingest_to_rag()` - ingest chunk to Milvus + Neo4j
- [ ] Implement `ViaStreamHandler._process_chunk()` - process single chunk (VLM + CV in parallel)
- [ ] Implement `ViaStreamHandler.process_video()` - full video processing with callbacks
- [ ] Implement `ViaStreamHandler.summarize()` - generate summary using batch summarization
- [ ] Implement `ViaStreamHandler.chat()` - answer questions using hybrid retrieval
- [ ] Implement `ViaStreamHandler.get_request_status()` - get request status
- [ ] Implement `ViaStreamHandler.get_request_progress()` - get progress percentage
- [ ] Implement `ViaStreamHandler.cancel_request()` - cancel processing
- [ ] Implement `ViaStreamHandler.cleanup_request()` - cleanup resources (reset DBs for stream)
- [ ] Write unit tests for `ViaStreamHandler`

---

## Phase 9: Configuration & Prompts

### 9.1 Configuration (`./config/config.yaml`)

- [x] Define Gemini configuration (VLM, LLM, embeddings)
- [x] Define CV pipeline configuration (YOLO, tracker)
- [x] Define processing configuration (chunk duration, overlap)
- [x] Define Milvus configuration (host, port, collection settings)
- [x] Define Neo4j configuration (host, port, credentials)
- [x] Define RAG configuration (batch size, top_k, functions)
- [x] Define context_manager configuration (enabled functions)
- [x] Define logging configuration

### 9.2 Prompts

- [x] Create `./config/prompts/caption.txt` - VLM captioning prompt (timestamped events)
- [x] Create `./config/prompts/caption_warehouse.txt` - warehouse-specific prompt (safety focus)
- [x] Create `./config/prompts/summarization.txt` - caption summarization prompt (batch)
- [x] Create `./config/prompts/aggregation.txt` - summary aggregation prompt (cluster by category)
- [x] Create `./config/prompts/chat.txt` - chat system prompt (video analysis assistant)
- [x] Create `./config/prompts/entity_extraction.txt` - entity extraction prompt (PERSON, VEHICLE, OBJECT, LOCATION)
- [x] Create `./config/prompts/event_extraction.txt` - event extraction prompt (MOVEMENT, INTERACTION, SAFETY_INCIDENT, OPERATION, ANOMALY)

### 9.3 Configuration Loader (`./src/config.py`)

- [x] Implement `load_config()` - load YAML config
- [x] Implement environment variable substitution (!ENV syntax)
- [x] Implement `get_prompt()` - load prompt from file
- [x] Implement configuration validation
- [x] Implement `build_tools_config()` - build tools configuration dict
- [x] Implement `build_functions_config()` - build functions configuration dict

---

## Phase 10: Entry Point & CLI

### 10.1 Entry Point (`./run_poc.py`)

- [ ] Implement CLI argument parsing (argparse)
- [ ] Implement `--video` argument for input video path
- [ ] Implement `--output` argument for output directory
- [ ] Implement `--config` argument for config file path
- [ ] Implement `--prompt` argument for custom VLM prompt
- [ ] Implement `--enable-cv` flag for CV pipeline
- [ ] Implement `--summarize` flag to generate summary
- [ ] Implement `--chat` flag for interactive Q&A mode
- [ ] Implement `--reset-db` flag to reset databases before processing
- [ ] Implement configuration loading
- [ ] Implement database connection initialization
- [ ] Implement component initialization
- [ ] Implement video processing workflow
- [ ] Implement result output (JSON, console)
- [ ] Add graceful shutdown handling

---

## Phase 11: Gradio UI (Optional)

### 11.1 Gradio App (`./src/ui/gradio_app.py`)

- [ ] Implement video upload component
- [ ] Implement processing status display
- [ ] Implement progress bar
- [ ] Implement caption display (per chunk)
- [ ] Implement summary display
- [ ] Implement chat interface with history
- [ ] Implement CV detection toggle
- [ ] Implement configuration panel
- [ ] Implement database status display (Milvus, Neo4j connection)
- [ ] Implement results export (JSON download)

### 11.2 UI Module Init (`./src/ui/__init__.py`)

- [ ] Export Gradio app

---

## Phase 12: Docker & Deployment

### 12.1 Dockerfile (`./Dockerfile`)

- [ ] Create multi-stage Dockerfile
- [ ] Install Python dependencies
- [ ] Install ffmpeg
- [ ] Configure YOLO model download
- [ ] Set up entrypoint (CLI or Gradio)

### 12.2 Docker Compose (`./docker-compose.yaml`)

- [ ] Define vss-poc service (main application)
- [ ] Define milvus-standalone service
  - Image: `milvusdb/milvus:v2.5.4`
  - Ports: 19530 (gRPC), 9091 (HTTP)
  - Health check configuration
  - Volume for data persistence
- [ ] Define neo4j service (graph-db)
  - Image: `neo4j:5.26.4`
  - Ports: 7474 (HTTP), 7687 (Bolt)
  - Environment: NEO4J_AUTH, NEO4J_PLUGINS (apoc)
  - Volume for data persistence
- [ ] Configure environment variables
- [ ] Configure volumes for video storage
- [ ] Configure network for inter-service communication
- [ ] Configure GPU access (optional)
- [ ] Add depends_on with health checks

### 12.3 Environment Files

- [ ] Create `./.env.example` with all variables:
  - GEMINI_API_KEY (required)
  - MILVUS_HOST, MILVUS_PORT
  - NEO4J_HOST, NEO4J_PORT, NEO4J_USERNAME, NEO4J_PASSWORD
  - YOLO settings
  - Processing settings
- [ ] Document required vs optional variables

---

## Phase 13: Testing

### 13.1 Unit Tests

- [ ] `tests/test_gemini_file_manager.py`
- [ ] `tests/test_gemini_vlm.py`
- [ ] `tests/test_gemini_llm.py`
- [ ] `tests/test_gemini_embeddings.py`
- [ ] `tests/test_yolo_pipeline.py`
- [ ] `tests/test_tracker.py`
- [ ] `tests/test_file_splitter.py`
- [ ] `tests/test_asset_manager.py`
- [ ] `tests/test_milvus_client.py`
- [ ] `tests/test_neo4j_client.py`
- [ ] `tests/test_graph_ingestion.py`
- [ ] `tests/test_graph_retrieval.py`
- [ ] `tests/test_context_store.py`
- [ ] `tests/test_context_manager.py`
- [ ] `tests/test_hybrid_retrieval.py`
- [ ] `tests/test_stream_handler.py`

### 13.2 Integration Tests

- [ ] `tests/integration/test_gemini_integration.py` - test real Gemini API
- [ ] `tests/integration/test_milvus_integration.py` - test real Milvus
- [ ] `tests/integration/test_neo4j_integration.py` - test real Neo4j
- [ ] `tests/integration/test_video_processing.py` - end-to-end video test
- [ ] `tests/integration/test_rag_pipeline.py` - test full RAG pipeline

### 13.3 Test Fixtures

- [ ] Create test video files (short clips)
- [ ] Create mock responses for Gemini API
- [ ] Create test configuration files
- [ ] Create Docker Compose for test databases

---

## Phase 14: Documentation

### 14.1 README (`./README.md`)

- [ ] Project overview
- [ ] Architecture diagram
- [ ] Quick start guide
- [ ] CLI usage examples
- [ ] Configuration documentation
- [ ] Database setup instructions
- [ ] Gradio UI documentation
- [ ] Deployment instructions

---

## Phase 15: Final Integration & Testing

- [ ] End-to-end test with sample video
- [ ] Test vector search accuracy
- [ ] Test graph traversal queries
- [ ] Test hybrid retrieval quality
- [ ] Performance benchmarking
- [ ] Error handling verification
- [ ] Logging verification
- [ ] Memory usage profiling
- [ ] API rate limit handling verification
- [ ] Database connection resilience testing
- [ ] Cleanup and code review

---

## Success Criteria Checklist

- [ ] Upload and process video files up to 1 hour
- [ ] Generate timestamped captions using Gemini VLM
- [ ] Detect objects using YOLOv8/v26
- [ ] Store captions in Milvus with embeddings
- [ ] Build knowledge graph in Neo4j with entities and relationships
- [ ] Generate video summaries using batch summarization
- [ ] Answer questions using hybrid vector + graph retrieval
- [ ] Run without NVIDIA GPUs (CPU mode for YOLO)
- [ ] Deploy with Docker Compose (app + Milvus + Neo4j)

---

## Directory Structure

```
./
├── src/
│   ├── __init__.py
│   ├── via_stream_handler.py     # Main processing orchestrator
│   ├── chunk_info.py             # Chunk metadata
│   ├── file_splitter.py          # Video chunking
│   ├── asset_manager.py          # Video asset management
│   ├── config.py                 # Configuration loader
│   ├── utils.py                  # Utilities
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
│   │   ├── __init__.py
│   │   ├── yolo_pipeline.py
│   │   ├── tracker.py
│   │   └── cv_metadata_fuser.py
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── milvus_client.py      # Milvus vector DB client
│   │   ├── neo4j_client.py       # Neo4j graph DB client
│   │   ├── graph_ingestion.py    # Graph ingestion logic
│   │   └── graph_retrieval.py    # Graph retrieval logic
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── context_manager.py    # CA-RAG context manager
│   │   ├── context_store.py      # Abstract ContextStore + InMemoryContextStore
│   │   ├── milvus_store.py       # MilvusContextStore implementation
│   │   ├── hybrid_retrieval.py   # Combined vector + graph retrieval
│   │   └── functions/
│   │       ├── __init__.py
│   │       ├── base.py           # RAG function base class
│   │       ├── graph_ingestion.py
│   │       ├── graph_retrieval.py
│   │       └── summarization.py
│   │
│   └── ui/
│       ├── __init__.py
│       └── gradio_app.py         # Gradio UI (optional)
│
├── config/
│   ├── config.yaml
│   └── prompts/
│       ├── caption.txt
│       ├── caption_warehouse.txt
│       ├── summarization.txt
│       ├── aggregation.txt
│       ├── chat.txt
│       ├── entity_extraction.txt
│       └── event_extraction.txt
│
├── tests/
│   ├── test_gemini_file_manager.py
│   ├── test_gemini_vlm.py
│   ├── test_gemini_llm.py
│   ├── test_gemini_embeddings.py
│   ├── test_yolo_pipeline.py
│   ├── test_tracker.py
│   ├── test_file_splitter.py
│   ├── test_asset_manager.py
│   ├── test_milvus_client.py
│   ├── test_neo4j_client.py
│   ├── test_graph_ingestion.py
│   ├── test_graph_retrieval.py
│   ├── test_context_manager.py
│   ├── test_hybrid_retrieval.py
│   ├── test_stream_handler.py
│   └── integration/
│       ├── test_gemini_integration.py
│       ├── test_milvus_integration.py
│       ├── test_neo4j_integration.py
│       ├── test_video_processing.py
│       └── test_rag_pipeline.py
│
├── run_poc.py                    # Entry point (CLI)
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
├── .env.example
└── README.md
```

---

## Configuration Example

```yaml
# config/config.yaml

# Gemini Configuration
gemini:
  api_key: !ENV ${GEMINI_API_KEY}

  vlm:
    model: "gemini-2.0-flash"
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

  llm:
    model: "gemini-2.0-flash"
    generation_config:
      temperature: 0.3
      top_p: 0.9
      max_output_tokens: 4096

  embeddings:
    model: "text-embedding-004"
    dimensions: 768

# CV Pipeline Configuration
cv_pipeline:
  enabled: true
  yolo:
    model: "yolov26n-seg"
    confidence: 0.5
    iou_threshold: 0.45
    device: "cuda:0"
    half_precision: true
    target_classes:
      - person
      - car
      - truck
      - forklift
  tracker:
    algorithm: "bytetrack"
    track_thresh: 0.5
    track_buffer: 30
    match_thresh: 0.8

# Video Processing
processing:
  chunk_duration: 60
  chunk_overlap: 2
  max_concurrent_chunks: 4

# RAG Configuration
rag:
  enabled: true
  max_context_tokens: 100000

  vector_db:
    enabled: true
    type: milvus
    host: !ENV ${MILVUS_HOST:localhost}
    port: !ENV ${MILVUS_PORT:19530}
    collection_name: "vss_poc_captions"

  graph_db:
    enabled: true
    type: neo4j
    host: !ENV ${NEO4J_HOST:localhost}
    port: !ENV ${NEO4J_BOLT_PORT:7687}
    username: !ENV ${NEO4J_USERNAME:neo4j}
    password: !ENV ${NEO4J_PASSWORD}

  retrieval:
    mode: hybrid  # vector_only, graph_only, hybrid
    vector_weight: 0.6
    graph_weight: 0.4
    top_k: 5
    rerank: true
    temporal_boost: 1.2

  functions:
    summarization:
      batch_size: 6
    ingestion:
      extract_entities: true
      extract_events: true
      link_entities: true

# Tools Configuration (for CA-RAG)
tools:
  graph_db:
    type: neo4j
    params:
      host: !ENV ${NEO4J_HOST:localhost}
      port: !ENV ${NEO4J_BOLT_PORT:7687}
      username: !ENV ${NEO4J_USERNAME:neo4j}
      password: !ENV ${NEO4J_PASSWORD}
    tools:
      embedding: gemini_embedding

  vector_db:
    type: milvus
    params:
      host: !ENV ${MILVUS_HOST:localhost}
      port: !ENV ${MILVUS_PORT:19530}
      collection_name: "vss_poc_captions"
    tools:
      embedding: gemini_embedding

  gemini_vlm:
    type: vlm
    params:
      model: "gemini-2.0-flash"
      api_key: !ENV ${GEMINI_API_KEY}

  gemini_llm:
    type: llm
    params:
      model: "gemini-2.0-flash"
      api_key: !ENV ${GEMINI_API_KEY}
      max_tokens: 4096
      temperature: 0.3

  gemini_embedding:
    type: embedding
    params:
      model: "text-embedding-004"
      api_key: !ENV ${GEMINI_API_KEY}

functions:
  summarization:
    type: batch_summarization
    params:
      batch_size: 6
      prompts:
        caption_summarization: !include prompts/summarization.txt
        summary_aggregation: !include prompts/aggregation.txt
    tools:
      llm: gemini_llm
      db: vector_db

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

# Logging
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```
