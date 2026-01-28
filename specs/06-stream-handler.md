# Stream Handler Specification

## Overview

The `ViaStreamHandler` is the main orchestrator for video processing. It coordinates video chunking, VLM analysis, CV pipeline execution, and RAG context management. This is a simplified version of the original VSS engine's stream handler, adapted for the Gemini-based PoC.

## Gap Analysis

### Original Implementation
- `src/vss-engine/src/via_stream_handler.py` - Complex orchestrator with:
  - Multi-GPU support
  - Live RTSP stream handling
  - Prometheus metrics
  - Alert callbacks
  - Graph DB integration
  - Multiple VLM model types

### PoC Requirement
- Simplified orchestrator for file-based processing
- Coordinate Gemini video upload and analysis
- Handle CV pipeline for object detection
- Manage RAG context for summarization/chat
- No live stream support (out of scope)

## Component Location

```
./src/via_stream_handler.py
```

## Dependencies

```python
# Internal dependencies
from poc.src.models.gemini import GeminiFileManager, GeminiVLM, GeminiLLM, GeminiEmbeddings
from poc.src.cv_pipeline import YOLOPipeline, ObjectTracker, CVMetadataFuser
from poc.src.rag import ContextManager, Summarizer, Retriever
from poc.src.file_splitter import FileSplitter
from poc.src.chunk_info import ChunkInfo
from poc.src.asset_manager import AssetManager
```

## Data Classes

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from datetime import datetime


class RequestStatus(Enum):
    """Video processing request status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCESSFUL = "successful"
    FAILED = "failed"


@dataclass
class VlmRequestParams:
    """Parameters for VLM processing."""
    vlm_prompt: Optional[str] = None
    vlm_generation_config: Optional[dict] = None


@dataclass
class ProcessingResponse:
    """Response for a processed chunk."""
    start_timestamp: str
    end_timestamp: str
    response: str
    reasoning_description: str = ""


@dataclass
class ChunkResult:
    """Result of processing a single chunk."""
    chunk: ChunkInfo
    vlm_response: Optional[str] = None
    cv_metadata: Optional[dict] = None
    frame_times: list[float] = field(default_factory=list)
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class RequestInfo:
    """Information about a processing request."""
    request_id: str
    stream_id: str
    file: str
    status: RequestStatus = RequestStatus.QUEUED
    chunk_count: int = 0
    processed_chunks: list[ChunkResult] = field(default_factory=list)
    progress: float = 0.0
    responses: list[ProcessingResponse] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: str = ""

    # Processing parameters
    vlm_request_params: Optional[VlmRequestParams] = None
    enable_cv_pipeline: bool = False
    enable_chat: bool = True
    summarize: bool = True

    # Summarization parameters
    caption_summarization_prompt: str = ""
    summary_aggregation_prompt: str = ""
    summarize_batch_size: int = 6
```

## Class Interface

```python
import asyncio
import uuid
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor


class ViaStreamHandler:
    """Video processing stream handler for PoC."""

    def __init__(
        self,
        asset_manager: AssetManager,
        gemini_file_manager: GeminiFileManager,
        gemini_vlm: GeminiVLM,
        gemini_llm: GeminiLLM,
        gemini_embeddings: GeminiEmbeddings,
        yolo_pipeline: Optional[YOLOPipeline] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize the stream handler.

        Args:
            asset_manager: Asset storage manager
            gemini_file_manager: Gemini file upload manager
            gemini_vlm: Gemini VLM for video analysis
            gemini_llm: Gemini LLM for summarization/chat
            gemini_embeddings: Gemini embeddings for vector search
            yolo_pipeline: Optional YOLO pipeline for CV detection
            config: Configuration dictionary
        """
        pass

    async def process_video(
        self,
        file_path: str,
        vlm_prompt: str,
        chunk_duration: int = 60,
        chunk_overlap: int = 2,
        enable_cv_pipeline: bool = False,
        cv_text_prompt: str = "",
        on_chunk_complete: Optional[Callable[[ChunkResult], None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> RequestInfo:
        """
        Process a video file.

        Args:
            file_path: Path to video file
            vlm_prompt: Prompt for VLM analysis
            chunk_duration: Duration of each chunk in seconds
            chunk_overlap: Overlap between chunks in seconds
            enable_cv_pipeline: Enable object detection
            cv_text_prompt: Text prompt for CV detection (classes to detect)
            on_chunk_complete: Callback when a chunk is processed
            on_progress: Callback for progress updates

        Returns:
            RequestInfo with processing results
        """
        pass

    async def summarize(
        self,
        request_id: str,
        caption_summarization_prompt: str,
        summary_aggregation_prompt: str,
        batch_size: int = 6,
    ) -> str:
        """
        Generate summary for a processed video.

        Args:
            request_id: ID of the processed request
            caption_summarization_prompt: Prompt for caption summarization
            summary_aggregation_prompt: Prompt for summary aggregation
            batch_size: Number of captions per batch

        Returns:
            Final summary text
        """
        pass

    async def chat(
        self,
        request_id: str,
        question: str,
        chat_history: list[dict] = None,
        system_prompt: str = None,
    ) -> str:
        """
        Answer a question about a processed video.

        Args:
            request_id: ID of the processed request
            question: User question
            chat_history: Previous chat messages
            system_prompt: System prompt for chat

        Returns:
            Answer text
        """
        pass

    def get_request_status(self, request_id: str) -> RequestInfo:
        """Get status of a processing request."""
        pass

    def get_request_progress(self, request_id: str) -> float:
        """Get progress percentage of a request."""
        pass

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a processing request."""
        pass

    def cleanup_request(self, request_id: str) -> bool:
        """Clean up resources for a completed request."""
        pass

    async def _process_chunk(
        self,
        chunk: ChunkInfo,
        vlm_prompt: str,
        enable_cv_pipeline: bool,
        cv_text_prompt: str,
    ) -> ChunkResult:
        """Process a single video chunk."""
        pass

    async def _run_vlm_pipeline(
        self,
        chunk: ChunkInfo,
        vlm_prompt: str,
    ) -> tuple[str, list[float]]:
        """Run VLM analysis on a chunk."""
        pass

    async def _run_cv_pipeline(
        self,
        chunk: ChunkInfo,
        text_prompt: str,
    ) -> dict:
        """Run CV detection on a chunk."""
        pass

    def _fuse_metadata(
        self,
        vlm_response: str,
        cv_metadata: dict,
    ) -> str:
        """Fuse VLM response with CV metadata."""
        pass
```

## Implementation Notes

### Video Processing Flow

```python
async def process_video(
    self,
    file_path: str,
    vlm_prompt: str,
    chunk_duration: int = 60,
    chunk_overlap: int = 2,
    enable_cv_pipeline: bool = False,
    cv_text_prompt: str = "",
    on_chunk_complete: Optional[Callable[[ChunkResult], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> RequestInfo:
    # Create request
    request_id = str(uuid.uuid4())
    stream_id = str(uuid.uuid4())

    request = RequestInfo(
        request_id=request_id,
        stream_id=stream_id,
        file=file_path,
        status=RequestStatus.PROCESSING,
        start_time=datetime.now(),
        vlm_request_params=VlmRequestParams(vlm_prompt=vlm_prompt),
        enable_cv_pipeline=enable_cv_pipeline,
    )

    self._requests[request_id] = request

    try:
        # Split video into chunks
        chunks = await self._split_video(
            file_path,
            chunk_duration,
            chunk_overlap,
        )
        request.chunk_count = len(chunks)

        # Process chunks (can be parallel or sequential)
        for i, chunk in enumerate(chunks):
            chunk.streamId = stream_id

            result = await self._process_chunk(
                chunk,
                vlm_prompt,
                enable_cv_pipeline,
                cv_text_prompt,
            )

            request.processed_chunks.append(result)
            request.progress = (i + 1) / len(chunks) * 100

            if on_chunk_complete:
                on_chunk_complete(result)
            if on_progress:
                on_progress(request.progress)

            # Add to context manager for RAG
            if result.vlm_response:
                await self._context_manager.add_document(
                    text=result.vlm_response,
                    metadata={
                        "chunk_idx": chunk.chunkIdx,
                        "start_time": chunk.start_ntp,
                        "end_time": chunk.end_ntp,
                    },
                )

        request.status = RequestStatus.SUCCESSFUL
        request.end_time = datetime.now()

    except Exception as e:
        request.status = RequestStatus.FAILED
        request.error_message = str(e)
        request.end_time = datetime.now()

    return request
```

### Chunk Processing

```python
async def _process_chunk(
    self,
    chunk: ChunkInfo,
    vlm_prompt: str,
    enable_cv_pipeline: bool,
    cv_text_prompt: str,
) -> ChunkResult:
    import time
    start_time = time.time()

    result = ChunkResult(chunk=chunk)

    try:
        # Run VLM and CV pipelines concurrently
        tasks = [self._run_vlm_pipeline(chunk, vlm_prompt)]

        if enable_cv_pipeline and self._yolo_pipeline:
            tasks.append(self._run_cv_pipeline(chunk, cv_text_prompt))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process VLM result
        vlm_result = results[0]
        if isinstance(vlm_result, Exception):
            result.error = str(vlm_result)
        else:
            result.vlm_response, result.frame_times = vlm_result

        # Process CV result if available
        if len(results) > 1:
            cv_result = results[1]
            if isinstance(cv_result, Exception):
                # Log but don't fail the whole chunk
                pass
            else:
                result.cv_metadata = cv_result

                # Fuse metadata with VLM response
                if result.vlm_response:
                    result.vlm_response = self._fuse_metadata(
                        result.vlm_response,
                        result.cv_metadata,
                    )

    except Exception as e:
        result.error = str(e)

    result.processing_time = time.time() - start_time
    return result
```

### VLM Pipeline

```python
async def _run_vlm_pipeline(
    self,
    chunk: ChunkInfo,
    vlm_prompt: str,
) -> tuple[str, list[float]]:
    # Upload chunk to Gemini
    file_result = await self._gemini_file_manager.upload_and_wait(
        video_path=chunk.file,
        display_name=f"chunk_{chunk.chunkIdx}",
    )

    try:
        # Analyze with Gemini VLM
        analysis = await self._gemini_vlm.analyze_video(
            file_uri=file_result.uri,
            prompt=vlm_prompt,
        )

        return analysis.captions, []  # Frame times from video

    finally:
        # Clean up uploaded file
        await self._gemini_file_manager.delete_file(file_result.uri)
```

### CV Pipeline

```python
async def _run_cv_pipeline(
    self,
    chunk: ChunkInfo,
    text_prompt: str,
) -> dict:
    # Parse text prompt into target classes
    target_classes = [c.strip() for c in text_prompt.split(".") if c.strip()]

    # Run YOLO detection on video
    detections = self._yolo_pipeline.detect_video(
        video_path=chunk.file,
        frame_interval=5,  # Process every 5th frame
    )

    # Aggregate detection statistics
    class_counts = {}
    for fd in detections:
        for name in fd.detections.class_names:
            if not target_classes or name in target_classes:
                class_counts[name] = class_counts.get(name, 0) + 1

    return {
        "class_counts": class_counts,
        "total_frames": len(detections),
        "detections_per_frame": [
            {
                "frame_idx": fd.frame_idx,
                "timestamp": fd.timestamp,
                "objects": [
                    {"class": n, "confidence": float(c)}
                    for n, c in zip(
                        fd.detections.class_names,
                        fd.detections.confidences,
                    )
                ],
            }
            for fd in detections
        ],
    }
```

### Summarization

```python
async def summarize(
    self,
    request_id: str,
    caption_summarization_prompt: str,
    summary_aggregation_prompt: str,
    batch_size: int = 6,
) -> str:
    request = self._requests.get(request_id)
    if not request:
        raise ValueError(f"Request {request_id} not found")

    # Collect all captions
    captions = [
        result.vlm_response
        for result in request.processed_chunks
        if result.vlm_response
    ]

    if not captions:
        return "No captions available for summarization."

    # Batch summarization
    batch_summaries = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        summary = await self._gemini_llm.summarize_captions(
            captions=batch,
            prompt_template=caption_summarization_prompt,
        )
        batch_summaries.append(summary)

    # Aggregate if multiple batches
    if len(batch_summaries) == 1:
        return batch_summaries[0]

    return await self._gemini_llm.aggregate_summaries(
        summaries=batch_summaries,
        aggregation_prompt=summary_aggregation_prompt,
    )
```

### Chat/Q&A

```python
async def chat(
    self,
    request_id: str,
    question: str,
    chat_history: list[dict] = None,
    system_prompt: str = None,
) -> str:
    request = self._requests.get(request_id)
    if not request:
        raise ValueError(f"Request {request_id} not found")

    # Retrieve relevant context
    context = await self._context_manager.retrieve(
        query=question,
        top_k=5,
    )

    # Build context string
    context_str = "\n\n".join([
        f"[{doc['metadata']['start_time']} - {doc['metadata']['end_time']}]\n{doc['text']}"
        for doc in context
    ])

    # Build prompt with context
    prompt = f"""Context from video:
{context_str}

Question: {question}

Answer based on the context above. If the information is not available, say so."""

    # Generate response
    if chat_history:
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in chat_history
        ]
        messages.append({"role": "user", "content": prompt})

        result = await self._gemini_llm.chat(
            messages=messages,
            system_prompt=system_prompt,
        )
    else:
        result = await self._gemini_llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )

    return result.text
```

## Configuration

```yaml
# config/config.yaml
processing:
  chunk_duration: 60
  chunk_overlap: 2
  max_concurrent_chunks: 4

rag:
  enabled: true
  batch_size: 6
  top_k: 5
```

## Testing

```python
# tests/test_stream_handler.py

import pytest
from poc.src.via_stream_handler import ViaStreamHandler, RequestStatus


class TestViaStreamHandler:
    async def test_process_video(self):
        """Test video processing."""
        pass

    async def test_summarize(self):
        """Test summarization."""
        pass

    async def test_chat(self):
        """Test chat/Q&A."""
        pass

    async def test_cancel_request(self):
        """Test request cancellation."""
        pass
```
