"""Video processing stream handler - main orchestrator for VSS PoC."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from .asset_manager import AssetManager
from .chunk_info import ChunkInfo
from .file_splitter import FileSplitter
from .models.gemini.gemini_embeddings import GeminiEmbeddings
from .models.gemini.gemini_file_manager import GeminiFileManager
from .models.gemini.gemini_llm import GeminiLLM
from .models.gemini.gemini_vlm import GeminiVLM
from .rag.context_manager import ContextManager
from .rag.context_store import InMemoryContextStore
from .rag.functions.summarization import (
    SummarizationFunction,
    SummarizationInput,
)
from .rag.hybrid_retrieval import HybridConfig, HybridRetriever, RetrievalMode

# Optional CV pipeline import (requires cv2)
try:
    from .cv_pipeline.cv_metadata_fuser import CVMetadataFuser

    _CV_AVAILABLE = True
except ImportError:
    CVMetadataFuser = None  # type: ignore[misc, assignment]
    _CV_AVAILABLE = False

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Video processing request status."""

    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VlmRequestParams:
    """Parameters for VLM processing."""

    vlm_prompt: Optional[str] = None
    vlm_generation_config: Optional[dict[str, Any]] = None


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
    cv_metadata: Optional[dict[str, Any]] = None
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


# Type alias for optional YOLO pipeline
try:
    from .cv_pipeline.yolo_pipeline import YOLOPipeline
except ImportError:
    YOLOPipeline = None  # type: ignore[misc, assignment]

# Type alias for optional database clients
try:
    from .db.milvus_client import MilvusClient
except ImportError:
    MilvusClient = None  # type: ignore[misc, assignment]

try:
    from .db.neo4j_client import Neo4jClient
except ImportError:
    Neo4jClient = None  # type: ignore[misc, assignment]


class ViaStreamHandler:
    """Video processing stream handler for PoC.

    Main orchestrator that coordinates:
    - Video chunking via FileSplitter
    - VLM analysis via GeminiVLM
    - CV detection via YOLOPipeline (optional)
    - RAG context management via ContextManager
    - Summarization and chat via GeminiLLM
    """

    def __init__(
        self,
        asset_manager: AssetManager,
        gemini_file_manager: GeminiFileManager,
        gemini_vlm: GeminiVLM,
        gemini_llm: GeminiLLM,
        gemini_embeddings: GeminiEmbeddings,
        yolo_pipeline: Optional[Any] = None,
        milvus_client: Optional[Any] = None,
        neo4j_client: Optional[Any] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the stream handler.

        Args:
            asset_manager: Asset storage manager
            gemini_file_manager: Gemini file upload manager
            gemini_vlm: Gemini VLM for video analysis
            gemini_llm: Gemini LLM for summarization/chat
            gemini_embeddings: Gemini embeddings for vector search
            yolo_pipeline: Optional YOLO pipeline for CV detection
            milvus_client: Optional Milvus client for vector DB
            neo4j_client: Optional Neo4j client for graph DB
            config: Configuration dictionary
        """
        self._asset_manager = asset_manager
        self._gemini_file_manager = gemini_file_manager
        self._gemini_vlm = gemini_vlm
        self._gemini_llm = gemini_llm
        self._gemini_embeddings = gemini_embeddings
        self._yolo_pipeline = yolo_pipeline
        self._milvus_client = milvus_client
        self._neo4j_client = neo4j_client
        self._config = config or {}

        # Request tracking
        self._requests: dict[str, RequestInfo] = {}
        self._cancel_flags: dict[str, bool] = {}

        # Context managers per stream
        self._context_managers: dict[str, ContextManager] = {}

        # Hybrid retriever (if DBs available)
        self._hybrid_retriever: Optional[HybridRetriever] = None
        if milvus_client or neo4j_client:
            self._hybrid_retriever = HybridRetriever(
                milvus_client=milvus_client,
                neo4j_client=neo4j_client,
                embeddings=gemini_embeddings,
                llm=gemini_llm,
                config=HybridConfig(
                    mode=RetrievalMode.HYBRID if milvus_client and neo4j_client else RetrievalMode.VECTOR_ONLY,
                    top_k=self._config.get("rag", {}).get("top_k", 5),
                ),
            )

        # CV metadata fuser (optional)
        self._cv_fuser: Optional[Any] = None
        if _CV_AVAILABLE and CVMetadataFuser is not None:
            self._cv_fuser = CVMetadataFuser()

        # Summarization function
        self._summarization_function = SummarizationFunction(
            llm=gemini_llm,
            batch_size=self._config.get("rag", {}).get("batch_size", 6),
        )

    def _get_context_manager(self, stream_id: str) -> ContextManager:
        """Get or create context manager for a stream."""
        if stream_id not in self._context_managers:
            self._context_managers[stream_id] = ContextManager(
                embeddings=self._gemini_embeddings,
                store=InMemoryContextStore(),
                max_context_tokens=self._config.get("rag", {}).get("max_context_tokens", 100000),
            )
        return self._context_managers[stream_id]

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
        self._cancel_flags[request_id] = False

        try:
            # Split video into chunks
            chunks = await self._split_video(
                file_path=file_path,
                stream_id=stream_id,
                chunk_duration=chunk_duration,
                chunk_overlap=chunk_overlap,
            )
            request.chunk_count = len(chunks)

            logger.info(f"Split video into {len(chunks)} chunks")

            # Process chunks sequentially (to maintain context)
            for i, chunk in enumerate(chunks):
                # Check for cancellation
                if self._cancel_flags.get(request_id, False):
                    request.status = RequestStatus.CANCELLED
                    request.error_message = "Processing cancelled by user"
                    break

                chunk.streamId = stream_id

                result = await self._process_chunk(
                    chunk=chunk,
                    vlm_prompt=vlm_prompt,
                    enable_cv_pipeline=enable_cv_pipeline,
                    cv_text_prompt=cv_text_prompt,
                )

                request.processed_chunks.append(result)
                request.progress = (i + 1) / len(chunks) * 100

                if on_chunk_complete:
                    on_chunk_complete(result)
                if on_progress:
                    on_progress(request.progress)

                # Add to context manager for RAG
                if result.vlm_response:
                    context_manager = self._get_context_manager(stream_id)
                    await context_manager.add_document(
                        text=result.vlm_response,
                        metadata={
                            "stream_id": stream_id,
                            "chunk_idx": chunk.chunkIdx,
                            "start_time": chunk.start_ntp,
                            "end_time": chunk.end_ntp,
                            "start_pts": chunk.start_pts,
                            "end_pts": chunk.end_pts,
                            "cv_meta": json.dumps(result.cv_metadata) if result.cv_metadata else None,
                        },
                    )

                    # Create response
                    request.responses.append(
                        ProcessingResponse(
                            start_timestamp=chunk.start_ntp,
                            end_timestamp=chunk.end_ntp,
                            response=result.vlm_response,
                        )
                    )

            if request.status != RequestStatus.CANCELLED:
                request.status = RequestStatus.SUCCESSFUL
            request.end_time = datetime.now()

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            request.status = RequestStatus.FAILED
            request.error_message = str(e)
            request.end_time = datetime.now()

        return request

    async def _split_video(
        self,
        file_path: str,
        stream_id: str,
        chunk_duration: int,
        chunk_overlap: int,
    ) -> list[ChunkInfo]:
        """Split video into chunks."""
        splitter = FileSplitter(
            chunk_duration=chunk_duration,
            chunk_overlap=chunk_overlap,
            output_dir=self._asset_manager.get_output_path(stream_id),
        )
        return await splitter.split(file_path, stream_id)

    async def _process_chunk(
        self,
        chunk: ChunkInfo,
        vlm_prompt: str,
        enable_cv_pipeline: bool,
        cv_text_prompt: str,
    ) -> ChunkResult:
        """Process a single video chunk."""
        start_time = time.time()
        result = ChunkResult(chunk=chunk)

        try:
            # Run VLM and CV pipelines concurrently
            tasks: list[Any] = [self._run_vlm_pipeline(chunk, vlm_prompt)]

            if enable_cv_pipeline and self._yolo_pipeline:
                tasks.append(self._run_cv_pipeline(chunk, cv_text_prompt))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process VLM result
            vlm_result = results[0]
            if isinstance(vlm_result, Exception):
                result.error = str(vlm_result)
                logger.error(f"VLM pipeline failed for chunk {chunk.chunkIdx}: {vlm_result}")
            else:
                result.vlm_response, result.frame_times = vlm_result

            # Process CV result if available
            if len(results) > 1:
                cv_result = results[1]
                if isinstance(cv_result, Exception):
                    logger.warning(f"CV pipeline failed for chunk {chunk.chunkIdx}: {cv_result}")
                else:
                    result.cv_metadata = cv_result

                    # Fuse metadata with VLM response
                    if result.vlm_response and result.cv_metadata:
                        result.vlm_response = self._fuse_metadata(
                            result.vlm_response,
                            result.cv_metadata,
                        )

        except Exception as e:
            result.error = str(e)
            logger.error(f"Chunk processing failed: {e}")

        result.processing_time = time.time() - start_time
        return result

    async def _run_vlm_pipeline(
        self,
        chunk: ChunkInfo,
        vlm_prompt: str,
    ) -> tuple[str, list[float]]:
        """Run VLM analysis on a chunk."""
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

            return analysis.captions, []

        finally:
            # Clean up uploaded file
            await self._gemini_file_manager.delete_file(file_result.uri)

    async def _run_cv_pipeline(
        self,
        chunk: ChunkInfo,
        text_prompt: str,
    ) -> dict[str, Any]:
        """Run CV detection on a chunk."""
        if not self._yolo_pipeline:
            return {}

        # Parse text prompt into target classes
        target_classes = [c.strip() for c in text_prompt.split(".") if c.strip()]

        # Run YOLO detection on video
        detections = self._yolo_pipeline.detect_video(
            video_path=chunk.file,
            frame_interval=5,  # Process every 5th frame
        )

        # Aggregate detection statistics
        class_counts: dict[str, int] = {}
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

    def _fuse_metadata(
        self,
        vlm_response: str,
        cv_metadata: dict[str, Any],
    ) -> str:
        """Fuse VLM response with CV metadata.

        Args:
            vlm_response: VLM-generated caption text
            cv_metadata: Pre-aggregated CV metadata dict from _run_cv_pipeline()

        Returns:
            Enriched caption with CV detection summary
        """
        if self._cv_fuser is not None:
            # Use the dict-based fusion method for pre-aggregated metadata
            return self._cv_fuser.fuse_from_dict(vlm_response, cv_metadata)
        # Fallback: append CV metadata as JSON
        cv_summary = json.dumps(cv_metadata.get("class_counts", {}))
        return f"{vlm_response}\n\n[CV Detection: {cv_summary}]"

    async def summarize(
        self,
        request_id: str,
        caption_summarization_prompt: Optional[str] = None,
        summary_aggregation_prompt: Optional[str] = None,
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

        # Collect metadata
        metadatas = [
            {
                "start_time": result.chunk.start_ntp,
                "end_time": result.chunk.end_ntp,
                "chunk_idx": result.chunk.chunkIdx,
            }
            for result in request.processed_chunks
            if result.vlm_response
        ]

        # Use summarization function
        input_data = SummarizationInput(captions=captions, metadata=metadatas)

        kwargs: dict[str, Any] = {"batch_size": batch_size}
        if caption_summarization_prompt:
            kwargs["summarization_prompt"] = caption_summarization_prompt
        if summary_aggregation_prompt:
            kwargs["aggregation_prompt"] = summary_aggregation_prompt

        result = await self._summarization_function.execute(input_data, **kwargs)

        if result.success and result.output:
            return result.output.summary
        else:
            raise RuntimeError(f"Summarization failed: {result.error}")

    async def chat(
        self,
        request_id: str,
        question: str,
        chat_history: Optional[list[dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
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
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")

        # Get context manager for this stream
        context_manager = self._get_context_manager(request.stream_id)

        # Retrieve relevant context
        context_results = await context_manager.retrieve(
            query=question,
            top_k=5,
        )

        # Build context string
        context_str = "\n\n".join(
            [
                f"[{doc['metadata'].get('start_time', '?')} - {doc['metadata'].get('end_time', '?')}]\n{doc['text']}"
                for doc in context_results
            ]
        )

        # Build prompt with context
        prompt = f"""Context from video:
{context_str}

Question: {question}

Answer based on the context above. If the information is not available, say so."""

        # Generate response
        from .models.gemini.gemini_llm import Message

        if chat_history:
            messages = [
                Message(role=msg["role"], content=msg["content"])  # type: ignore[arg-type]
                for msg in chat_history
            ]
            messages.append(Message(role="user", content=prompt))

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

    def get_request_status(self, request_id: str) -> Optional[RequestInfo]:
        """Get status of a processing request."""
        return self._requests.get(request_id)

    def get_request_progress(self, request_id: str) -> float:
        """Get progress percentage of a request."""
        request = self._requests.get(request_id)
        if request:
            return request.progress
        return 0.0

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a processing request."""
        if request_id in self._requests:
            self._cancel_flags[request_id] = True
            return True
        return False

    def cleanup_request(self, request_id: str) -> bool:
        """Clean up resources for a completed request."""
        request = self._requests.get(request_id)
        if not request:
            return False

        try:
            # Clean up context manager
            if request.stream_id in self._context_managers:
                del self._context_managers[request.stream_id]

            # Clean up asset manager
            self._asset_manager.cleanup(request.stream_id)

            # Remove from tracking
            del self._requests[request_id]
            if request_id in self._cancel_flags:
                del self._cancel_flags[request_id]

            return True
        except Exception as e:
            logger.error(f"Cleanup failed for request {request_id}: {e}")
            return False

    def list_requests(self) -> list[RequestInfo]:
        """List all requests."""
        return list(self._requests.values())

    def get_captions(self, request_id: str) -> list[str]:
        """Get all captions for a request."""
        request = self._requests.get(request_id)
        if not request:
            return []

        return [
            result.vlm_response
            for result in request.processed_chunks
            if result.vlm_response
        ]

    async def get_context_for_query(
        self,
        request_id: str,
        query: str,
        max_tokens: int = 8000,
    ) -> str:
        """Get formatted context for a query."""
        request = self._requests.get(request_id)
        if not request:
            return "Request not found."

        if self._hybrid_retriever:
            return await self._hybrid_retriever.get_context_for_query(
                query=query,
                stream_id=request.stream_id,
                max_tokens=max_tokens,
            )

        # Fallback to context manager
        context_manager = self._get_context_manager(request.stream_id)
        results = await context_manager.retrieve(query, top_k=10)

        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4

        for r in results:
            start = r["metadata"].get("start_time", "")
            end = r["metadata"].get("end_time", "")
            timestamp = f"[{start} - {end}]" if start and end else ""
            part = f"{timestamp}\n{r['text']}"

            if total_chars + len(part) > max_chars:
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n\n".join(context_parts) if context_parts else "No relevant context found."
