"""Unit tests for ViaStreamHandler."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.asset_manager import AssetManager
from src.chunk_info import ChunkInfo
from src.models.gemini.gemini_embeddings import GeminiEmbeddings
from src.models.gemini.gemini_file_manager import (
    FileState,
    FileUploadResult,
    GeminiFileManager,
)
from src.models.gemini.gemini_llm import (
    GenerationResult,
    GeminiLLM,
    TokenUsage as LLMTokenUsage,
)
from src.models.gemini.gemini_vlm import (
    GeminiVLM,
    SafetyRating,
    TokenUsage,
    VideoAnalysisResult,
    VideoEvent,
)
from src.via_stream_handler import (
    ChunkResult,
    ProcessingResponse,
    RequestInfo,
    RequestStatus,
    ViaStreamHandler,
    VlmRequestParams,
)


@pytest.fixture
def mock_asset_manager() -> MagicMock:
    """Create mock asset manager."""
    manager = MagicMock(spec=AssetManager)
    manager.get_output_path.return_value = "/tmp/output"
    manager.cleanup.return_value = None
    return manager


@pytest.fixture
def mock_gemini_file_manager() -> MagicMock:
    """Create mock Gemini file manager."""
    manager = MagicMock(spec=GeminiFileManager)
    manager.upload_and_wait = AsyncMock(
        return_value=FileUploadResult(
            uri="files/test-file-123",
            name="test-file",
            display_name="chunk_0",
            mime_type="video/mp4",
            size_bytes=1000,
            create_time=datetime.now(),
            expiration_time=datetime.now(),
            state=FileState.ACTIVE,
        )
    )
    manager.delete_file = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_gemini_vlm() -> MagicMock:
    """Create mock Gemini VLM."""
    vlm = MagicMock(spec=GeminiVLM)
    vlm.analyze_video = AsyncMock(
        return_value=VideoAnalysisResult(
            captions="<00:00:00> A person walks into the room.\n<00:00:05> They sit down at a desk.",
            parsed_events=[
                VideoEvent(
                    start_time="00:00:00",
                    end_time="00:00:05",
                    description="A person walks into the room.",
                )
            ],
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            safety_ratings=[],
        )
    )
    return vlm


@pytest.fixture
def mock_gemini_llm() -> MagicMock:
    """Create mock Gemini LLM."""
    llm = MagicMock(spec=GeminiLLM)
    llm.generate = AsyncMock(
        return_value=GenerationResult(
            text="This is a summary of the video.",
            usage=LLMTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            finish_reason="STOP",
            safety_ratings=[],
        )
    )
    llm.chat = AsyncMock(
        return_value=GenerationResult(
            text="The person in the video is sitting at a desk.",
            usage=LLMTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            finish_reason="STOP",
            safety_ratings=[],
        )
    )
    llm.summarize_captions = AsyncMock(return_value="Summary of captions")
    llm.aggregate_summaries = AsyncMock(return_value="Final aggregated summary")
    return llm


@pytest.fixture
def mock_gemini_embeddings() -> MagicMock:
    """Create mock Gemini embeddings."""
    embeddings = MagicMock(spec=GeminiEmbeddings)
    embeddings.embed_text = AsyncMock(
        return_value=MagicMock(embedding=[0.1] * 768)
    )
    embeddings.embed_query = AsyncMock(return_value=[0.1] * 768)
    embeddings.embed_document = AsyncMock(return_value=[0.1] * 768)
    return embeddings


@pytest.fixture
def stream_handler(
    mock_asset_manager: MagicMock,
    mock_gemini_file_manager: MagicMock,
    mock_gemini_vlm: MagicMock,
    mock_gemini_llm: MagicMock,
    mock_gemini_embeddings: MagicMock,
) -> ViaStreamHandler:
    """Create ViaStreamHandler with mocked dependencies."""
    return ViaStreamHandler(
        asset_manager=mock_asset_manager,
        gemini_file_manager=mock_gemini_file_manager,
        gemini_vlm=mock_gemini_vlm,
        gemini_llm=mock_gemini_llm,
        gemini_embeddings=mock_gemini_embeddings,
        config={"rag": {"max_context_tokens": 100000, "batch_size": 6, "top_k": 5}},
    )


class TestRequestStatus:
    """Tests for RequestStatus enum."""

    def test_status_values(self) -> None:
        """Test all status values exist."""
        assert RequestStatus.QUEUED.value == "queued"
        assert RequestStatus.PROCESSING.value == "processing"
        assert RequestStatus.SUCCESSFUL.value == "successful"
        assert RequestStatus.FAILED.value == "failed"
        assert RequestStatus.CANCELLED.value == "cancelled"


class TestVlmRequestParams:
    """Tests for VlmRequestParams dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        params = VlmRequestParams()
        assert params.vlm_prompt is None
        assert params.vlm_generation_config is None

    def test_custom_values(self) -> None:
        """Test custom values."""
        params = VlmRequestParams(
            vlm_prompt="Describe the video",
            vlm_generation_config={"temperature": 0.5},
        )
        assert params.vlm_prompt == "Describe the video"
        assert params.vlm_generation_config == {"temperature": 0.5}


class TestProcessingResponse:
    """Tests for ProcessingResponse dataclass."""

    def test_create_response(self) -> None:
        """Test creating a response."""
        response = ProcessingResponse(
            start_timestamp="00:00:00",
            end_timestamp="00:01:00",
            response="A person walks into the room.",
            reasoning_description="VLM analysis",
        )
        assert response.start_timestamp == "00:00:00"
        assert response.end_timestamp == "00:01:00"
        assert response.response == "A person walks into the room."
        assert response.reasoning_description == "VLM analysis"

    def test_default_reasoning(self) -> None:
        """Test default reasoning description."""
        response = ProcessingResponse(
            start_timestamp="00:00:00",
            end_timestamp="00:01:00",
            response="Test response",
        )
        assert response.reasoning_description == ""


class TestChunkResult:
    """Tests for ChunkResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a chunk result."""
        chunk = ChunkInfo(
            chunkIdx=0,
            streamId="stream-1",
            file="/tmp/chunk_0.mp4",
            start_ntp="00:00:00",
            end_ntp="00:01:00",
        )
        result = ChunkResult(
            chunk=chunk,
            vlm_response="A person walks into the room.",
            cv_metadata={"class_counts": {"person": 1}},
            frame_times=[0.0, 1.0, 2.0],
            processing_time=5.5,
        )
        assert result.chunk == chunk
        assert result.vlm_response == "A person walks into the room."
        assert result.cv_metadata == {"class_counts": {"person": 1}}
        assert result.frame_times == [0.0, 1.0, 2.0]
        assert result.error is None
        assert result.processing_time == 5.5

    def test_default_values(self) -> None:
        """Test default values."""
        chunk = ChunkInfo(
            chunkIdx=0,
            streamId="stream-1",
            file="/tmp/chunk_0.mp4",
            start_ntp="00:00:00",
            end_ntp="00:01:00",
        )
        result = ChunkResult(chunk=chunk)
        assert result.vlm_response is None
        assert result.cv_metadata is None
        assert result.frame_times == []
        assert result.error is None
        assert result.processing_time == 0.0


class TestRequestInfo:
    """Tests for RequestInfo dataclass."""

    def test_create_request_info(self) -> None:
        """Test creating request info."""
        info = RequestInfo(
            request_id="req-123",
            stream_id="stream-456",
            file="/path/to/video.mp4",
        )
        assert info.request_id == "req-123"
        assert info.stream_id == "stream-456"
        assert info.file == "/path/to/video.mp4"
        assert info.status == RequestStatus.QUEUED
        assert info.chunk_count == 0
        assert info.processed_chunks == []
        assert info.progress == 0.0
        assert info.responses == []

    def test_default_processing_params(self) -> None:
        """Test default processing parameters."""
        info = RequestInfo(
            request_id="req-123",
            stream_id="stream-456",
            file="/path/to/video.mp4",
        )
        assert info.vlm_request_params is None
        assert info.enable_cv_pipeline is False
        assert info.enable_chat is True
        assert info.summarize is True

    def test_summarization_params(self) -> None:
        """Test summarization parameters."""
        info = RequestInfo(
            request_id="req-123",
            stream_id="stream-456",
            file="/path/to/video.mp4",
            caption_summarization_prompt="Summarize this",
            summary_aggregation_prompt="Aggregate these",
            summarize_batch_size=10,
        )
        assert info.caption_summarization_prompt == "Summarize this"
        assert info.summary_aggregation_prompt == "Aggregate these"
        assert info.summarize_batch_size == 10


class TestViaStreamHandlerInit:
    """Tests for ViaStreamHandler initialization."""

    def test_init_basic(self, stream_handler: ViaStreamHandler) -> None:
        """Test basic initialization."""
        assert stream_handler._asset_manager is not None
        assert stream_handler._gemini_file_manager is not None
        assert stream_handler._gemini_vlm is not None
        assert stream_handler._gemini_llm is not None
        assert stream_handler._gemini_embeddings is not None
        assert stream_handler._requests == {}
        assert stream_handler._cancel_flags == {}
        assert stream_handler._context_managers == {}

    def test_init_without_databases(self, stream_handler: ViaStreamHandler) -> None:
        """Test initialization without database clients."""
        assert stream_handler._milvus_client is None
        assert stream_handler._neo4j_client is None

    def test_init_with_config(self, stream_handler: ViaStreamHandler) -> None:
        """Test initialization with config."""
        assert stream_handler._config["rag"]["max_context_tokens"] == 100000
        assert stream_handler._config["rag"]["batch_size"] == 6

    def test_init_creates_summarization_function(
        self, stream_handler: ViaStreamHandler
    ) -> None:
        """Test that summarization function is created."""
        assert stream_handler._summarization_function is not None


class TestViaStreamHandlerGetContextManager:
    """Tests for _get_context_manager method."""

    def test_creates_new_context_manager(
        self, stream_handler: ViaStreamHandler
    ) -> None:
        """Test creating a new context manager."""
        cm = stream_handler._get_context_manager("stream-1")
        assert cm is not None
        assert "stream-1" in stream_handler._context_managers

    def test_returns_existing_context_manager(
        self, stream_handler: ViaStreamHandler
    ) -> None:
        """Test returning existing context manager."""
        cm1 = stream_handler._get_context_manager("stream-1")
        cm2 = stream_handler._get_context_manager("stream-1")
        assert cm1 is cm2

    def test_different_streams_get_different_managers(
        self, stream_handler: ViaStreamHandler
    ) -> None:
        """Test different streams get different context managers."""
        cm1 = stream_handler._get_context_manager("stream-1")
        cm2 = stream_handler._get_context_manager("stream-2")
        assert cm1 is not cm2


class TestViaStreamHandlerProcessVideo:
    """Tests for process_video method."""

    @pytest.mark.asyncio
    async def test_process_video_creates_request(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test that process_video creates a request."""
        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = []

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            assert result.request_id is not None
            assert result.stream_id is not None
            assert result.file == "/path/to/video.mp4"
            assert result.status == RequestStatus.SUCCESSFUL

    @pytest.mark.asyncio
    async def test_process_video_with_chunks(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test processing video with chunks."""
        chunks = [
            ChunkInfo(
                chunkIdx=0,
                streamId="stream-1",
                file="/tmp/chunk_0.mp4",
                start_ntp="00:00:00",
                end_ntp="00:01:00",
            ),
            ChunkInfo(
                chunkIdx=1,
                streamId="stream-1",
                file="/tmp/chunk_1.mp4",
                start_ntp="00:01:00",
                end_ntp="00:02:00",
            ),
        ]

        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = chunks

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            assert result.chunk_count == 2
            assert len(result.processed_chunks) == 2
            assert result.progress == 100.0

    @pytest.mark.asyncio
    async def test_process_video_calls_callbacks(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test that callbacks are called during processing."""
        chunks = [
            ChunkInfo(
                chunkIdx=0,
                streamId="stream-1",
                file="/tmp/chunk_0.mp4",
                start_ntp="00:00:00",
                end_ntp="00:01:00",
            ),
        ]

        chunk_callback = MagicMock()
        progress_callback = MagicMock()

        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = chunks

            await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
                on_chunk_complete=chunk_callback,
                on_progress=progress_callback,
            )

            chunk_callback.assert_called_once()
            progress_callback.assert_called_once_with(100.0)

    @pytest.mark.asyncio
    async def test_process_video_handles_failure(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test handling of processing failure."""
        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.side_effect = Exception("Split failed")

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            assert result.status == RequestStatus.FAILED
            assert "Split failed" in result.error_message


class TestViaStreamHandlerSplitVideo:
    """Tests for _split_video method."""

    @pytest.mark.asyncio
    async def test_split_video_calls_splitter(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test that split_video calls FileSplitter."""
        with patch("src.via_stream_handler.FileSplitter") as MockSplitter:
            mock_instance = MagicMock()
            mock_instance.split = AsyncMock(return_value=[])
            MockSplitter.return_value = mock_instance

            await stream_handler._split_video(
                file_path="/path/to/video.mp4",
                stream_id="stream-1",
                chunk_duration=60,
                chunk_overlap=2,
            )

            MockSplitter.assert_called_once()
            mock_instance.split.assert_called_once()


class TestViaStreamHandlerProcessChunk:
    """Tests for _process_chunk method."""

    @pytest.mark.asyncio
    async def test_process_chunk_vlm_only(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test processing chunk with VLM only."""
        chunk = ChunkInfo(
            chunkIdx=0,
            streamId="stream-1",
            file="/tmp/chunk_0.mp4",
            start_ntp="00:00:00",
            end_ntp="00:01:00",
        )

        result = await stream_handler._process_chunk(
            chunk=chunk,
            vlm_prompt="Describe the video",
            enable_cv_pipeline=False,
            cv_text_prompt="",
        )

        assert result.chunk == chunk
        assert result.vlm_response is not None
        assert result.error is None
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_process_chunk_handles_vlm_error(
        self,
        stream_handler: ViaStreamHandler,
        mock_gemini_file_manager: MagicMock,
    ) -> None:
        """Test handling VLM error during chunk processing."""
        chunk = ChunkInfo(
            chunkIdx=0,
            streamId="stream-1",
            file="/tmp/chunk_0.mp4",
            start_ntp="00:00:00",
            end_ntp="00:01:00",
        )

        mock_gemini_file_manager.upload_and_wait.side_effect = Exception(
            "Upload failed"
        )

        result = await stream_handler._process_chunk(
            chunk=chunk,
            vlm_prompt="Describe the video",
            enable_cv_pipeline=False,
            cv_text_prompt="",
        )

        assert result.error is not None
        assert "Upload failed" in result.error


class TestViaStreamHandlerRunVlmPipeline:
    """Tests for _run_vlm_pipeline method."""

    @pytest.mark.asyncio
    async def test_run_vlm_pipeline_uploads_and_analyzes(
        self,
        stream_handler: ViaStreamHandler,
        mock_gemini_file_manager: MagicMock,
        mock_gemini_vlm: MagicMock,
    ) -> None:
        """Test VLM pipeline uploads and analyzes video."""
        chunk = ChunkInfo(
            chunkIdx=0,
            streamId="stream-1",
            file="/tmp/chunk_0.mp4",
            start_ntp="00:00:00",
            end_ntp="00:01:00",
        )

        captions, frame_times = await stream_handler._run_vlm_pipeline(
            chunk=chunk,
            vlm_prompt="Describe the video",
        )

        mock_gemini_file_manager.upload_and_wait.assert_called_once()
        mock_gemini_vlm.analyze_video.assert_called_once()
        mock_gemini_file_manager.delete_file.assert_called_once()
        assert captions is not None

    @pytest.mark.asyncio
    async def test_run_vlm_pipeline_cleans_up_on_error(
        self,
        stream_handler: ViaStreamHandler,
        mock_gemini_file_manager: MagicMock,
        mock_gemini_vlm: MagicMock,
    ) -> None:
        """Test VLM pipeline cleans up even on error."""
        chunk = ChunkInfo(
            chunkIdx=0,
            streamId="stream-1",
            file="/tmp/chunk_0.mp4",
            start_ntp="00:00:00",
            end_ntp="00:01:00",
        )

        mock_gemini_vlm.analyze_video.side_effect = Exception("Analysis failed")

        with pytest.raises(Exception, match="Analysis failed"):
            await stream_handler._run_vlm_pipeline(
                chunk=chunk,
                vlm_prompt="Describe the video",
            )

        # Cleanup should still be called
        mock_gemini_file_manager.delete_file.assert_called_once()


class TestViaStreamHandlerFuseMetadata:
    """Tests for _fuse_metadata method."""

    def test_fuse_metadata_without_fuser(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test fusing metadata without CV fuser."""
        stream_handler._cv_fuser = None

        result = stream_handler._fuse_metadata(
            vlm_response="A person walks into the room.",
            cv_metadata={"class_counts": {"person": 1}},
        )

        assert "A person walks into the room." in result
        assert "CV Detection" in result
        assert "person" in result


class TestViaStreamHandlerSummarize:
    """Tests for summarize method."""

    @pytest.mark.asyncio
    async def test_summarize_success(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test successful summarization."""
        # Create a request with processed chunks
        chunks = [
            ChunkInfo(
                chunkIdx=0,
                streamId="stream-1",
                file="/tmp/chunk_0.mp4",
                start_ntp="00:00:00",
                end_ntp="00:01:00",
            ),
        ]

        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = chunks

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            # Now summarize
            with patch.object(
                stream_handler._summarization_function,
                "execute",
                new_callable=AsyncMock,
            ) as mock_execute:
                from src.rag.functions.base import FunctionResult
                from src.rag.functions.summarization import SummarizationOutput

                mock_execute.return_value = FunctionResult(
                    success=True,
                    output=SummarizationOutput(
                        summary="This is a summary.",
                        batch_summaries=["Summary 1"],
                    ),
                )

                summary = await stream_handler.summarize(result.request_id)
                assert summary == "This is a summary."

    @pytest.mark.asyncio
    async def test_summarize_request_not_found(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test summarization with invalid request ID."""
        with pytest.raises(ValueError, match="not found"):
            await stream_handler.summarize("invalid-request-id")

    @pytest.mark.asyncio
    async def test_summarize_no_captions(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test summarization with no captions."""
        # Create a request with no processed chunks
        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = []

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            summary = await stream_handler.summarize(result.request_id)
            assert "No captions available" in summary


class TestViaStreamHandlerChat:
    """Tests for chat method."""

    @pytest.mark.asyncio
    async def test_chat_success(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test successful chat."""
        chunks = [
            ChunkInfo(
                chunkIdx=0,
                streamId="stream-1",
                file="/tmp/chunk_0.mp4",
                start_ntp="00:00:00",
                end_ntp="00:01:00",
            ),
        ]

        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = chunks

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            answer = await stream_handler.chat(
                request_id=result.request_id,
                question="What is happening in the video?",
            )

            assert answer is not None
            assert len(answer) > 0

    @pytest.mark.asyncio
    async def test_chat_request_not_found(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test chat with invalid request ID."""
        with pytest.raises(ValueError, match="not found"):
            await stream_handler.chat(
                request_id="invalid-request-id",
                question="What is happening?",
            )

    @pytest.mark.asyncio
    async def test_chat_with_history(
        self,
        stream_handler: ViaStreamHandler,
        mock_gemini_llm: MagicMock,
    ) -> None:
        """Test chat with conversation history."""
        chunks = [
            ChunkInfo(
                chunkIdx=0,
                streamId="stream-1",
                file="/tmp/chunk_0.mp4",
                start_ntp="00:00:00",
                end_ntp="00:01:00",
            ),
        ]

        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = chunks

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            chat_history = [
                {"role": "user", "content": "What is in the video?"},
                {"role": "assistant", "content": "A person is walking."},
            ]

            await stream_handler.chat(
                request_id=result.request_id,
                question="What are they doing now?",
                chat_history=chat_history,
            )

            mock_gemini_llm.chat.assert_called_once()


class TestViaStreamHandlerRequestManagement:
    """Tests for request management methods."""

    @pytest.mark.asyncio
    async def test_get_request_status(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test getting request status."""
        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = []

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            status = stream_handler.get_request_status(result.request_id)
            assert status is not None
            assert status.request_id == result.request_id

    def test_get_request_status_not_found(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test getting status for non-existent request."""
        status = stream_handler.get_request_status("invalid-id")
        assert status is None

    @pytest.mark.asyncio
    async def test_get_request_progress(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test getting request progress."""
        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = []

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            progress = stream_handler.get_request_progress(result.request_id)
            assert progress == 0.0  # No chunks processed

    def test_get_request_progress_not_found(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test getting progress for non-existent request."""
        progress = stream_handler.get_request_progress("invalid-id")
        assert progress == 0.0

    @pytest.mark.asyncio
    async def test_cancel_request(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test cancelling a request."""
        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = []

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            cancelled = await stream_handler.cancel_request(result.request_id)
            assert cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_request_not_found(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test cancelling non-existent request."""
        cancelled = await stream_handler.cancel_request("invalid-id")
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_cleanup_request(
        self,
        stream_handler: ViaStreamHandler,
        mock_asset_manager: MagicMock,
    ) -> None:
        """Test cleaning up a request."""
        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = []

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            cleaned = stream_handler.cleanup_request(result.request_id)
            assert cleaned is True
            mock_asset_manager.cleanup.assert_called_once()

    def test_cleanup_request_not_found(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test cleaning up non-existent request."""
        cleaned = stream_handler.cleanup_request("invalid-id")
        assert cleaned is False

    @pytest.mark.asyncio
    async def test_list_requests(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test listing all requests."""
        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = []

            await stream_handler.process_video(
                file_path="/path/to/video1.mp4",
                vlm_prompt="Describe the video",
            )
            await stream_handler.process_video(
                file_path="/path/to/video2.mp4",
                vlm_prompt="Describe the video",
            )

            requests = stream_handler.list_requests()
            assert len(requests) == 2


class TestViaStreamHandlerGetCaptions:
    """Tests for get_captions method."""

    @pytest.mark.asyncio
    async def test_get_captions(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test getting captions for a request."""
        chunks = [
            ChunkInfo(
                chunkIdx=0,
                streamId="stream-1",
                file="/tmp/chunk_0.mp4",
                start_ntp="00:00:00",
                end_ntp="00:01:00",
            ),
        ]

        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = chunks

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            captions = stream_handler.get_captions(result.request_id)
            assert len(captions) == 1
            assert captions[0] is not None

    def test_get_captions_not_found(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test getting captions for non-existent request."""
        captions = stream_handler.get_captions("invalid-id")
        assert captions == []


class TestViaStreamHandlerGetContextForQuery:
    """Tests for get_context_for_query method."""

    @pytest.mark.asyncio
    async def test_get_context_for_query(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test getting context for a query."""
        chunks = [
            ChunkInfo(
                chunkIdx=0,
                streamId="stream-1",
                file="/tmp/chunk_0.mp4",
                start_ntp="00:00:00",
                end_ntp="00:01:00",
            ),
        ]

        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = chunks

            result = await stream_handler.process_video(
                file_path="/path/to/video.mp4",
                vlm_prompt="Describe the video",
            )

            context = await stream_handler.get_context_for_query(
                request_id=result.request_id,
                query="What is happening?",
            )

            assert context is not None

    @pytest.mark.asyncio
    async def test_get_context_for_query_not_found(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test getting context for non-existent request."""
        context = await stream_handler.get_context_for_query(
            request_id="invalid-id",
            query="What is happening?",
        )
        assert "not found" in context.lower()


class TestViaStreamHandlerCancellation:
    """Tests for request cancellation."""

    @pytest.mark.asyncio
    async def test_cancellation_during_processing(
        self,
        stream_handler: ViaStreamHandler,
    ) -> None:
        """Test cancellation during video processing."""
        chunks = [
            ChunkInfo(
                chunkIdx=i,
                streamId="stream-1",
                file=f"/tmp/chunk_{i}.mp4",
                start_ntp=f"00:0{i}:00",
                end_ntp=f"00:0{i+1}:00",
            )
            for i in range(5)
        ]

        async def slow_process_chunk(*args: Any, **kwargs: Any) -> ChunkResult:
            await asyncio.sleep(0.1)
            return ChunkResult(
                chunk=args[0] if args else kwargs.get("chunk"),
                vlm_response="Test response",
            )

        with patch.object(
            stream_handler, "_split_video", new_callable=AsyncMock
        ) as mock_split:
            mock_split.return_value = chunks

            with patch.object(
                stream_handler, "_process_chunk", side_effect=slow_process_chunk
            ):
                # Start processing in background
                task = asyncio.create_task(
                    stream_handler.process_video(
                        file_path="/path/to/video.mp4",
                        vlm_prompt="Describe the video",
                    )
                )

                # Wait a bit then cancel
                await asyncio.sleep(0.05)
                request_id = list(stream_handler._requests.keys())[0]
                await stream_handler.cancel_request(request_id)

                result = await task
                assert result.status == RequestStatus.CANCELLED
                assert "cancelled" in result.error_message.lower()


class TestViaStreamHandlerWithDatabases:
    """Tests for ViaStreamHandler with database clients."""

    def test_init_with_milvus_only(
        self,
        mock_asset_manager: MagicMock,
        mock_gemini_file_manager: MagicMock,
        mock_gemini_vlm: MagicMock,
        mock_gemini_llm: MagicMock,
        mock_gemini_embeddings: MagicMock,
    ) -> None:
        """Test initialization with Milvus client only."""
        mock_milvus = MagicMock()

        handler = ViaStreamHandler(
            asset_manager=mock_asset_manager,
            gemini_file_manager=mock_gemini_file_manager,
            gemini_vlm=mock_gemini_vlm,
            gemini_llm=mock_gemini_llm,
            gemini_embeddings=mock_gemini_embeddings,
            milvus_client=mock_milvus,
        )

        assert handler._milvus_client is mock_milvus
        assert handler._hybrid_retriever is not None

    def test_init_with_both_databases(
        self,
        mock_asset_manager: MagicMock,
        mock_gemini_file_manager: MagicMock,
        mock_gemini_vlm: MagicMock,
        mock_gemini_llm: MagicMock,
        mock_gemini_embeddings: MagicMock,
    ) -> None:
        """Test initialization with both database clients."""
        mock_milvus = MagicMock()
        mock_neo4j = MagicMock()

        handler = ViaStreamHandler(
            asset_manager=mock_asset_manager,
            gemini_file_manager=mock_gemini_file_manager,
            gemini_vlm=mock_gemini_vlm,
            gemini_llm=mock_gemini_llm,
            gemini_embeddings=mock_gemini_embeddings,
            milvus_client=mock_milvus,
            neo4j_client=mock_neo4j,
        )

        assert handler._milvus_client is mock_milvus
        assert handler._neo4j_client is mock_neo4j
        assert handler._hybrid_retriever is not None
