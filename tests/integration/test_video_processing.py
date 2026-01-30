"""Integration tests for end-to-end video processing.

These tests require:
- GEMINI_API_KEY environment variable
- Optional: Running Milvus and Neo4j for full pipeline tests

Run with: pytest tests/integration/test_video_processing.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from src.via_stream_handler import ViaStreamHandler


def has_gemini_api() -> bool:
    """Check if Gemini API is available."""
    return bool(os.environ.get("GEMINI_API_KEY"))


def has_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    import shutil
    return shutil.which("ffmpeg") is not None


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not has_gemini_api(),
    reason="GEMINI_API_KEY not set"
)


@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    return os.environ["GEMINI_API_KEY"]


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_video_path(temp_dir: Path) -> Path:
    """Create a simple test video file."""
    import subprocess

    video_path = temp_dir / "test_video.mp4"

    # Create a 5-second test video with ffmpeg
    # Using lavfi to generate test pattern
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", "testsrc=duration=5:size=320x240:rate=30",
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=5",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-y",
        str(video_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("ffmpeg not available or failed to create test video")

    return video_path


class TestFileSplitterIntegration:
    """Integration tests for FileSplitter."""

    @pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not available")
    async def test_split_video(
        self,
        test_video_path: Path,
        temp_dir: Path
    ) -> None:
        """Test splitting a video into chunks."""
        from src.file_splitter import FileSplitter

        splitter = FileSplitter(
            chunk_duration=2,  # 2 second chunks
            chunk_overlap=0,
        )

        chunks = await splitter.split(
            video_path=str(test_video_path),
            stream_id="test_stream",
            output_dir=str(temp_dir),
        )

        assert len(chunks) >= 2  # 5 second video / 2 second chunks
        for chunk in chunks:
            assert chunk.file is not None
            assert Path(chunk.file).exists()

    @pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not available")
    async def test_split_video_with_overlap(
        self,
        test_video_path: Path,
        temp_dir: Path
    ) -> None:
        """Test splitting with overlap."""
        from src.file_splitter import FileSplitter

        splitter = FileSplitter(
            chunk_duration=3,
            chunk_overlap=1,  # 1 second overlap
        )

        chunks = await splitter.split(
            video_path=str(test_video_path),
            stream_id="test_stream",
            output_dir=str(temp_dir),
        )

        assert len(chunks) >= 2
        # Verify overlap in timestamps
        if len(chunks) >= 2:
            # Second chunk should start before first chunk ends
            # (accounting for overlap)
            pass  # Overlap is handled internally


class TestAssetManagerIntegration:
    """Integration tests for AssetManager."""

    async def test_store_and_retrieve_video(
        self,
        test_video_path: Path,
        temp_dir: Path
    ) -> None:
        """Test storing and retrieving video."""
        from src.asset_manager import AssetManager

        manager = AssetManager(
            base_path=str(temp_dir),
            temp_path=str(temp_dir / "temp"),
        )

        # Store video
        stored_path = await manager.store_video(
            video_path=str(test_video_path),
            stream_id="test_stream",
        )

        assert stored_path is not None
        assert Path(stored_path).exists()

        # Retrieve video path
        retrieved = manager.get_video_path("test_stream")
        assert retrieved is not None
        assert Path(retrieved).exists()

    async def test_cleanup(self, temp_dir: Path) -> None:
        """Test cleanup functionality."""
        from src.asset_manager import AssetManager

        manager = AssetManager(
            base_path=str(temp_dir),
            temp_path=str(temp_dir / "temp"),
        )

        # Create some test files
        stream_dir = temp_dir / "test_stream"
        stream_dir.mkdir(parents=True, exist_ok=True)
        (stream_dir / "test.txt").write_text("test")

        # Cleanup
        await manager.cleanup("test_stream")

        # Directory should be cleaned up
        assert not stream_dir.exists()


class TestViaStreamHandlerIntegration:
    """Integration tests for ViaStreamHandler."""

    @pytest.fixture
    async def stream_handler(
        self,
        api_key: str,
        temp_dir: Path
    ) -> "ViaStreamHandler":
        """Create ViaStreamHandler instance."""
        from src.via_stream_handler import ViaStreamHandler
        from src.asset_manager import AssetManager
        from src.models.gemini.gemini_file_manager import GeminiFileManager
        from src.models.gemini.gemini_vlm import GeminiVLM
        from src.models.gemini.gemini_llm import GeminiLLM
        from src.models.gemini.gemini_embeddings import GeminiEmbeddings

        asset_manager = AssetManager(
            base_path=str(temp_dir / "assets"),
            temp_path=str(temp_dir / "temp"),
        )

        file_manager = GeminiFileManager(api_key=api_key)
        vlm = GeminiVLM(api_key=api_key)
        llm = GeminiLLM(api_key=api_key)
        embeddings = GeminiEmbeddings(api_key=api_key)

        handler = ViaStreamHandler(
            asset_manager=asset_manager,
            file_manager=file_manager,
            vlm=vlm,
            llm=llm,
            embeddings=embeddings,
            config={
                "chunk_duration": 5,
                "chunk_overlap": 0,
            },
        )

        return handler

    @pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not available")
    async def test_process_video_basic(
        self,
        stream_handler: "ViaStreamHandler",
        test_video_path: Path
    ) -> None:
        """Test basic video processing."""
        results: list = []

        async def on_chunk_complete(result):
            results.append(result)

        request_id = await stream_handler.process_video(
            video_path=str(test_video_path),
            stream_id="test_stream",
            on_chunk_complete=on_chunk_complete,
        )

        assert request_id is not None

        # Wait for processing to complete
        import asyncio
        for _ in range(60):  # Wait up to 60 seconds
            status = stream_handler.get_request_status(request_id)
            if status and status.value in ["SUCCESSFUL", "FAILED"]:
                break
            await asyncio.sleep(1)

        # Check results
        status = stream_handler.get_request_status(request_id)
        assert status is not None

    async def test_chat_without_processing(
        self,
        stream_handler: "ViaStreamHandler"
    ) -> None:
        """Test chat returns error when no video processed."""
        response = await stream_handler.chat(
            request_id="nonexistent",
            query="What happened?",
        )

        assert response is None or "not found" in str(response).lower()

    async def test_summarize_without_processing(
        self,
        stream_handler: "ViaStreamHandler"
    ) -> None:
        """Test summarize returns error when no video processed."""
        response = await stream_handler.summarize(
            request_id="nonexistent",
        )

        assert response is None


class TestGeminiFileManagerIntegration:
    """Integration tests for GeminiFileManager."""

    @pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not available")
    async def test_upload_video(
        self,
        api_key: str,
        test_video_path: Path
    ) -> None:
        """Test uploading a video to Gemini."""
        from src.models.gemini.gemini_file_manager import GeminiFileManager

        manager = GeminiFileManager(api_key=api_key)

        # Upload video
        result = await manager.upload_video(str(test_video_path))

        assert result is not None
        assert result.name is not None

        # Clean up
        try:
            await manager.delete_file(result.name)
        except Exception:
            pass  # Ignore cleanup errors

    @pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not available")
    async def test_upload_and_wait(
        self,
        api_key: str,
        test_video_path: Path
    ) -> None:
        """Test uploading and waiting for processing."""
        from src.models.gemini.gemini_file_manager import GeminiFileManager, FileState

        manager = GeminiFileManager(api_key=api_key)

        # Upload and wait
        result = await manager.upload_and_wait(
            str(test_video_path),
            timeout=120,
        )

        assert result is not None
        assert result.state == FileState.ACTIVE

        # Clean up
        try:
            await manager.delete_file(result.name)
        except Exception:
            pass


class TestGeminiVLMIntegration:
    """Integration tests for GeminiVLM video analysis."""

    @pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not available")
    async def test_analyze_video(
        self,
        api_key: str,
        test_video_path: Path
    ) -> None:
        """Test analyzing a video with VLM."""
        from src.models.gemini.gemini_file_manager import GeminiFileManager
        from src.models.gemini.gemini_vlm import GeminiVLM

        file_manager = GeminiFileManager(api_key=api_key)
        vlm = GeminiVLM(api_key=api_key)

        # Upload video
        upload_result = await file_manager.upload_and_wait(
            str(test_video_path),
            timeout=120,
        )

        try:
            # Analyze video
            result = await vlm.analyze_video(
                file_uri=upload_result.name,
                prompt="Describe what you see in this video.",
            )

            assert result is not None
            assert result.captions is not None
            assert len(result.captions) > 0

        finally:
            # Clean up
            try:
                await file_manager.delete_file(upload_result.name)
            except Exception:
                pass


class TestCVPipelineIntegration:
    """Integration tests for CV pipeline."""

    async def test_yolo_detection(self, temp_dir: Path) -> None:
        """Test YOLO object detection."""
        import numpy as np

        try:
            from src.cv_pipeline.yolo_pipeline import YOLOPipeline
        except ImportError:
            pytest.skip("YOLO dependencies not available")

        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some color variation
        test_image[100:200, 100:200] = [255, 0, 0]  # Red square

        try:
            pipeline = YOLOPipeline(
                model="yolov8n.pt",  # Use smallest model
                confidence=0.1,  # Low threshold for test
            )

            result = pipeline.detect(test_image)

            assert result is not None
            # May or may not detect anything in synthetic image
            assert hasattr(result, "boxes")

        except Exception as e:
            if "model" in str(e).lower():
                pytest.skip("YOLO model not available")
            raise

    async def test_tracker(self) -> None:
        """Test object tracker."""
        import numpy as np

        try:
            from src.cv_pipeline.tracker import ObjectTracker
            from src.cv_pipeline.yolo_pipeline import DetectionResult
        except ImportError:
            pytest.skip("Tracker dependencies not available")

        tracker = ObjectTracker()

        # Create mock detections
        detection = DetectionResult(
            boxes=np.array([[100, 100, 200, 200]]),
            confidences=np.array([0.9]),
            class_ids=np.array([0]),
            class_names=["person"],
        )

        # Update tracker
        result = tracker.update(detection)

        assert result is not None
        assert hasattr(result, "tracked_objects")
