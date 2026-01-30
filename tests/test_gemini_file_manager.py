"""Tests for Gemini File Manager."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.gemini.gemini_file_manager import (
    FileState,
    FileStatus,
    FileUploadResult,
    GeminiFileManager,
    ProcessingTimeoutError,
    UnsupportedFormatError,
    UploadError,
    VideoTooLongError,
)


class TestGeminiFileManager:
    """Tests for GeminiFileManager class."""

    @pytest.fixture
    def file_manager(self) -> GeminiFileManager:
        """Create a file manager instance."""
        with patch("google.generativeai.configure"):
            return GeminiFileManager(api_key="test_key")

    def test_init(self, file_manager: GeminiFileManager) -> None:
        """Test initialization."""
        assert file_manager._api_key == "test_key"
        assert file_manager._max_concurrent_uploads == 3
        assert file_manager._uploaded_files == {}

    def test_supported_formats(self, file_manager: GeminiFileManager) -> None:
        """Test supported video formats."""
        assert ".mp4" in file_manager.SUPPORTED_FORMATS
        assert ".mov" in file_manager.SUPPORTED_FORMATS
        assert ".webm" in file_manager.SUPPORTED_FORMATS
        assert ".txt" not in file_manager.SUPPORTED_FORMATS

    def test_validate_video_file_not_found(self, file_manager: GeminiFileManager) -> None:
        """Test validation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            file_manager._validate_video("/nonexistent/video.mp4")

    def test_validate_video_unsupported_format(self, file_manager: GeminiFileManager) -> None:
        """Test validation with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            with pytest.raises(UnsupportedFormatError):
                file_manager._validate_video(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validate_video_supported_format(self, file_manager: GeminiFileManager) -> None:
        """Test validation with supported format."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"test video content")
            temp_path = f.name

        try:
            mime_type, file_size = file_manager._validate_video(temp_path)
            assert mime_type == "video/mp4"
            assert file_size > 0
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_upload_video(self, file_manager: GeminiFileManager) -> None:
        """Test video upload."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"test video content")
            temp_path = f.name

        try:
            mock_file = MagicMock()
            mock_file.uri = "files/test123"
            mock_file.name = "test123"
            mock_file.state = MagicMock(name="PROCESSING")

            with patch("google.generativeai.upload_file", return_value=mock_file):
                result = await file_manager.upload_video(temp_path, display_name="test")

            assert result.uri == "files/test123"
            assert result.display_name == "test"
            assert result.mime_type == "video/mp4"
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_get_file_status(self, file_manager: GeminiFileManager) -> None:
        """Test getting file status."""
        mock_file = MagicMock()
        mock_state = MagicMock()
        mock_state.name = "ACTIVE"
        mock_file.state = mock_state

        with patch("google.generativeai.get_file", return_value=mock_file):
            status = await file_manager.get_file_status("files/test123")

        assert status.state == FileState.ACTIVE
        assert status.error is None

    @pytest.mark.asyncio
    async def test_wait_for_processing_success(self, file_manager: GeminiFileManager) -> None:
        """Test waiting for processing to complete."""
        mock_file = MagicMock()
        mock_state = MagicMock()
        mock_state.name = "ACTIVE"
        mock_file.state = mock_state

        with patch("google.generativeai.get_file", return_value=mock_file):
            status = await file_manager.wait_for_processing(
                "files/test123", timeout=10, poll_interval=0.1
            )

        assert status.state == FileState.ACTIVE

    @pytest.mark.asyncio
    async def test_wait_for_processing_timeout(self, file_manager: GeminiFileManager) -> None:
        """Test timeout during processing wait."""
        mock_file = MagicMock()
        mock_state = MagicMock()
        mock_state.name = "PROCESSING"
        mock_file.state = mock_state

        with patch("google.generativeai.get_file", return_value=mock_file):
            with pytest.raises(ProcessingTimeoutError):
                await file_manager.wait_for_processing(
                    "files/test123", timeout=0.2, poll_interval=0.1
                )

    @pytest.mark.asyncio
    async def test_delete_file(self, file_manager: GeminiFileManager) -> None:
        """Test file deletion."""
        with patch("google.generativeai.delete_file") as mock_delete:
            result = await file_manager.delete_file("files/test123")

        assert result is True
        mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_files(self, file_manager: GeminiFileManager) -> None:
        """Test listing files."""
        mock_file = MagicMock()
        mock_file.uri = "files/test123"
        mock_file.name = "test123"
        mock_state = MagicMock()
        mock_state.name = "ACTIVE"
        mock_file.state = mock_state

        with patch("google.generativeai.list_files", return_value=[mock_file]):
            files = await file_manager.list_files()

        assert len(files) == 1
        assert files[0].uri == "files/test123"

    @pytest.mark.asyncio
    async def test_upload_multiple(self, file_manager: GeminiFileManager) -> None:
        """Test uploading multiple videos."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            paths = []
            for i in range(2):  # Reduced to 2 files for faster test
                path = Path(temp_dir) / f"video{i}.mp4"
                path.write_bytes(b"test video content")
                paths.append(str(path))

            mock_file = MagicMock()
            mock_file.uri = "files/test123"
            mock_file.name = "test123"
            mock_state = MagicMock()
            mock_state.name = "ACTIVE"
            mock_file.state = mock_state

            with patch("google.generativeai.upload_file", return_value=mock_file):
                with patch("google.generativeai.get_file", return_value=mock_file):
                    results = await file_manager.upload_multiple(paths, max_concurrent=2)

            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_cleanup_expired_files(self, file_manager: GeminiFileManager) -> None:
        """Test cleanup of expired files."""
        # Add a file to tracking
        from datetime import datetime, timedelta

        file_manager._uploaded_files["files/test123"] = FileUploadResult(
            uri="files/test123",
            name="test123",
            display_name="test",
            mime_type="video/mp4",
            size_bytes=100,
            create_time=datetime.now() - timedelta(hours=50),
            expiration_time=datetime.now() - timedelta(hours=2),
            state=FileState.ACTIVE,
        )

        count = await file_manager.cleanup_expired_files()
        assert count == 1
        assert "files/test123" not in file_manager._uploaded_files


class TestFileState:
    """Tests for FileState enum."""

    def test_file_states(self) -> None:
        """Test file state values."""
        assert FileState.PROCESSING.value == "PROCESSING"
        assert FileState.ACTIVE.value == "ACTIVE"
        assert FileState.FAILED.value == "FAILED"


class TestFileUploadResult:
    """Tests for FileUploadResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a file upload result."""
        from datetime import datetime

        result = FileUploadResult(
            uri="files/test123",
            name="test123",
            display_name="test video",
            mime_type="video/mp4",
            size_bytes=1024,
            create_time=datetime.now(),
            expiration_time=datetime.now(),
            state=FileState.ACTIVE,
        )

        assert result.uri == "files/test123"
        assert result.display_name == "test video"
        assert result.state == FileState.ACTIVE
