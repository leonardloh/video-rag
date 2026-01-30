"""Unit tests for AssetManager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.asset_manager import AssetManager


class TestAssetManagerInit:
    """Tests for AssetManager initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=temp_dir,
                temp_dir=f"{temp_dir}/temp",
            )

            assert manager.base_dir == Path(temp_dir)
            assert manager.temp_dir == Path(f"{temp_dir}/temp")
            assert manager.base_dir.exists()
            assert manager.temp_dir.exists()

    def test_init_creates_directories(self) -> None:
        """Test that init creates required directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = f"{temp_dir}/videos"
            temp_storage = f"{temp_dir}/temp"

            # Directories don't exist yet
            assert not Path(base_dir).exists()
            assert not Path(temp_storage).exists()

            manager = AssetManager(
                base_dir=base_dir,
                temp_dir=temp_storage,
            )

            # Directories should be created
            assert Path(base_dir).exists()
            assert Path(temp_storage).exists()


class TestAssetManagerStoreVideo:
    """Tests for store_video method."""

    def test_store_video_success(self) -> None:
        """Test successful video storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            # Create a fake video file
            source_video = Path(temp_dir) / "test_video.mp4"
            source_video.write_text("fake video content")

            stored_path = manager.store_video(str(source_video), "stream_123")

            assert Path(stored_path).exists()
            assert "stream_123" in stored_path
            assert Path(stored_path).read_text() == "fake video content"

    def test_store_video_file_not_found(self) -> None:
        """Test storing non-existent video."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            with pytest.raises(FileNotFoundError, match="Video file not found"):
                manager.store_video("/nonexistent/video.mp4", "stream_123")

    def test_store_video_creates_stream_dir(self) -> None:
        """Test that storing video creates stream directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            source_video = Path(temp_dir) / "test_video.mp4"
            source_video.touch()

            manager.store_video(str(source_video), "new_stream")

            stream_dir = Path(f"{temp_dir}/videos/new_stream")
            assert stream_dir.exists()
            assert stream_dir.is_dir()


class TestAssetManagerGetChunkPath:
    """Tests for get_chunk_path method."""

    def test_get_chunk_path(self) -> None:
        """Test getting chunk path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            chunk_path = manager.get_chunk_path("stream_123", 0)

            assert "stream_123" in chunk_path
            assert "chunk_0000.mp4" in chunk_path
            # Directory should be created
            assert Path(chunk_path).parent.exists()

    def test_get_chunk_path_formatting(self) -> None:
        """Test chunk path index formatting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            # Test various indices
            assert "chunk_0000.mp4" in manager.get_chunk_path("stream", 0)
            assert "chunk_0001.mp4" in manager.get_chunk_path("stream", 1)
            assert "chunk_0099.mp4" in manager.get_chunk_path("stream", 99)
            assert "chunk_1234.mp4" in manager.get_chunk_path("stream", 1234)


class TestAssetManagerGetOutputPath:
    """Tests for get_output_path method."""

    def test_get_output_path(self) -> None:
        """Test getting output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            output_path = manager.get_output_path("stream_123")

            assert "stream_123" in output_path
            assert "output" in output_path
            assert Path(output_path).exists()
            assert Path(output_path).is_dir()


class TestAssetManagerCleanup:
    """Tests for cleanup method."""

    def test_cleanup_specific_stream(self) -> None:
        """Test cleaning up specific stream."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            # Create temp files for two streams
            stream1_dir = manager.temp_dir / "stream_1" / "chunks"
            stream1_dir.mkdir(parents=True)
            (stream1_dir / "chunk_0000.mp4").touch()
            (stream1_dir / "chunk_0001.mp4").touch()

            stream2_dir = manager.temp_dir / "stream_2" / "chunks"
            stream2_dir.mkdir(parents=True)
            (stream2_dir / "chunk_0000.mp4").touch()

            # Clean up only stream_1
            deleted = manager.cleanup("stream_1")

            assert deleted == 2
            assert not (manager.temp_dir / "stream_1").exists()
            assert (manager.temp_dir / "stream_2").exists()

    def test_cleanup_all_streams(self) -> None:
        """Test cleaning up all streams."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            # Create temp files
            stream_dir = manager.temp_dir / "stream_1" / "chunks"
            stream_dir.mkdir(parents=True)
            (stream_dir / "chunk_0000.mp4").touch()
            (stream_dir / "chunk_0001.mp4").touch()

            deleted = manager.cleanup()

            assert deleted == 2
            # Temp dir should be recreated but empty
            assert manager.temp_dir.exists()
            assert list(manager.temp_dir.iterdir()) == []

    def test_cleanup_nonexistent_stream(self) -> None:
        """Test cleaning up non-existent stream."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            deleted = manager.cleanup("nonexistent_stream")

            assert deleted == 0


class TestAssetManagerGetVideoPath:
    """Tests for get_video_path method."""

    def test_get_video_path_exists(self) -> None:
        """Test getting existing video path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            # Store a video
            source_video = Path(temp_dir) / "test_video.mp4"
            source_video.touch()
            manager.store_video(str(source_video), "stream_123")

            # Get the path
            video_path = manager.get_video_path("stream_123", "test_video.mp4")

            assert video_path is not None
            assert Path(video_path).exists()

    def test_get_video_path_not_exists(self) -> None:
        """Test getting non-existent video path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            video_path = manager.get_video_path("stream_123", "nonexistent.mp4")

            assert video_path is None


class TestAssetManagerListStreams:
    """Tests for list_streams method."""

    def test_list_streams(self) -> None:
        """Test listing streams."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            # Create some stream directories
            (manager.base_dir / "stream_1").mkdir()
            (manager.base_dir / "stream_2").mkdir()
            (manager.base_dir / "stream_3").mkdir()

            streams = manager.list_streams()

            assert len(streams) == 3
            assert "stream_1" in streams
            assert "stream_2" in streams
            assert "stream_3" in streams

    def test_list_streams_empty(self) -> None:
        """Test listing streams when none exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            streams = manager.list_streams()

            assert streams == []


class TestAssetManagerDeleteStream:
    """Tests for delete_stream method."""

    def test_delete_stream_success(self) -> None:
        """Test successful stream deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            # Create stream directories
            stream_base = manager.base_dir / "stream_123"
            stream_base.mkdir(parents=True)
            (stream_base / "video.mp4").touch()

            stream_temp = manager.temp_dir / "stream_123"
            stream_temp.mkdir(parents=True)
            (stream_temp / "chunk.mp4").touch()

            result = manager.delete_stream("stream_123")

            assert result is True
            assert not stream_base.exists()
            assert not stream_temp.exists()

    def test_delete_stream_nonexistent(self) -> None:
        """Test deleting non-existent stream."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            result = manager.delete_stream("nonexistent_stream")

            # Should return True even if stream doesn't exist
            assert result is True

    def test_delete_stream_partial(self) -> None:
        """Test deleting stream with only base dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            # Create only base dir
            stream_base = manager.base_dir / "stream_123"
            stream_base.mkdir(parents=True)
            (stream_base / "video.mp4").touch()

            result = manager.delete_stream("stream_123")

            assert result is True
            assert not stream_base.exists()


class TestAssetManagerCountFiles:
    """Tests for _count_files method."""

    def test_count_files(self) -> None:
        """Test counting files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            # Create nested files
            test_dir = Path(temp_dir) / "test"
            test_dir.mkdir()
            (test_dir / "file1.txt").touch()
            (test_dir / "file2.txt").touch()

            subdir = test_dir / "subdir"
            subdir.mkdir()
            (subdir / "file3.txt").touch()

            count = manager._count_files(test_dir)

            assert count == 3

    def test_count_files_empty(self) -> None:
        """Test counting files in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            empty_dir = Path(temp_dir) / "empty"
            empty_dir.mkdir()

            count = manager._count_files(empty_dir)

            assert count == 0

    def test_count_files_nonexistent(self) -> None:
        """Test counting files in non-existent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AssetManager(
                base_dir=f"{temp_dir}/videos",
                temp_dir=f"{temp_dir}/temp",
            )

            count = manager._count_files(Path("/nonexistent/path"))

            assert count == 0
