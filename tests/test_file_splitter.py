"""Unit tests for FileSplitter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.chunk_info import ChunkInfo
from src.file_splitter import FileSplitter


class TestFileSplitterInit:
    """Tests for FileSplitter initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        splitter = FileSplitter()

        assert splitter._chunk_duration == 60
        assert splitter._chunk_overlap == 2
        assert splitter._output_dir is None

    def test_custom_init(self) -> None:
        """Test initialization with custom parameters."""
        splitter = FileSplitter(
            chunk_duration=120,
            chunk_overlap=5,
            output_dir="/tmp/chunks",
        )

        assert splitter._chunk_duration == 120
        assert splitter._chunk_overlap == 5
        assert splitter._output_dir == "/tmp/chunks"


class TestFileSplitterGetDuration:
    """Tests for _get_video_duration method."""

    @patch("subprocess.run")
    def test_get_video_duration_success(self, mock_run: MagicMock) -> None:
        """Test successful duration retrieval."""
        mock_run.return_value = MagicMock(stdout="120.5\n")

        splitter = FileSplitter()
        duration = splitter._get_video_duration("/path/to/video.mp4")

        assert duration == 120.5
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "ffprobe" in call_args
        assert "/path/to/video.mp4" in call_args

    @patch("subprocess.run")
    def test_get_video_duration_failure(self, mock_run: MagicMock) -> None:
        """Test duration retrieval failure."""
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "ffprobe")

        splitter = FileSplitter()

        with pytest.raises(RuntimeError, match="Failed to get video duration"):
            splitter._get_video_duration("/path/to/video.mp4")

    @patch("subprocess.run")
    def test_get_video_duration_invalid_output(self, mock_run: MagicMock) -> None:
        """Test duration retrieval with invalid output."""
        mock_run.return_value = MagicMock(stdout="invalid")

        splitter = FileSplitter()

        with pytest.raises(RuntimeError, match="Failed to get video duration"):
            splitter._get_video_duration("/path/to/video.mp4")


class TestFileSplitterExtractChunk:
    """Tests for _extract_chunk method."""

    @patch("subprocess.run")
    def test_extract_chunk_success(self, mock_run: MagicMock) -> None:
        """Test successful chunk extraction."""
        mock_run.return_value = MagicMock()

        splitter = FileSplitter()
        result = splitter._extract_chunk(
            video_path="/path/to/video.mp4",
            output_path="/path/to/chunk.mp4",
            start_time=60.0,
            duration=60.0,
        )

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args
        assert "-ss" in call_args
        assert "60.0" in call_args

    @patch("subprocess.run")
    def test_extract_chunk_failure(self, mock_run: MagicMock) -> None:
        """Test chunk extraction failure."""
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "ffmpeg")

        splitter = FileSplitter()
        result = splitter._extract_chunk(
            video_path="/path/to/video.mp4",
            output_path="/path/to/chunk.mp4",
            start_time=60.0,
            duration=60.0,
        )

        assert result is False


class TestFileSplitterSplitSync:
    """Tests for _split_sync method."""

    def test_split_sync_file_not_found(self) -> None:
        """Test split with non-existent file."""
        splitter = FileSplitter()

        with pytest.raises(FileNotFoundError, match="Video file not found"):
            splitter._split_sync("/nonexistent/video.mp4", "stream_123")

    @patch.object(FileSplitter, "_extract_chunk")
    @patch.object(FileSplitter, "_get_video_duration")
    def test_split_sync_single_chunk(
        self,
        mock_duration: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        """Test splitting video that fits in single chunk."""
        mock_duration.return_value = 30.0  # 30 seconds
        mock_extract.return_value = True

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
            Path(video_path).touch()

            try:
                with tempfile.TemporaryDirectory() as output_dir:
                    splitter = FileSplitter(
                        chunk_duration=60,
                        chunk_overlap=2,
                        output_dir=output_dir,
                    )

                    # Create fake output file
                    expected_output = Path(output_dir) / f"{Path(video_path).stem}_chunk_0000.mp4"

                    def create_output(*args, **kwargs):
                        expected_output.touch()
                        return True

                    mock_extract.side_effect = create_output

                    chunks = splitter._split_sync(video_path, "stream_123")

                    assert len(chunks) == 1
                    assert chunks[0].chunkIdx == 0
                    assert chunks[0].streamId == "stream_123"
                    assert chunks[0].start_ntp == "00:00:00"
                    assert chunks[0].end_ntp == "00:00:30"
            finally:
                Path(video_path).unlink(missing_ok=True)

    @patch.object(FileSplitter, "_extract_chunk")
    @patch.object(FileSplitter, "_get_video_duration")
    def test_split_sync_multiple_chunks(
        self,
        mock_duration: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        """Test splitting video into multiple chunks."""
        mock_duration.return_value = 150.0  # 2.5 minutes

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
            Path(video_path).touch()

            try:
                with tempfile.TemporaryDirectory() as output_dir:
                    splitter = FileSplitter(
                        chunk_duration=60,
                        chunk_overlap=2,
                        output_dir=output_dir,
                    )

                    chunk_idx = [0]

                    def create_output(video_path, output_path, start_time, duration):
                        Path(output_path).touch()
                        chunk_idx[0] += 1
                        return True

                    mock_extract.side_effect = create_output

                    chunks = splitter._split_sync(video_path, "stream_123")

                    # With 150s duration, 60s chunks, 2s overlap:
                    # Chunk 0: 0-60s
                    # Chunk 1: 58-118s
                    # Chunk 2: 116-150s
                    assert len(chunks) == 3
                    assert all(c.streamId == "stream_123" for c in chunks)
                    assert chunks[0].chunkIdx == 0
                    assert chunks[1].chunkIdx == 1
                    assert chunks[2].chunkIdx == 2
            finally:
                Path(video_path).unlink(missing_ok=True)

    @patch.object(FileSplitter, "_extract_chunk")
    @patch.object(FileSplitter, "_get_video_duration")
    def test_split_sync_with_temp_dir(
        self,
        mock_duration: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        """Test splitting without explicit output directory."""
        mock_duration.return_value = 30.0

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
            Path(video_path).touch()

            try:
                splitter = FileSplitter(
                    chunk_duration=60,
                    chunk_overlap=2,
                    output_dir=None,  # Use temp dir
                )

                def create_output(video_path, output_path, start_time, duration):
                    Path(output_path).touch()
                    return True

                mock_extract.side_effect = create_output

                chunks = splitter._split_sync(video_path, "stream_123")

                assert len(chunks) == 1
                # Output should be in temp directory
                assert "vss_chunks_" in chunks[0].file
            finally:
                Path(video_path).unlink(missing_ok=True)
                # Clean up temp chunks
                for chunk in chunks:
                    Path(chunk.file).unlink(missing_ok=True)

    @patch.object(FileSplitter, "_extract_chunk")
    @patch.object(FileSplitter, "_get_video_duration")
    def test_split_sync_extraction_failure(
        self,
        mock_duration: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        """Test handling extraction failure."""
        mock_duration.return_value = 30.0
        mock_extract.return_value = False  # Extraction fails

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
            Path(video_path).touch()

            try:
                with tempfile.TemporaryDirectory() as output_dir:
                    splitter = FileSplitter(
                        chunk_duration=60,
                        output_dir=output_dir,
                    )

                    chunks = splitter._split_sync(video_path, "stream_123")

                    # No chunks created due to extraction failure
                    assert len(chunks) == 0
            finally:
                Path(video_path).unlink(missing_ok=True)


class TestFileSplitterSplit:
    """Tests for async split method."""

    @pytest.mark.asyncio
    @patch.object(FileSplitter, "_split_sync")
    async def test_split_async(self, mock_split_sync: MagicMock) -> None:
        """Test async split method."""
        expected_chunks = [
            ChunkInfo(
                chunkIdx=0,
                streamId="stream_123",
                file="/path/to/chunk_0000.mp4",
                start_ntp="00:00:00",
                end_ntp="00:01:00",
            )
        ]
        mock_split_sync.return_value = expected_chunks

        splitter = FileSplitter()
        chunks = await splitter.split("/path/to/video.mp4", "stream_123")

        assert chunks == expected_chunks
        mock_split_sync.assert_called_once_with("/path/to/video.mp4", "stream_123")


class TestFileSplitterCleanup:
    """Tests for cleanup_chunks method."""

    def test_cleanup_chunks_success(self) -> None:
        """Test successful chunk cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake chunk files
            chunk_files = []
            for i in range(3):
                chunk_path = Path(temp_dir) / f"chunk_{i:04d}.mp4"
                chunk_path.touch()
                chunk_files.append(chunk_path)

            chunks = [
                ChunkInfo(
                    chunkIdx=i,
                    streamId="stream_123",
                    file=str(chunk_files[i]),
                    start_ntp="00:00:00",
                    end_ntp="00:01:00",
                )
                for i in range(3)
            ]

            splitter = FileSplitter()
            deleted = splitter.cleanup_chunks(chunks)

            assert deleted == 3
            for chunk_file in chunk_files:
                assert not chunk_file.exists()

    def test_cleanup_chunks_nonexistent(self) -> None:
        """Test cleanup with non-existent files."""
        chunks = [
            ChunkInfo(
                chunkIdx=0,
                streamId="stream_123",
                file="/nonexistent/chunk.mp4",
                start_ntp="00:00:00",
                end_ntp="00:01:00",
            )
        ]

        splitter = FileSplitter()
        deleted = splitter.cleanup_chunks(chunks)

        assert deleted == 0

    def test_cleanup_chunks_partial(self) -> None:
        """Test cleanup with some existing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only one chunk file
            existing_chunk = Path(temp_dir) / "chunk_0000.mp4"
            existing_chunk.touch()

            chunks = [
                ChunkInfo(
                    chunkIdx=0,
                    streamId="stream_123",
                    file=str(existing_chunk),
                    start_ntp="00:00:00",
                    end_ntp="00:01:00",
                ),
                ChunkInfo(
                    chunkIdx=1,
                    streamId="stream_123",
                    file="/nonexistent/chunk.mp4",
                    start_ntp="00:01:00",
                    end_ntp="00:02:00",
                ),
            ]

            splitter = FileSplitter()
            deleted = splitter.cleanup_chunks(chunks)

            assert deleted == 1
            assert not existing_chunk.exists()

    def test_cleanup_chunks_empty_list(self) -> None:
        """Test cleanup with empty list."""
        splitter = FileSplitter()
        deleted = splitter.cleanup_chunks([])

        assert deleted == 0


class TestFileSplitterChunkCalculation:
    """Tests for chunk boundary calculation."""

    @patch.object(FileSplitter, "_extract_chunk")
    @patch.object(FileSplitter, "_get_video_duration")
    def test_chunk_boundaries_with_overlap(
        self,
        mock_duration: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        """Test that chunk boundaries account for overlap."""
        mock_duration.return_value = 180.0  # 3 minutes

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
            Path(video_path).touch()

            try:
                with tempfile.TemporaryDirectory() as output_dir:
                    splitter = FileSplitter(
                        chunk_duration=60,
                        chunk_overlap=5,
                        output_dir=output_dir,
                    )

                    extract_calls = []

                    def track_extract(video_path, output_path, start_time, duration):
                        extract_calls.append((start_time, duration))
                        Path(output_path).touch()
                        return True

                    mock_extract.side_effect = track_extract

                    chunks = splitter._split_sync(video_path, "stream_123")

                    # With 180s duration, 60s chunks, 5s overlap:
                    # Chunk 0: 0-60s (start=0)
                    # Chunk 1: 55-115s (start=55)
                    # Chunk 2: 110-170s (start=110)
                    # Chunk 3: 165-180s (start=165)
                    assert len(chunks) >= 3

                    # Verify start times account for overlap
                    assert extract_calls[0][0] == 0.0  # First chunk starts at 0
                    if len(extract_calls) > 1:
                        assert extract_calls[1][0] == 55.0  # Second chunk starts at 60-5=55
            finally:
                Path(video_path).unlink(missing_ok=True)

    @patch.object(FileSplitter, "_extract_chunk")
    @patch.object(FileSplitter, "_get_video_duration")
    def test_chunk_pts_values(
        self,
        mock_duration: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        """Test that PTS values are correctly calculated."""
        mock_duration.return_value = 60.0

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
            Path(video_path).touch()

            try:
                with tempfile.TemporaryDirectory() as output_dir:
                    splitter = FileSplitter(
                        chunk_duration=60,
                        output_dir=output_dir,
                    )

                    def create_output(video_path, output_path, start_time, duration):
                        Path(output_path).touch()
                        return True

                    mock_extract.side_effect = create_output

                    chunks = splitter._split_sync(video_path, "stream_123")

                    assert len(chunks) == 1
                    # PTS should be in nanoseconds
                    assert chunks[0].start_pts == 0
                    assert chunks[0].end_pts == 60_000_000_000  # 60 seconds in nanoseconds
            finally:
                Path(video_path).unlink(missing_ok=True)
