"""Video file splitter for chunking videos."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from src.chunk_info import ChunkInfo


class FileSplitter:
    """Splits video files into chunks for processing."""

    def __init__(
        self,
        chunk_duration: int = 60,
        chunk_overlap: int = 2,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the file splitter.

        Args:
            chunk_duration: Duration of each chunk in seconds
            chunk_overlap: Overlap between chunks in seconds
            output_dir: Directory for output chunks (uses temp dir if None)
        """
        self._chunk_duration = chunk_duration
        self._chunk_overlap = chunk_overlap
        self._output_dir = output_dir
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _get_video_duration(self, video_path: str) -> float:
        """
        Get video duration using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            raise RuntimeError(f"Failed to get video duration: {e}") from e

    def _extract_chunk(
        self,
        video_path: str,
        output_path: str,
        start_time: float,
        duration: float,
    ) -> bool:
        """
        Extract a chunk from the video using ffmpeg.

        Args:
            video_path: Source video path
            output_path: Output chunk path
            start_time: Start time in seconds
            duration: Duration in seconds

        Returns:
            True if extraction was successful
        """
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start_time),
                    "-i",
                    video_path,
                    "-t",
                    str(duration),
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    output_path,
                ],
                capture_output=True,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _split_sync(
        self,
        video_path: str,
        stream_id: str,
    ) -> list[ChunkInfo]:
        """Synchronous split for use with executor."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get video duration
        total_duration = self._get_video_duration(video_path)

        # Create output directory
        if self._output_dir:
            output_dir = Path(self._output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(tempfile.mkdtemp(prefix="vss_chunks_"))

        # Calculate chunk boundaries
        chunks = []
        chunk_idx = 0
        start_time = 0.0

        while start_time < total_duration:
            # Calculate end time
            end_time = min(start_time + self._chunk_duration, total_duration)
            duration = end_time - start_time

            # Generate output path
            output_path = output_dir / f"{path.stem}_chunk_{chunk_idx:04d}.mp4"

            # Extract chunk
            success = self._extract_chunk(
                video_path=video_path,
                output_path=str(output_path),
                start_time=start_time,
                duration=duration,
            )

            if success and output_path.exists():
                # Create chunk info
                chunk = ChunkInfo(
                    chunkIdx=chunk_idx,
                    streamId=stream_id,
                    file=str(output_path),
                    start_ntp=ChunkInfo.seconds_to_timestamp(start_time),
                    end_ntp=ChunkInfo.seconds_to_timestamp(end_time),
                    start_pts=ChunkInfo.seconds_to_pts(start_time),
                    end_pts=ChunkInfo.seconds_to_pts(end_time),
                    duration=duration,
                )
                chunks.append(chunk)

            # Move to next chunk with overlap
            start_time = end_time - self._chunk_overlap
            if start_time >= total_duration - self._chunk_overlap:
                break

            chunk_idx += 1

        return chunks

    async def split(
        self,
        video_path: str,
        stream_id: str,
    ) -> list[ChunkInfo]:
        """
        Split video into chunks.

        Args:
            video_path: Path to video file
            stream_id: Stream identifier for the video

        Returns:
            List of ChunkInfo for each chunk
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._split_sync,
            video_path,
            stream_id,
        )

    def cleanup_chunks(self, chunks: list[ChunkInfo]) -> int:
        """
        Clean up chunk files.

        Args:
            chunks: List of chunks to clean up

        Returns:
            Number of files deleted
        """
        deleted = 0
        for chunk in chunks:
            try:
                path = Path(chunk.file)
                if path.exists():
                    path.unlink()
                    deleted += 1
            except Exception:
                pass
        return deleted
