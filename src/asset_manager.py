"""Asset manager for video files and temporary storage."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional


class AssetManager:
    """Manages video assets and temporary files."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the asset manager.

        Args:
            base_dir: Base directory for storing videos
            temp_dir: Directory for temporary files
        """
        self._base_dir = Path(base_dir) if base_dir else Path.cwd() / "videos"
        self._temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "vss_poc"

        # Create directories
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        """Get base directory."""
        return self._base_dir

    @property
    def temp_dir(self) -> Path:
        """Get temp directory."""
        return self._temp_dir

    def store_video(self, video_path: str, stream_id: str) -> str:
        """
        Store an uploaded video.

        Args:
            video_path: Source video path
            stream_id: Stream identifier

        Returns:
            Path to stored video
        """
        source = Path(video_path)
        if not source.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create stream directory
        stream_dir = self._base_dir / stream_id
        stream_dir.mkdir(parents=True, exist_ok=True)

        # Copy video to storage
        dest = stream_dir / source.name
        shutil.copy2(source, dest)

        return str(dest)

    def get_chunk_path(self, stream_id: str, chunk_idx: int) -> str:
        """
        Get path for chunk storage.

        Args:
            stream_id: Stream identifier
            chunk_idx: Chunk index

        Returns:
            Path for chunk file
        """
        chunk_dir = self._temp_dir / stream_id / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        return str(chunk_dir / f"chunk_{chunk_idx:04d}.mp4")

    def get_output_path(self, stream_id: str) -> str:
        """
        Get output directory for results.

        Args:
            stream_id: Stream identifier

        Returns:
            Path to output directory
        """
        output_dir = self._base_dir / stream_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        return str(output_dir)

    def cleanup(self, stream_id: Optional[str] = None) -> int:
        """
        Clean up temporary files.

        Args:
            stream_id: Optional stream to clean up (cleans all if None)

        Returns:
            Number of files deleted
        """
        deleted = 0

        if stream_id:
            # Clean up specific stream
            stream_temp = self._temp_dir / stream_id
            if stream_temp.exists():
                deleted = self._count_files(stream_temp)
                shutil.rmtree(stream_temp, ignore_errors=True)
        else:
            # Clean up all temp files
            if self._temp_dir.exists():
                deleted = self._count_files(self._temp_dir)
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                self._temp_dir.mkdir(parents=True, exist_ok=True)

        return deleted

    def _count_files(self, directory: Path) -> int:
        """Count files in directory recursively."""
        count = 0
        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    count += 1
        except Exception:
            pass
        return count

    def get_video_path(self, stream_id: str, filename: str) -> Optional[str]:
        """
        Get path to a stored video.

        Args:
            stream_id: Stream identifier
            filename: Video filename

        Returns:
            Path to video or None if not found
        """
        video_path = self._base_dir / stream_id / filename
        if video_path.exists():
            return str(video_path)
        return None

    def list_streams(self) -> list[str]:
        """
        List all stored streams.

        Returns:
            List of stream IDs
        """
        streams = []
        try:
            for item in self._base_dir.iterdir():
                if item.is_dir():
                    streams.append(item.name)
        except Exception:
            pass
        return streams

    def delete_stream(self, stream_id: str) -> bool:
        """
        Delete a stream and all its files.

        Args:
            stream_id: Stream identifier

        Returns:
            True if deleted successfully
        """
        try:
            # Delete from base dir
            stream_dir = self._base_dir / stream_id
            if stream_dir.exists():
                shutil.rmtree(stream_dir)

            # Delete from temp dir
            temp_dir = self._temp_dir / stream_id
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            return True
        except Exception:
            return False
