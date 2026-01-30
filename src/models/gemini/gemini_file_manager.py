"""Gemini File Manager for uploading video files to Gemini File API."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class FileState(Enum):
    """State of a file in Gemini File API."""

    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


@dataclass
class FileUploadResult:
    """Result of uploading a file to Gemini."""

    uri: str  # File URI for generation requests (e.g., "files/abc123")
    name: str  # Full resource name
    display_name: str  # Human-readable name
    mime_type: str  # MIME type of the file
    size_bytes: int  # File size in bytes
    create_time: datetime  # When the file was created
    expiration_time: datetime  # When the file will expire (48 hours from creation)
    state: FileState  # Current processing state
    video_metadata: Optional[dict] = None  # Video-specific metadata (duration, etc.)


@dataclass
class FileStatus:
    """Status of a file in Gemini File API."""

    state: FileState
    error: Optional[str] = None  # Error message if state is FAILED


class VideoTooLongError(Exception):
    """Raised when video exceeds maximum duration."""

    pass


class UnsupportedFormatError(Exception):
    """Raised when video format is not supported."""

    pass


class UploadError(Exception):
    """Raised when upload fails."""

    pass


class ProcessingTimeoutError(Exception):
    """Raised when file processing times out."""

    pass


class GeminiFileManager:
    """Manages video file uploads to Gemini File API."""

    # Supported video formats
    SUPPORTED_FORMATS: dict[str, str] = {
        ".mp4": "video/mp4",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }

    # Maximum video duration in seconds (Gemini limit: 1 hour)
    MAX_VIDEO_DURATION: int = 3600

    # Maximum file size in bytes (2GB)
    MAX_FILE_SIZE: int = 2 * 1024 * 1024 * 1024

    def __init__(self, api_key: str, max_concurrent_uploads: int = 3) -> None:
        """
        Initialize the Gemini File Manager.

        Args:
            api_key: Google AI Studio API key
            max_concurrent_uploads: Maximum concurrent uploads
        """
        self._api_key = api_key
        self._max_concurrent_uploads = max_concurrent_uploads
        genai.configure(api_key=api_key)
        self._uploaded_files: dict[str, FileUploadResult] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_uploads)

    def _validate_video(self, video_path: str) -> tuple[str, int]:
        """
        Validate video file before upload.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (mime_type, file_size)

        Raises:
            FileNotFoundError: If file doesn't exist
            UnsupportedFormatError: If format not supported
            VideoTooLongError: If video too long (requires ffprobe)
        """
        path = Path(video_path)

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Check file extension
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported video format: {ext}. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )

        mime_type = self.SUPPORTED_FORMATS[ext]

        # Check file size
        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise VideoTooLongError(
                f"File size {file_size} bytes exceeds maximum {self.MAX_FILE_SIZE} bytes"
            )

        return mime_type, file_size

    def _upload_sync(
        self,
        video_path: str,
        display_name: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> FileUploadResult:
        """Synchronous upload for use with executor."""
        path = Path(video_path)

        # Validate and get mime type
        detected_mime_type, file_size = self._validate_video(video_path)
        mime_type = mime_type or detected_mime_type

        # Set display name
        if display_name is None:
            display_name = path.stem

        # Upload file
        try:
            file = genai.upload_file(
                path=video_path,
                display_name=display_name,
                mime_type=mime_type,
            )
        except Exception as e:
            raise UploadError(f"Failed to upload video: {e}") from e

        # Parse state
        state = FileState.PROCESSING
        if hasattr(file, "state"):
            state_str = str(file.state.name) if hasattr(file.state, "name") else str(file.state)
            try:
                state = FileState(state_str)
            except ValueError:
                state = FileState.PROCESSING

        # Create result
        result = FileUploadResult(
            uri=file.uri,
            name=file.name,
            display_name=display_name,
            mime_type=mime_type,
            size_bytes=file_size,
            create_time=datetime.now(),
            expiration_time=datetime.now(),  # Will be updated when ACTIVE
            state=state,
            video_metadata=None,
        )

        # Track uploaded file
        self._uploaded_files[file.uri] = result

        return result

    async def upload_video(
        self,
        video_path: str,
        display_name: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> FileUploadResult:
        """
        Upload a video file to Gemini File API.

        Args:
            video_path: Local path to the video file
            display_name: Optional display name for the file
            mime_type: Optional MIME type (auto-detected if not provided)

        Returns:
            FileUploadResult with file URI and metadata

        Raises:
            FileNotFoundError: If video file doesn't exist
            VideoTooLongError: If video exceeds maximum duration
            UnsupportedFormatError: If video format is not supported
            UploadError: If upload fails
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._upload_sync,
            video_path,
            display_name,
            mime_type,
        )

    def _get_file_sync(self, file_uri: str) -> FileStatus:
        """Synchronous get file status."""
        try:
            # Extract file name from URI
            file_name = file_uri.split("/")[-1] if "/" in file_uri else file_uri
            file = genai.get_file(file_name)

            state_str = str(file.state.name) if hasattr(file.state, "name") else str(file.state)
            try:
                state = FileState(state_str)
            except ValueError:
                state = FileState.PROCESSING

            error = None
            if state == FileState.FAILED:
                error = getattr(file, "error", "Unknown error")

            return FileStatus(state=state, error=error)
        except Exception as e:
            return FileStatus(state=FileState.FAILED, error=str(e))

    async def get_file_status(self, file_uri: str) -> FileStatus:
        """
        Get the current status of a file.

        Args:
            file_uri: URI of the file

        Returns:
            FileStatus with current state
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._get_file_sync, file_uri)

    async def wait_for_processing(
        self,
        file_uri: str,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> FileStatus:
        """
        Wait for an uploaded file to be processed.

        Args:
            file_uri: URI from upload response
            timeout: Maximum wait time in seconds
            poll_interval: Time between status checks in seconds

        Returns:
            FileStatus indicating ACTIVE or error

        Raises:
            ProcessingTimeoutError: If processing doesn't complete within timeout
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            status = await self.get_file_status(file_uri)

            if status.state == FileState.ACTIVE:
                return status

            if status.state == FileState.FAILED:
                return status

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                raise ProcessingTimeoutError(
                    f"File processing timed out after {timeout} seconds"
                )

            await asyncio.sleep(poll_interval)

    async def upload_and_wait(
        self,
        video_path: str,
        display_name: Optional[str] = None,
        mime_type: Optional[str] = None,
        timeout: float = 300,
    ) -> FileUploadResult:
        """
        Upload a video and wait for it to be processed.

        Convenience method that combines upload_video and wait_for_processing.

        Args:
            video_path: Local path to the video file
            display_name: Optional display name
            mime_type: Optional MIME type
            timeout: Maximum wait time for processing

        Returns:
            FileUploadResult with ACTIVE state
        """
        result = await self.upload_video(video_path, display_name, mime_type)

        status = await self.wait_for_processing(result.uri, timeout)

        if status.state == FileState.FAILED:
            raise UploadError(f"File processing failed: {status.error}")

        # Update result state
        result.state = status.state

        return result

    def _delete_file_sync(self, file_uri: str) -> bool:
        """Synchronous delete file."""
        try:
            file_name = file_uri.split("/")[-1] if "/" in file_uri else file_uri
            genai.delete_file(file_name)
            return True
        except Exception:
            return False

    async def delete_file(self, file_uri: str) -> bool:
        """
        Delete an uploaded file from Gemini.

        Args:
            file_uri: URI of the file to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(self._executor, self._delete_file_sync, file_uri)

        if success and file_uri in self._uploaded_files:
            del self._uploaded_files[file_uri]

        return success

    def _list_files_sync(self) -> list[FileUploadResult]:
        """Synchronous list files."""
        results = []
        try:
            for file in genai.list_files():
                state_str = (
                    str(file.state.name) if hasattr(file.state, "name") else str(file.state)
                )
                try:
                    state = FileState(state_str)
                except ValueError:
                    state = FileState.PROCESSING

                result = FileUploadResult(
                    uri=file.uri,
                    name=file.name,
                    display_name=getattr(file, "display_name", file.name),
                    mime_type=getattr(file, "mime_type", "video/mp4"),
                    size_bytes=getattr(file, "size_bytes", 0),
                    create_time=getattr(file, "create_time", datetime.now()),
                    expiration_time=getattr(file, "expiration_time", datetime.now()),
                    state=state,
                )
                results.append(result)
        except Exception:
            pass

        return results

    async def list_files(self) -> list[FileUploadResult]:
        """
        List all uploaded files.

        Returns:
            List of FileUploadResult for all files
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._list_files_sync)

    async def upload_multiple(
        self,
        video_paths: list[str],
        max_concurrent: Optional[int] = None,
    ) -> list[FileUploadResult | Exception]:
        """
        Upload multiple videos concurrently.

        Args:
            video_paths: List of video file paths
            max_concurrent: Maximum concurrent uploads (uses default if None)

        Returns:
            List of FileUploadResult or Exception for each video
        """
        max_concurrent = max_concurrent or self._max_concurrent_uploads
        semaphore = asyncio.Semaphore(max_concurrent)

        async def upload_with_semaphore(path: str) -> FileUploadResult:
            async with semaphore:
                return await self.upload_and_wait(path)

        tasks = [upload_with_semaphore(path) for path in video_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return list(results)

    async def cleanup_expired_files(self) -> int:
        """
        Clean up tracking of expired files.

        Returns:
            Number of files removed from tracking
        """
        now = datetime.now()
        expired = [
            uri
            for uri, result in self._uploaded_files.items()
            if result.expiration_time < now
        ]

        for uri in expired:
            del self._uploaded_files[uri]

        return len(expired)
