# Gemini File Manager Specification

## Overview

The `GeminiFileManager` is responsible for uploading video files to the Gemini File API and managing their lifecycle. This replaces the local frame extraction approach used in the original VSS engine with native video upload to Gemini.

## Gap Analysis

### Original Implementation
- `src/vss-engine/src/vlm_pipeline/vlm_pipeline.py` - `DecoderProcess` extracts frames from video chunks
- `src/vss-engine/src/vlm_pipeline/embedding_helper.py` - Saves frame embeddings locally
- Frame-by-frame processing with local GPU inference

### PoC Requirement
- Upload entire video chunks to Gemini File API
- Wait for processing to complete (ACTIVE status)
- Return file URI for use in generation requests
- Handle file expiration (48 hours) and cleanup

## Component Location

```
./src/models/gemini/gemini_file_manager.py
```

## Dependencies

```python
# From requirements.txt
google-generativeai>=0.8.0
aiofiles>=23.0.0
tenacity>=8.0.0
```

## Data Classes

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class FileState(Enum):
    """State of a file in Gemini File API."""
    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


@dataclass
class FileUploadResult:
    """Result of uploading a file to Gemini."""
    uri: str                      # File URI for generation requests (e.g., "files/abc123")
    name: str                     # Full resource name
    display_name: str             # Human-readable name
    mime_type: str                # MIME type of the file
    size_bytes: int               # File size in bytes
    create_time: datetime         # When the file was created
    expiration_time: datetime     # When the file will expire (48 hours from creation)
    state: FileState              # Current processing state
    video_metadata: Optional[dict] = None  # Video-specific metadata (duration, etc.)


@dataclass
class FileStatus:
    """Status of a file in Gemini File API."""
    state: FileState
    error: Optional[str] = None   # Error message if state is FAILED


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
```

## Class Interface

```python
import asyncio
import os
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential


class GeminiFileManager:
    """Manages video file uploads to Gemini File API."""

    # Supported video formats
    SUPPORTED_FORMATS = {
        ".mp4": "video/mp4",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }

    # Maximum video duration in seconds (Gemini limit: 1 hour)
    MAX_VIDEO_DURATION = 3600

    # Maximum file size in bytes (2GB)
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024

    def __init__(self, api_key: str):
        """
        Initialize the Gemini File Manager.

        Args:
            api_key: Google AI Studio API key
        """
        self._api_key = api_key
        genai.configure(api_key=api_key)
        self._uploaded_files: dict[str, FileUploadResult] = {}

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
        pass

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
        pass

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
        pass

    async def delete_file(self, file_uri: str) -> bool:
        """
        Delete an uploaded file from Gemini.

        Args:
            file_uri: URI of the file to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        pass

    async def list_files(self) -> list[FileUploadResult]:
        """
        List all uploaded files.

        Returns:
            List of FileUploadResult for all files
        """
        pass

    async def get_file_status(self, file_uri: str) -> FileStatus:
        """
        Get the current status of a file.

        Args:
            file_uri: URI of the file

        Returns:
            FileStatus with current state
        """
        pass

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
        pass

    async def cleanup_expired_files(self) -> int:
        """
        Clean up tracking of expired files.

        Returns:
            Number of files removed from tracking
        """
        pass
```

## Implementation Notes

### Upload Flow

1. **Validate video file**
   - Check file exists
   - Verify supported format
   - Check file size < 2GB
   - Optionally check duration < 1 hour (requires ffprobe)

2. **Upload to Gemini**
   - Use `genai.upload_file()` for synchronous upload
   - Or implement chunked upload for large files

3. **Wait for processing**
   - Poll `genai.get_file()` until state is ACTIVE
   - Handle FAILED state with error message
   - Implement timeout

4. **Track uploaded files**
   - Store file URIs and expiration times
   - Clean up expired files from tracking

### Error Handling

```python
RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 1.0,
    "max_delay": 60.0,
    "exponential_base": 2,
    "retryable_errors": [
        "QUOTA_EXCEEDED",
        "INTERNAL_ERROR",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
    ]
}
```

### Concurrent Upload Support

```python
async def upload_multiple(
    self,
    video_paths: list[str],
    max_concurrent: int = 3,
) -> list[FileUploadResult]:
    """Upload multiple videos concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def upload_with_semaphore(path: str) -> FileUploadResult:
        async with semaphore:
            return await self.upload_and_wait(path)

    tasks = [upload_with_semaphore(path) for path in video_paths]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

## Configuration

```yaml
# config/config.yaml
gemini:
  file_manager:
    max_concurrent_uploads: 3
    upload_timeout: 300
    poll_interval: 5
    auto_cleanup: true
```

## Environment Variables

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MAX_CONCURRENT_UPLOADS=3
GEMINI_UPLOAD_TIMEOUT=300
```

## Testing

```python
# tests/test_gemini_file_manager.py

import pytest
from poc.src.models.gemini.gemini_file_manager import (
    GeminiFileManager,
    FileState,
    UnsupportedFormatError,
)


@pytest.fixture
def file_manager():
    return GeminiFileManager(api_key="test_key")


class TestGeminiFileManager:
    async def test_validate_supported_format(self, file_manager):
        """Test that supported formats are validated correctly."""
        pass

    async def test_validate_unsupported_format(self, file_manager):
        """Test that unsupported formats raise UnsupportedFormatError."""
        pass

    async def test_upload_and_wait(self, file_manager):
        """Test full upload and wait flow."""
        pass

    async def test_delete_file(self, file_manager):
        """Test file deletion."""
        pass

    async def test_concurrent_uploads(self, file_manager):
        """Test concurrent upload with semaphore limiting."""
        pass
```

## Integration with VLM Pipeline

The `GeminiFileManager` will be used by `VideoUploader` in the VLM pipeline:

```python
# ./src/vlm_pipeline/video_uploader.py

class VideoUploader:
    def __init__(self, file_manager: GeminiFileManager):
        self._file_manager = file_manager

    async def upload_chunk(self, chunk: ChunkInfo) -> str:
        """Upload a video chunk and return the file URI."""
        result = await self._file_manager.upload_and_wait(
            video_path=chunk.file,
            display_name=f"chunk_{chunk.chunkIdx}",
        )
        return result.uri
```
