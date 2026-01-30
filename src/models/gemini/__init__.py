"""Gemini model wrappers for video understanding, text generation, and embeddings."""

from src.models.gemini.gemini_embeddings import (
    BatchEmbeddingResult,
    EmbeddingResult,
    GeminiEmbeddings,
    TaskType,
)
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
from src.models.gemini.gemini_llm import (
    GenerationResult,
    GeminiLLM,
    LLMGenerationConfig,
    Message,
)
from src.models.gemini.gemini_vlm import (
    ContextLengthExceededError,
    GenerationConfig,
    GenerationError,
    GeminiVLM,
    SafetyBlockedError,
    SafetyRating,
    SafetySettings,
    TokenUsage,
    VideoAnalysisResult,
    VideoEvent,
    VLMError,
)

__all__ = [
    # File Manager
    "GeminiFileManager",
    "FileState",
    "FileUploadResult",
    "FileStatus",
    "VideoTooLongError",
    "UnsupportedFormatError",
    "UploadError",
    "ProcessingTimeoutError",
    # VLM
    "GeminiVLM",
    "VideoEvent",
    "TokenUsage",
    "SafetyRating",
    "VideoAnalysisResult",
    "GenerationConfig",
    "SafetySettings",
    "VLMError",
    "SafetyBlockedError",
    "ContextLengthExceededError",
    "GenerationError",
    # LLM
    "GeminiLLM",
    "GenerationResult",
    "Message",
    "LLMGenerationConfig",
    # Embeddings
    "GeminiEmbeddings",
    "TaskType",
    "EmbeddingResult",
    "BatchEmbeddingResult",
]
