"""Video Search and Summarization (VSS) PoC.

A video analysis pipeline using Google Gemini APIs (VLM, LLM, embeddings)
and YOLOv26 for object detection. Processes video files to generate
timestamped captions, extract entities/events, and enable semantic search
via hybrid RAG (Milvus vector DB + Neo4j graph DB).
"""

__version__ = "0.1.0"

# Core components (lazy imports to avoid cv2 dependency at module load)
from .chunk_info import ChunkInfo
from .file_splitter import FileSplitter
from .asset_manager import AssetManager

__all__ = [
    # Version
    "__version__",
    # Core
    "ChunkInfo",
    "FileSplitter",
    "AssetManager",
]


def __getattr__(name: str):
    """Lazy import for optional components with heavy dependencies."""
    if name in (
        "ViaStreamHandler",
        "RequestStatus",
        "RequestInfo",
        "ChunkResult",
        "ProcessingResponse",
        "VlmRequestParams",
    ):
        from . import via_stream_handler

        return getattr(via_stream_handler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
