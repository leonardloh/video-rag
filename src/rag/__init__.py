"""RAG (Retrieval-Augmented Generation) components for video context management."""

from .context_store import (
    ContextStore,
    ContextWindow,
    Document,
    InMemoryContextStore,
    RetrievalResult,
)
from .context_manager import ContextManager, ContextManagerConfig
from .milvus_store import MilvusContextStore
from .hybrid_retrieval import (
    HybridConfig,
    HybridRetriever,
    RetrievalMode,
    RetrievalResult as HybridRetrievalResult,
)
from .functions import (
    CompositeFunction,
    FunctionConfig,
    FunctionResult,
    FunctionStatus,
    RAGFunction,
    SummarizationFunction,
    SummarizationInput,
    SummarizationOutput,
)

__all__ = [
    # Context Store
    "ContextStore",
    "InMemoryContextStore",
    "MilvusContextStore",
    "Document",
    "RetrievalResult",
    "ContextWindow",
    # Context Manager
    "ContextManager",
    "ContextManagerConfig",
    # Hybrid Retrieval
    "HybridRetriever",
    "HybridConfig",
    "RetrievalMode",
    "HybridRetrievalResult",
    # RAG Functions
    "RAGFunction",
    "FunctionConfig",
    "FunctionResult",
    "FunctionStatus",
    "CompositeFunction",
    "SummarizationFunction",
    "SummarizationInput",
    "SummarizationOutput",
]
