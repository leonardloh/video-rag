"""Integration tests for the full RAG pipeline.

These tests require:
- GEMINI_API_KEY environment variable
- Running Milvus instance (for vector search tests)
- Running Neo4j instance (for graph search tests)

Run with: pytest tests/integration/test_rag_pipeline.py -v
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from src.rag.context_manager import ContextManager
    from src.rag.hybrid_retrieval import HybridRetriever


def has_gemini_api() -> bool:
    """Check if Gemini API is available."""
    return bool(os.environ.get("GEMINI_API_KEY"))


def is_milvus_available() -> bool:
    """Check if Milvus is available."""
    try:
        from pymilvus import connections
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = int(os.environ.get("MILVUS_PORT", "19530"))
        connections.connect(alias="rag_test_check", host=host, port=port)
        connections.disconnect(alias="rag_test_check")
        return True
    except Exception:
        return False


def is_neo4j_available() -> bool:
    """Check if Neo4j is available."""
    try:
        from neo4j import GraphDatabase
        host = os.environ.get("NEO4J_HOST", "localhost")
        port = int(os.environ.get("NEO4J_BOLT_PORT", "7687"))
        username = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password")

        driver = GraphDatabase.driver(
            f"bolt://{host}:{port}",
            auth=(username, password)
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not has_gemini_api(),
    reason="GEMINI_API_KEY not set"
)


@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    return os.environ["GEMINI_API_KEY"]


@pytest.fixture
def embeddings(api_key: str):
    """Create GeminiEmbeddings instance."""
    from src.models.gemini.gemini_embeddings import GeminiEmbeddings
    return GeminiEmbeddings(api_key=api_key)


@pytest.fixture
def llm(api_key: str):
    """Create GeminiLLM instance."""
    from src.models.gemini.gemini_llm import GeminiLLM
    return GeminiLLM(api_key=api_key)


class TestContextManagerIntegration:
    """Integration tests for ContextManager with real embeddings."""

    async def test_add_and_retrieve_documents(self, embeddings) -> None:
        """Test adding documents and retrieving them."""
        from src.rag.context_manager import ContextManager
        from src.rag.context_store import InMemoryContextStore

        store = InMemoryContextStore()
        manager = ContextManager(
            embeddings=embeddings,
            store=store,
            max_context_tokens=10000,
        )

        # Add documents
        documents = [
            {
                "id": "doc_001",
                "text": "A forklift moves pallets in the warehouse.",
                "metadata": {"stream_id": "test_stream", "chunk_idx": 0},
            },
            {
                "id": "doc_002",
                "text": "Workers take a lunch break in the cafeteria.",
                "metadata": {"stream_id": "test_stream", "chunk_idx": 1},
            },
            {
                "id": "doc_003",
                "text": "A delivery truck arrives at the loading dock.",
                "metadata": {"stream_id": "test_stream", "chunk_idx": 2},
            },
        ]

        for doc in documents:
            await manager.add_document(
                doc_id=doc["id"],
                text=doc["text"],
                metadata=doc["metadata"],
            )

        # Retrieve relevant documents
        results = await manager.retrieve(
            query="forklift operations",
            top_k=2,
        )

        assert len(results.documents) > 0
        # First result should be about forklift
        assert "forklift" in results.documents[0].text.lower()

    async def test_get_context_window(self, embeddings) -> None:
        """Test getting context window for RAG."""
        from src.rag.context_manager import ContextManager
        from src.rag.context_store import InMemoryContextStore

        store = InMemoryContextStore()
        manager = ContextManager(
            embeddings=embeddings,
            store=store,
            max_context_tokens=1000,
        )

        # Add documents
        captions = [
            "00:00:00-00:01:00: A red forklift enters the warehouse.",
            "00:01:00-00:02:00: The forklift picks up a pallet of electronics.",
            "00:02:00-00:03:00: A worker guides the forklift to the loading area.",
        ]

        for i, caption in enumerate(captions):
            await manager.add_document(
                doc_id=f"cap_{i:03d}",
                text=caption,
                metadata={"stream_id": "test", "chunk_idx": i},
            )

        # Get context window
        context = await manager.get_context_window(
            query="What is the forklift doing?",
            top_k=3,
        )

        assert context is not None
        assert len(context.documents) > 0
        assert context.total_tokens > 0

    async def test_filter_by_stream(self, embeddings) -> None:
        """Test filtering documents by stream_id."""
        from src.rag.context_manager import ContextManager
        from src.rag.context_store import InMemoryContextStore

        store = InMemoryContextStore()
        manager = ContextManager(
            embeddings=embeddings,
            store=store,
        )

        # Add documents from different streams
        await manager.add_document(
            doc_id="stream1_001",
            text="Forklift in warehouse A",
            metadata={"stream_id": "stream1"},
        )
        await manager.add_document(
            doc_id="stream2_001",
            text="Forklift in warehouse B",
            metadata={"stream_id": "stream2"},
        )

        # Retrieve only from stream1
        results = await manager.retrieve(
            query="forklift",
            top_k=10,
            filter_metadata={"stream_id": "stream1"},
        )

        assert len(results.documents) == 1
        assert results.documents[0].metadata["stream_id"] == "stream1"


class TestSummarizationFunctionIntegration:
    """Integration tests for SummarizationFunction."""

    async def test_summarize_captions(self, llm) -> None:
        """Test summarizing video captions."""
        from src.rag.functions.summarization import (
            SummarizationFunction,
            SummarizationInput,
        )

        func = SummarizationFunction(llm=llm)

        captions = [
            {
                "text": "00:00:00-00:01:00: A forklift enters the warehouse.",
                "metadata": {"start_time": "00:00:00", "end_time": "00:01:00"},
            },
            {
                "text": "00:01:00-00:02:00: The forklift picks up pallets.",
                "metadata": {"start_time": "00:01:00", "end_time": "00:02:00"},
            },
            {
                "text": "00:02:00-00:03:00: Workers load boxes onto a truck.",
                "metadata": {"start_time": "00:02:00", "end_time": "00:03:00"},
            },
        ]

        input_data = SummarizationInput(
            captions=captions,
            stream_id="test_stream",
        )

        result = await func.execute(input_data)

        assert result is not None
        assert result.data is not None
        assert result.data.summary is not None
        assert len(result.data.summary) > 0

    async def test_summarize_large_batch(self, llm) -> None:
        """Test summarizing a large batch of captions."""
        from src.rag.functions.summarization import (
            SummarizationFunction,
            SummarizationInput,
        )

        func = SummarizationFunction(llm=llm, batch_size=3)

        # Create many captions
        captions = [
            {
                "text": f"00:{i:02d}:00-00:{i+1:02d}:00: Event {i} in the warehouse.",
                "metadata": {"start_time": f"00:{i:02d}:00", "end_time": f"00:{i+1:02d}:00"},
            }
            for i in range(10)
        ]

        input_data = SummarizationInput(
            captions=captions,
            stream_id="test_stream",
        )

        result = await func.execute(input_data)

        assert result is not None
        assert result.data.summary is not None


class TestHybridRetrievalIntegration:
    """Integration tests for HybridRetriever."""

    @pytest.mark.skipif(
        not is_milvus_available(),
        reason="Milvus not available"
    )
    async def test_vector_only_retrieval(self, embeddings) -> None:
        """Test vector-only retrieval mode."""
        from src.rag.hybrid_retrieval import (
            HybridRetriever,
            HybridConfig,
            RetrievalMode,
        )
        from src.db.milvus_client import MilvusClient, MilvusConfig, VectorDocument

        # Setup Milvus
        milvus_config = MilvusConfig(
            host=os.environ.get("MILVUS_HOST", "localhost"),
            port=int(os.environ.get("MILVUS_PORT", "19530")),
            collection_name=f"hybrid_test_{int(time.time())}",
        )
        milvus_client = MilvusClient(milvus_config)
        await milvus_client.connect()
        await milvus_client.ensure_collection()

        try:
            # Insert test documents
            captions = [
                "A forklift moves pallets in the warehouse.",
                "Workers take a lunch break.",
                "A truck arrives at the loading dock.",
            ]

            for i, caption in enumerate(captions):
                embedding = await embeddings.embed_document(caption)
                doc = VectorDocument(
                    id=f"hybrid_test_{i:03d}",
                    text=caption,
                    embedding=embedding,
                    stream_id="hybrid_test_stream",
                    chunk_idx=i,
                    start_time=f"00:{i:02d}:00",
                    end_time=f"00:{i+1:02d}:00",
                )
                await milvus_client.insert(doc)

            # Create retriever
            config = HybridConfig(
                mode=RetrievalMode.VECTOR_ONLY,
                top_k=2,
            )
            retriever = HybridRetriever(
                milvus_client=milvus_client,
                neo4j_client=None,
                embeddings=embeddings,
                config=config,
            )

            # Retrieve
            results = await retriever.retrieve(
                query="forklift operations",
                stream_id="hybrid_test_stream",
            )

            assert len(results) > 0
            # Should find forklift-related document
            assert any("forklift" in r.document.text.lower() for r in results)

        finally:
            await milvus_client.drop_collection()
            await milvus_client.close()

    @pytest.mark.skipif(
        not (is_milvus_available() and is_neo4j_available()),
        reason="Milvus or Neo4j not available"
    )
    async def test_hybrid_retrieval(self, embeddings) -> None:
        """Test hybrid retrieval combining vector and graph."""
        from src.rag.hybrid_retrieval import (
            HybridRetriever,
            HybridConfig,
            RetrievalMode,
        )
        from src.db.milvus_client import MilvusClient, MilvusConfig, VectorDocument
        from src.db.neo4j_client import Neo4jClient, Neo4jConfig

        # Setup Milvus
        milvus_config = MilvusConfig(
            host=os.environ.get("MILVUS_HOST", "localhost"),
            port=int(os.environ.get("MILVUS_PORT", "19530")),
            collection_name=f"hybrid_full_test_{int(time.time())}",
        )
        milvus_client = MilvusClient(milvus_config)
        await milvus_client.connect()
        await milvus_client.ensure_collection()

        # Setup Neo4j
        neo4j_config = Neo4jConfig(
            host=os.environ.get("NEO4J_HOST", "localhost"),
            port=int(os.environ.get("NEO4J_BOLT_PORT", "7687")),
            username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
        )
        neo4j_client = Neo4jClient(neo4j_config)
        await neo4j_client.connect()
        await neo4j_client.clear_database()

        try:
            # Insert test data
            captions = [
                "A red forklift enters the warehouse.",
                "The forklift picks up electronics pallets.",
                "Workers guide the forklift to loading area.",
            ]

            for i, caption in enumerate(captions):
                # Insert into Milvus
                embedding = await embeddings.embed_document(caption)
                doc = VectorDocument(
                    id=f"hybrid_full_{i:03d}",
                    text=caption,
                    embedding=embedding,
                    stream_id="hybrid_full_stream",
                    chunk_idx=i,
                    start_time=f"00:{i:02d}:00",
                    end_time=f"00:{i+1:02d}:00",
                )
                await milvus_client.insert(doc)

                # Insert into Neo4j
                await neo4j_client.create_node(
                    labels=["VideoChunk"],
                    properties={
                        "id": f"hybrid_full_{i:03d}",
                        "stream_id": "hybrid_full_stream",
                        "chunk_idx": i,
                        "caption": caption,
                    }
                )

            # Create entity nodes
            await neo4j_client.create_node(
                labels=["Entity"],
                properties={"id": "forklift_001", "name": "forklift", "type": "VEHICLE"}
            )

            # Create relationships
            for i in range(3):
                await neo4j_client.create_relationship(
                    f"hybrid_full_{i:03d}",
                    "forklift_001",
                    "CONTAINS"
                )

            # Create retriever
            config = HybridConfig(
                mode=RetrievalMode.HYBRID,
                vector_weight=0.6,
                graph_weight=0.4,
                top_k=3,
            )
            retriever = HybridRetriever(
                milvus_client=milvus_client,
                neo4j_client=neo4j_client,
                embeddings=embeddings,
                config=config,
            )

            # Retrieve
            results = await retriever.retrieve(
                query="What is the forklift doing?",
                stream_id="hybrid_full_stream",
            )

            assert len(results) > 0

        finally:
            await milvus_client.drop_collection()
            await milvus_client.close()
            await neo4j_client.clear_database()
            await neo4j_client.close()


class TestGraphIngestionFunctionIntegration:
    """Integration tests for GraphIngestionFunction."""

    @pytest.mark.skipif(
        not is_neo4j_available(),
        reason="Neo4j not available"
    )
    async def test_ingest_document(self, llm, embeddings) -> None:
        """Test ingesting a document into the graph."""
        from src.rag.functions.graph_ingestion import GraphIngestionFunction
        from src.db.neo4j_client import Neo4jClient, Neo4jConfig

        neo4j_config = Neo4jConfig(
            host=os.environ.get("NEO4J_HOST", "localhost"),
            port=int(os.environ.get("NEO4J_BOLT_PORT", "7687")),
            username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
        )
        neo4j_client = Neo4jClient(neo4j_config)
        await neo4j_client.connect()
        await neo4j_client.clear_database()

        try:
            func = GraphIngestionFunction(
                neo4j_client=neo4j_client,
                llm=llm,
                embeddings=embeddings,
            )

            result = await func.execute({
                "chunk_id": "ingest_test_001",
                "stream_id": "ingest_test_stream",
                "chunk_idx": 0,
                "caption": "A forklift moves pallets in the warehouse.",
                "start_time": "00:00:00",
                "end_time": "00:01:00",
                "embedding_id": "milvus_001",
            })

            assert result is not None
            assert result.success

            # Verify chunk was created
            chunk = await neo4j_client.get_node_by_id("ingest_test_001")
            assert chunk is not None

        finally:
            await neo4j_client.clear_database()
            await neo4j_client.close()


class TestGraphRetrievalFunctionIntegration:
    """Integration tests for GraphRetrievalFunction."""

    @pytest.mark.skipif(
        not is_neo4j_available(),
        reason="Neo4j not available"
    )
    async def test_retrieve_context(self, llm, embeddings) -> None:
        """Test retrieving context from graph."""
        from src.rag.functions.graph_retrieval import GraphRetrievalFunction
        from src.db.neo4j_client import Neo4jClient, Neo4jConfig

        neo4j_config = Neo4jConfig(
            host=os.environ.get("NEO4J_HOST", "localhost"),
            port=int(os.environ.get("NEO4J_BOLT_PORT", "7687")),
            username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
        )
        neo4j_client = Neo4jClient(neo4j_config)
        await neo4j_client.connect()
        await neo4j_client.clear_database()

        try:
            # Create test data
            for i in range(3):
                await neo4j_client.create_node(
                    labels=["VideoChunk"],
                    properties={
                        "id": f"retrieval_test_{i:03d}",
                        "stream_id": "retrieval_test_stream",
                        "chunk_idx": i,
                        "caption": f"Event {i} in the warehouse.",
                    }
                )

            func = GraphRetrievalFunction(
                neo4j_client=neo4j_client,
                llm=llm,
                embeddings=embeddings,
            )

            result = await func.execute({
                "query": "What happened in the warehouse?",
                "stream_id": "retrieval_test_stream",
                "top_k": 3,
            })

            assert result is not None

        finally:
            await neo4j_client.clear_database()
            await neo4j_client.close()


class TestEndToEndRAGPipeline:
    """End-to-end tests for the complete RAG pipeline."""

    @pytest.mark.skipif(
        not is_milvus_available(),
        reason="Milvus not available"
    )
    async def test_ingest_and_query(self, embeddings, llm) -> None:
        """Test complete ingest and query workflow."""
        from src.rag.context_manager import ContextManager
        from src.rag.milvus_store import MilvusContextStore
        from src.db.milvus_client import MilvusConfig

        milvus_config = MilvusConfig(
            host=os.environ.get("MILVUS_HOST", "localhost"),
            port=int(os.environ.get("MILVUS_PORT", "19530")),
            collection_name=f"e2e_test_{int(time.time())}",
        )

        store = MilvusContextStore(config=milvus_config)
        await store.connect()

        try:
            manager = ContextManager(
                embeddings=embeddings,
                store=store,
                max_context_tokens=10000,
            )

            # Ingest video captions
            captions = [
                "00:00:00-00:01:00: A red forklift enters the warehouse from the main entrance.",
                "00:01:00-00:02:00: The forklift picks up a pallet of electronic components.",
                "00:02:00-00:03:00: A worker in a blue vest guides the forklift.",
                "00:03:00-00:04:00: The forklift moves to the shipping area.",
                "00:04:00-00:05:00: Boxes are loaded onto a delivery truck.",
            ]

            for i, caption in enumerate(captions):
                await manager.add_document(
                    doc_id=f"e2e_cap_{i:03d}",
                    text=caption,
                    metadata={
                        "stream_id": "e2e_test_stream",
                        "chunk_idx": i,
                        "start_time": f"00:{i:02d}:00",
                        "end_time": f"00:{i+1:02d}:00",
                    },
                )

            # Query the system
            query = "What color was the forklift?"

            # Retrieve context
            context = await manager.get_context_window(
                query=query,
                top_k=3,
            )

            assert len(context.documents) > 0

            # Generate answer using LLM
            context_text = "\n".join([d.text for d in context.documents])
            prompt = f"""Based on the following video captions, answer the question.

Captions:
{context_text}

Question: {query}

Answer:"""

            response = await llm.generate(prompt)

            assert response.text is not None
            assert "red" in response.text.lower()

        finally:
            await store.drop_collection()
            await store.close()
