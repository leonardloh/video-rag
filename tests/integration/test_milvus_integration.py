"""Integration tests for Milvus vector database.

These tests require a running Milvus instance.
Start Milvus with: docker-compose up -d milvus-standalone

Run with: pytest tests/integration/test_milvus_integration.py -v
"""

from __future__ import annotations

import os
import time
import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db.milvus_client import MilvusClient, MilvusConfig, VectorDocument


def is_milvus_available() -> bool:
    """Check if Milvus is available."""
    try:
        from pymilvus import connections
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = int(os.environ.get("MILVUS_PORT", "19530"))
        connections.connect(alias="test_check", host=host, port=port)
        connections.disconnect(alias="test_check")
        return True
    except Exception:
        return False


# Skip all tests if Milvus is not available
pytestmark = pytest.mark.skipif(
    not is_milvus_available(),
    reason="Milvus is not available"
)


@pytest.fixture
def milvus_config() -> "MilvusConfig":
    """Create Milvus configuration."""
    from src.db.milvus_client import MilvusConfig

    return MilvusConfig(
        host=os.environ.get("MILVUS_HOST", "localhost"),
        port=int(os.environ.get("MILVUS_PORT", "19530")),
        collection_name=f"test_collection_{int(time.time())}",
        embedding_dim=768,
        alias=f"test_{int(time.time())}",
    )


@pytest.fixture
async def milvus_client(milvus_config: "MilvusConfig") -> "MilvusClient":
    """Create and connect Milvus client."""
    from src.db.milvus_client import MilvusClient

    client = MilvusClient(milvus_config)
    await client.connect()
    await client.ensure_collection()

    yield client

    # Cleanup
    await client.drop_collection()
    await client.close()


def create_test_embedding(seed: int = 0) -> list[float]:
    """Create a test embedding vector."""
    import numpy as np
    np.random.seed(seed)
    return np.random.randn(768).tolist()


def create_test_document(
    doc_id: str,
    stream_id: str = "test_stream",
    chunk_idx: int = 0,
    text: str = "Test caption",
    embedding_seed: int = 0,
) -> "VectorDocument":
    """Create a test document."""
    from src.db.milvus_client import VectorDocument

    return VectorDocument(
        id=doc_id,
        text=text,
        embedding=create_test_embedding(embedding_seed),
        stream_id=stream_id,
        chunk_idx=chunk_idx,
        start_time=f"00:{chunk_idx:02d}:00",
        end_time=f"00:{chunk_idx + 1:02d}:00",
        start_pts=chunk_idx * 60_000_000_000,
        end_pts=(chunk_idx + 1) * 60_000_000_000,
        cv_meta='{"objects": ["person"]}',
        created_at=int(time.time()),
    )


class TestMilvusClientConnection:
    """Test Milvus connection and collection management."""

    async def test_connect_and_close(self, milvus_config: "MilvusConfig") -> None:
        """Test connecting and closing connection."""
        from src.db.milvus_client import MilvusClient

        client = MilvusClient(milvus_config)
        await client.connect()

        # Should be connected
        assert client._connected is True

        await client.close()

        # Should be disconnected
        assert client._connected is False

    async def test_ensure_collection_creates_new(
        self,
        milvus_config: "MilvusConfig"
    ) -> None:
        """Test that ensure_collection creates a new collection."""
        from src.db.milvus_client import MilvusClient
        from pymilvus import utility

        client = MilvusClient(milvus_config)
        await client.connect()

        # Collection should not exist yet
        assert not utility.has_collection(
            milvus_config.collection_name,
            using=milvus_config.alias
        )

        await client.ensure_collection()

        # Collection should now exist
        assert utility.has_collection(
            milvus_config.collection_name,
            using=milvus_config.alias
        )

        await client.drop_collection()
        await client.close()

    async def test_get_collection_stats(
        self,
        milvus_client: "MilvusClient"
    ) -> None:
        """Test getting collection statistics."""
        stats = await milvus_client.get_collection_stats()

        assert "name" in stats
        assert "num_entities" in stats
        assert stats["num_entities"] == 0


class TestMilvusClientInsert:
    """Test document insertion."""

    async def test_insert_single_document(
        self,
        milvus_client: "MilvusClient"
    ) -> None:
        """Test inserting a single document."""
        doc = create_test_document("doc_001")

        doc_id = await milvus_client.insert(doc)

        assert doc_id == "doc_001"

        # Verify document was inserted
        retrieved = await milvus_client.get_by_id("doc_001")
        assert retrieved is not None
        assert retrieved.text == doc.text
        assert retrieved.stream_id == doc.stream_id

    async def test_insert_batch(self, milvus_client: "MilvusClient") -> None:
        """Test batch inserting documents."""
        documents = [
            create_test_document(f"batch_{i:03d}", chunk_idx=i, embedding_seed=i)
            for i in range(10)
        ]

        doc_ids = await milvus_client.insert_batch(documents)

        assert len(doc_ids) == 10

        # Verify all documents were inserted
        stats = await milvus_client.get_collection_stats()
        assert stats["num_entities"] == 10

    async def test_insert_large_batch(self, milvus_client: "MilvusClient") -> None:
        """Test inserting a large batch of documents."""
        documents = [
            create_test_document(
                f"large_{i:04d}",
                chunk_idx=i,
                embedding_seed=i,
                text=f"Caption for chunk {i}"
            )
            for i in range(200)
        ]

        doc_ids = await milvus_client.insert_batch(documents, batch_size=50)

        assert len(doc_ids) == 200

        stats = await milvus_client.get_collection_stats()
        assert stats["num_entities"] == 200


class TestMilvusClientSearch:
    """Test vector search functionality."""

    async def test_search_basic(self, milvus_client: "MilvusClient") -> None:
        """Test basic vector search."""
        # Insert test documents
        documents = [
            create_test_document(f"search_{i:03d}", chunk_idx=i, embedding_seed=i)
            for i in range(5)
        ]
        await milvus_client.insert_batch(documents)

        # Search with embedding similar to first document
        query_embedding = create_test_embedding(0)
        results = await milvus_client.search(query_embedding, top_k=3)

        assert len(results) == 3
        # First result should be most similar (same seed)
        assert results[0].document.id == "search_000"
        assert results[0].score > 0.9  # High similarity

    async def test_search_by_stream(self, milvus_client: "MilvusClient") -> None:
        """Test searching within a specific stream."""
        # Insert documents in different streams
        doc1 = create_test_document("stream1_001", stream_id="stream1", embedding_seed=1)
        doc2 = create_test_document("stream1_002", stream_id="stream1", embedding_seed=2)
        doc3 = create_test_document("stream2_001", stream_id="stream2", embedding_seed=3)

        await milvus_client.insert_batch([doc1, doc2, doc3])

        # Search only in stream1
        query_embedding = create_test_embedding(1)
        results = await milvus_client.search_by_stream(
            query_embedding,
            stream_id="stream1",
            top_k=10
        )

        assert len(results) == 2
        assert all(r.document.stream_id == "stream1" for r in results)

    async def test_search_with_filter(self, milvus_client: "MilvusClient") -> None:
        """Test search with filter expression."""
        # Insert documents with different chunk indices
        documents = [
            create_test_document(f"filter_{i:03d}", chunk_idx=i, embedding_seed=i)
            for i in range(10)
        ]
        await milvus_client.insert_batch(documents)

        # Search with filter for chunk_idx >= 5
        query_embedding = create_test_embedding(7)
        results = await milvus_client.search(
            query_embedding,
            top_k=10,
            filter_expr="chunk_idx >= 5"
        )

        assert len(results) == 5
        assert all(r.document.chunk_idx >= 5 for r in results)

    async def test_search_empty_collection(
        self,
        milvus_client: "MilvusClient"
    ) -> None:
        """Test searching an empty collection."""
        query_embedding = create_test_embedding(0)
        results = await milvus_client.search(query_embedding, top_k=5)

        assert len(results) == 0


class TestMilvusClientRetrieval:
    """Test document retrieval."""

    async def test_get_by_id(self, milvus_client: "MilvusClient") -> None:
        """Test retrieving document by ID."""
        doc = create_test_document("get_test_001", text="Specific caption")
        await milvus_client.insert(doc)

        retrieved = await milvus_client.get_by_id("get_test_001")

        assert retrieved is not None
        assert retrieved.id == "get_test_001"
        assert retrieved.text == "Specific caption"

    async def test_get_by_id_not_found(
        self,
        milvus_client: "MilvusClient"
    ) -> None:
        """Test retrieving non-existent document."""
        result = await milvus_client.get_by_id("nonexistent_id")

        assert result is None

    async def test_get_by_stream(self, milvus_client: "MilvusClient") -> None:
        """Test retrieving all documents for a stream."""
        # Insert documents in target stream
        for i in range(5):
            doc = create_test_document(
                f"stream_test_{i:03d}",
                stream_id="target_stream",
                chunk_idx=i,
                embedding_seed=i
            )
            await milvus_client.insert(doc)

        # Insert document in different stream
        other_doc = create_test_document(
            "other_001",
            stream_id="other_stream",
            embedding_seed=100
        )
        await milvus_client.insert(other_doc)

        # Get documents for target stream
        documents = await milvus_client.get_by_stream("target_stream")

        assert len(documents) == 5
        assert all(d.stream_id == "target_stream" for d in documents)
        # Should be sorted by chunk_idx
        assert [d.chunk_idx for d in documents] == [0, 1, 2, 3, 4]


class TestMilvusClientDelete:
    """Test document deletion."""

    async def test_delete_single(self, milvus_client: "MilvusClient") -> None:
        """Test deleting a single document."""
        doc = create_test_document("delete_test_001")
        await milvus_client.insert(doc)

        # Verify document exists
        assert await milvus_client.get_by_id("delete_test_001") is not None

        # Delete document
        result = await milvus_client.delete("delete_test_001")
        assert result is True

        # Verify document is deleted (may need flush)
        # Note: Milvus delete is eventually consistent

    async def test_delete_by_stream(self, milvus_client: "MilvusClient") -> None:
        """Test deleting all documents for a stream."""
        # Insert documents in target stream
        for i in range(5):
            doc = create_test_document(
                f"del_stream_{i:03d}",
                stream_id="delete_stream",
                chunk_idx=i,
                embedding_seed=i
            )
            await milvus_client.insert(doc)

        # Insert document in different stream (should not be deleted)
        other_doc = create_test_document(
            "keep_001",
            stream_id="keep_stream",
            embedding_seed=100
        )
        await milvus_client.insert(other_doc)

        # Delete by stream
        deleted_count = await milvus_client.delete_by_stream("delete_stream")

        assert deleted_count == 5

        # Verify other stream document still exists
        kept = await milvus_client.get_by_id("keep_001")
        assert kept is not None


class TestMilvusClientWithEmbeddings:
    """Integration tests combining Milvus with real embeddings."""

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
    async def test_semantic_search_with_real_embeddings(
        self,
        milvus_client: "MilvusClient"
    ) -> None:
        """Test semantic search with real Gemini embeddings."""
        from src.models.gemini.gemini_embeddings import GeminiEmbeddings
        from src.db.milvus_client import VectorDocument

        api_key = os.environ["GEMINI_API_KEY"]
        embeddings = GeminiEmbeddings(api_key=api_key)

        # Create documents with real embeddings
        captions = [
            "A forklift moves pallets across the warehouse floor.",
            "Workers take a lunch break in the cafeteria.",
            "A delivery truck arrives at the loading dock.",
            "Security guard patrols the parking lot.",
            "Forklift operator loads boxes onto a truck.",
        ]

        documents = []
        for i, caption in enumerate(captions):
            embedding = await embeddings.embed_document(caption)
            doc = VectorDocument(
                id=f"real_emb_{i:03d}",
                text=caption,
                embedding=embedding,
                stream_id="real_test",
                chunk_idx=i,
                start_time=f"00:{i:02d}:00",
                end_time=f"00:{i + 1:02d}:00",
            )
            documents.append(doc)

        await milvus_client.insert_batch(documents)

        # Search for forklift-related content
        query = "forklift operations"
        query_embedding = await embeddings.embed_query(query)

        results = await milvus_client.search(query_embedding, top_k=3)

        assert len(results) == 3
        # Top results should be forklift-related
        top_texts = [r.document.text.lower() for r in results[:2]]
        assert any("forklift" in text for text in top_texts)

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
    async def test_full_rag_workflow(
        self,
        milvus_client: "MilvusClient"
    ) -> None:
        """Test complete RAG workflow with Milvus and Gemini."""
        from src.models.gemini.gemini_embeddings import GeminiEmbeddings
        from src.models.gemini.gemini_llm import GeminiLLM
        from src.db.milvus_client import VectorDocument

        api_key = os.environ["GEMINI_API_KEY"]
        embeddings = GeminiEmbeddings(api_key=api_key)
        llm = GeminiLLM(api_key=api_key)

        # Ingest video captions
        captions = [
            "00:00:00-00:01:00: A red forklift enters the warehouse.",
            "00:01:00-00:02:00: The forklift picks up a pallet of electronics.",
            "00:02:00-00:03:00: A worker in a blue vest guides the forklift.",
            "00:03:00-00:04:00: The forklift moves to the shipping area.",
            "00:04:00-00:05:00: Boxes are loaded onto a delivery truck.",
        ]

        documents = []
        for i, caption in enumerate(captions):
            embedding = await embeddings.embed_document(caption)
            doc = VectorDocument(
                id=f"rag_test_{i:03d}",
                text=caption,
                embedding=embedding,
                stream_id="rag_test_stream",
                chunk_idx=i,
                start_time=f"00:{i:02d}:00",
                end_time=f"00:{i + 1:02d}:00",
            )
            documents.append(doc)

        await milvus_client.insert_batch(documents)

        # User query
        query = "What color was the forklift?"
        query_embedding = await embeddings.embed_query(query)

        # Retrieve relevant context
        results = await milvus_client.search(query_embedding, top_k=2)

        # Generate answer
        context = "\n".join([r.document.text for r in results])
        prompt = f"""Based on the following video captions, answer the question.

Captions:
{context}

Question: {query}

Answer:"""

        response = await llm.generate(prompt)

        assert response.text is not None
        assert "red" in response.text.lower()


class TestMilvusClientPerformance:
    """Performance tests for Milvus operations."""

    async def test_bulk_insert_performance(
        self,
        milvus_client: "MilvusClient"
    ) -> None:
        """Test performance of bulk insert."""
        import time

        # Create 1000 documents
        documents = [
            create_test_document(
                f"perf_{i:05d}",
                chunk_idx=i % 100,
                embedding_seed=i,
                text=f"Performance test caption {i}"
            )
            for i in range(1000)
        ]

        start_time = time.time()
        await milvus_client.insert_batch(documents, batch_size=100)
        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 30 seconds)
        assert elapsed < 30

        stats = await milvus_client.get_collection_stats()
        assert stats["num_entities"] == 1000

    async def test_search_performance(
        self,
        milvus_client: "MilvusClient"
    ) -> None:
        """Test search performance with many documents."""
        import time

        # Insert documents
        documents = [
            create_test_document(
                f"search_perf_{i:04d}",
                chunk_idx=i % 100,
                embedding_seed=i
            )
            for i in range(500)
        ]
        await milvus_client.insert_batch(documents, batch_size=100)

        # Measure search time
        query_embedding = create_test_embedding(250)

        start_time = time.time()
        for _ in range(10):
            await milvus_client.search(query_embedding, top_k=10)
        elapsed = time.time() - start_time

        # 10 searches should complete in < 5 seconds
        assert elapsed < 5
