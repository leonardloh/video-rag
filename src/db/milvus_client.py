"""Milvus vector database client for semantic search."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


@dataclass
class MilvusConfig:
    """Milvus connection configuration."""

    host: str = "localhost"
    port: int = 19530
    collection_name: str = "vss_poc_captions"
    embedding_dim: int = 768
    alias: str = "default"


@dataclass
class VectorDocument:
    """A document stored in Milvus."""

    id: str
    text: str
    embedding: list[float]
    stream_id: str
    chunk_idx: int
    start_time: str
    end_time: str
    start_pts: int = 0
    end_pts: int = 0
    cv_meta: str = ""
    created_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class SearchResult:
    """Result from vector search."""

    document: VectorDocument
    score: float  # Cosine similarity score (0-1)


class MilvusClient:
    """Client for Milvus vector database operations."""

    # Index parameters
    INDEX_PARAMS: dict[str, Any] = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }

    # Search parameters
    SEARCH_PARAMS: dict[str, Any] = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    def __init__(self, config: MilvusConfig) -> None:
        """
        Initialize Milvus client.

        Args:
            config: Milvus connection configuration
        """
        self._config = config
        self._collection: Optional[Collection] = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to Milvus."""
        connections.connect(
            alias=self._config.alias,
            host=self._config.host,
            port=self._config.port,
        )
        self._connected = True

    async def close(self) -> None:
        """Close connection to Milvus."""
        if self._connected:
            connections.disconnect(alias=self._config.alias)
            self._connected = False

    async def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        if utility.has_collection(self._config.collection_name, using=self._config.alias):
            self._collection = Collection(
                name=self._config.collection_name,
                using=self._config.alias,
            )
            return

        # Define schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._config.embedding_dim,
            ),
            FieldSchema(
                name="stream_id",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="chunk_idx",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="start_time",
                dtype=DataType.VARCHAR,
                max_length=32,
            ),
            FieldSchema(
                name="end_time",
                dtype=DataType.VARCHAR,
                max_length=32,
            ),
            FieldSchema(
                name="start_pts",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="end_pts",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="cv_meta",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Video caption embeddings for VSS PoC",
        )

        self._collection = Collection(
            name=self._config.collection_name,
            schema=schema,
            using=self._config.alias,
        )

        # Create index on embedding field
        self._collection.create_index(
            field_name="embedding",
            index_params=self.INDEX_PARAMS,
        )

    def _get_collection(self) -> Collection:
        """Get collection, raising error if not initialized."""
        if self._collection is None:
            raise RuntimeError("Collection not initialized. Call ensure_collection() first.")
        return self._collection

    async def insert(self, document: VectorDocument) -> str:
        """
        Insert a single document.

        Args:
            document: Document to insert

        Returns:
            Document ID
        """
        collection = self._get_collection()

        data = [
            [document.id],
            [document.text],
            [document.embedding],
            [document.stream_id],
            [document.chunk_idx],
            [document.start_time],
            [document.end_time],
            [document.start_pts],
            [document.end_pts],
            [document.cv_meta],
            [document.created_at],
        ]

        collection.insert(data)
        collection.flush()

        return document.id

    async def insert_batch(
        self,
        documents: list[VectorDocument],
        batch_size: int = 100,
    ) -> list[str]:
        """
        Insert multiple documents in batches.

        Args:
            documents: Documents to insert
            batch_size: Number of documents per batch

        Returns:
            List of document IDs
        """
        collection = self._get_collection()
        doc_ids: list[str] = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            data = [
                [doc.id for doc in batch],
                [doc.text for doc in batch],
                [doc.embedding for doc in batch],
                [doc.stream_id for doc in batch],
                [doc.chunk_idx for doc in batch],
                [doc.start_time for doc in batch],
                [doc.end_time for doc in batch],
                [doc.start_pts for doc in batch],
                [doc.end_pts for doc in batch],
                [doc.cv_meta for doc in batch],
                [doc.created_at for doc in batch],
            ]

            collection.insert(data)
            doc_ids.extend([doc.id for doc in batch])

        collection.flush()
        return doc_ids

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector (768 dimensions)
            top_k: Number of results to return
            filter_expr: Optional Milvus filter expression

        Returns:
            List of search results with scores
        """
        collection = self._get_collection()
        collection.load()

        output_fields = [
            "id", "text", "stream_id", "chunk_idx",
            "start_time", "end_time", "start_pts", "end_pts",
            "cv_meta", "created_at",
        ]

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=self.SEARCH_PARAMS,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields,
        )

        search_results: list[SearchResult] = []
        for hit in results[0]:
            doc = VectorDocument(
                id=hit.id,
                text=hit.entity.get("text", ""),
                embedding=[],  # Don't return embedding to save memory
                stream_id=hit.entity.get("stream_id", ""),
                chunk_idx=hit.entity.get("chunk_idx", 0),
                start_time=hit.entity.get("start_time", ""),
                end_time=hit.entity.get("end_time", ""),
                start_pts=hit.entity.get("start_pts", 0),
                end_pts=hit.entity.get("end_pts", 0),
                cv_meta=hit.entity.get("cv_meta", ""),
                created_at=hit.entity.get("created_at", 0),
            )
            search_results.append(SearchResult(document=doc, score=hit.score))

        return search_results

    async def search_by_stream(
        self,
        query_embedding: list[float],
        stream_id: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search within a specific stream."""
        filter_expr = f'stream_id == "{stream_id}"'
        return await self.search(query_embedding, top_k, filter_expr)

    async def search_by_time_range(
        self,
        query_embedding: list[float],
        start_time: str,
        end_time: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search within a time range."""
        filter_expr = f'start_time >= "{start_time}" && end_time <= "{end_time}"'
        return await self.search(query_embedding, top_k, filter_expr)

    async def get_by_id(self, doc_id: str) -> Optional[VectorDocument]:
        """Retrieve document by ID."""
        collection = self._get_collection()
        collection.load()

        results = collection.query(
            expr=f'id == "{doc_id}"',
            output_fields=[
                "id", "text", "stream_id", "chunk_idx",
                "start_time", "end_time", "start_pts", "end_pts",
                "cv_meta", "created_at",
            ],
        )

        if not results:
            return None

        r = results[0]
        return VectorDocument(
            id=r["id"],
            text=r["text"],
            embedding=[],
            stream_id=r["stream_id"],
            chunk_idx=r["chunk_idx"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            start_pts=r["start_pts"],
            end_pts=r["end_pts"],
            cv_meta=r["cv_meta"],
            created_at=r["created_at"],
        )

    async def get_by_stream(
        self,
        stream_id: str,
        limit: int = 1000,
    ) -> list[VectorDocument]:
        """Get all documents for a stream."""
        collection = self._get_collection()
        collection.load()

        results = collection.query(
            expr=f'stream_id == "{stream_id}"',
            output_fields=[
                "id", "text", "stream_id", "chunk_idx",
                "start_time", "end_time", "start_pts", "end_pts",
                "cv_meta", "created_at",
            ],
            limit=limit,
        )

        documents = [
            VectorDocument(
                id=r["id"],
                text=r["text"],
                embedding=[],
                stream_id=r["stream_id"],
                chunk_idx=r["chunk_idx"],
                start_time=r["start_time"],
                end_time=r["end_time"],
                start_pts=r["start_pts"],
                end_pts=r["end_pts"],
                cv_meta=r["cv_meta"],
                created_at=r["created_at"],
            )
            for r in results
        ]

        # Sort by chunk_idx
        documents.sort(key=lambda d: d.chunk_idx)
        return documents

    async def delete(self, doc_id: str) -> bool:
        """Delete document by ID."""
        collection = self._get_collection()
        collection.delete(f'id == "{doc_id}"')
        return True

    async def delete_by_stream(self, stream_id: str) -> int:
        """Delete all documents for a stream. Returns count deleted."""
        collection = self._get_collection()
        collection.load()

        # Get count first
        results = collection.query(
            expr=f'stream_id == "{stream_id}"',
            output_fields=["id"],
        )
        count = len(results)

        if count > 0:
            collection.delete(f'stream_id == "{stream_id}"')

        return count

    async def drop_collection(self) -> None:
        """Drop the entire collection."""
        if utility.has_collection(self._config.collection_name, using=self._config.alias):
            utility.drop_collection(self._config.collection_name, using=self._config.alias)
        self._collection = None

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        collection = self._get_collection()
        collection.flush()

        stats = {
            "name": self._config.collection_name,
            "num_entities": collection.num_entities,
            "schema": str(collection.schema),
        }

        return stats

    @staticmethod
    def generate_id() -> str:
        """Generate a unique document ID."""
        return str(uuid.uuid4())
