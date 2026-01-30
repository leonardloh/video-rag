"""Milvus-based context store for RAG document storage."""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from .context_store import ContextStore, Document

if TYPE_CHECKING:
    pass


class MilvusContextStore(ContextStore):
    """Milvus-based context store for persistent vector storage."""

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

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "vss_poc_context",
        embedding_dim: int = 768,
        alias: str = "default",
    ) -> None:
        """
        Initialize Milvus context store.

        Args:
            host: Milvus host
            port: Milvus port
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
            alias: Connection alias
        """
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._alias = alias
        self._collection: Optional[Collection] = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to Milvus."""
        if not self._connected:
            connections.connect(
                alias=self._alias,
                host=self._host,
                port=self._port,
            )
            self._connected = True
        self._ensure_collection()

    async def close(self) -> None:
        """Close connection to Milvus."""
        if self._connected:
            connections.disconnect(alias=self._alias)
            self._connected = False

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        if utility.has_collection(self._collection_name, using=self._alias):
            self._collection = Collection(
                name=self._collection_name,
                using=self._alias,
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
                dim=self._embedding_dim,
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
                max_length=64,
            ),
            FieldSchema(
                name="end_time",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="metadata_json",
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
            description="RAG context documents",
        )

        self._collection = Collection(
            name=self._collection_name,
            schema=schema,
            using=self._alias,
        )

        # Create index on embedding field
        self._collection.create_index(
            field_name="embedding",
            index_params=self.INDEX_PARAMS,
        )

    def _get_collection(self) -> Collection:
        """Get collection, raising error if not initialized."""
        if self._collection is None:
            raise RuntimeError("Collection not initialized. Call connect() first.")
        return self._collection

    async def add_document(
        self,
        text: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> str:
        """Add a document to the store."""
        collection = self._get_collection()
        doc_id = str(uuid.uuid4())

        # Use zero vector if no embedding provided
        if embedding is None:
            embedding = [0.0] * self._embedding_dim

        data = [
            [doc_id],
            [text],
            [embedding],
            [metadata.get("stream_id", "")],
            [metadata.get("chunk_idx", 0)],
            [metadata.get("start_time", "")],
            [metadata.get("end_time", "")],
            [json.dumps(metadata)],
            [int(time.time())],
        ]

        collection.insert(data)
        collection.flush()

        return doc_id

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        collection = self._get_collection()
        collection.load()

        results = collection.query(
            expr=f'id == "{doc_id}"',
            output_fields=[
                "id",
                "text",
                "stream_id",
                "chunk_idx",
                "start_time",
                "end_time",
                "metadata_json",
                "created_at",
            ],
        )

        if not results:  # type: ignore[truthy-bool]
            return None

        r = results[0]  # type: ignore[index]
        metadata = json.loads(r.get("metadata_json", "{}"))

        return Document(
            id=r["id"],
            text=r["text"],
            embedding=None,  # Don't return embedding to save memory
            metadata=metadata,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents."""
        collection = self._get_collection()
        collection.load()

        # Build filter expression
        expr = None
        if filter_metadata:
            conditions = []
            if "stream_id" in filter_metadata:
                conditions.append(f'stream_id == "{filter_metadata["stream_id"]}"')
            if conditions:
                expr = " && ".join(conditions)

        output_fields = [
            "id",
            "text",
            "stream_id",
            "chunk_idx",
            "start_time",
            "end_time",
            "metadata_json",
            "created_at",
        ]

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=self.SEARCH_PARAMS,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )

        documents: list[tuple[Document, float]] = []
        for hit in results[0]:  # type: ignore[index]
            metadata = json.loads(hit.entity.get("metadata_json", "{}"))

            doc = Document(
                id=hit.id,
                text=hit.entity.get("text", ""),
                embedding=None,
                metadata=metadata,
            )
            documents.append((doc, hit.score))

        return documents

    async def get_all_documents(
        self,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        """Get all documents, optionally filtered."""
        collection = self._get_collection()
        collection.load()

        # Build filter expression
        expr = "id != ''"
        if filter_metadata and "stream_id" in filter_metadata:
            expr = f'stream_id == "{filter_metadata["stream_id"]}"'

        results = collection.query(
            expr=expr,
            output_fields=[
                "id",
                "text",
                "stream_id",
                "chunk_idx",
                "start_time",
                "end_time",
                "metadata_json",
                "created_at",
            ],
        )

        documents = []
        for r in results:  # type: ignore[union-attr]
            metadata = json.loads(r.get("metadata_json", "{}"))

            documents.append(
                Document(
                    id=r["id"],
                    text=r["text"],
                    embedding=None,
                    metadata=metadata,
                )
            )

        # Sort by chunk_idx
        documents.sort(key=lambda d: d.metadata.get("chunk_idx", 0))

        return documents

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        collection = self._get_collection()
        collection.delete(f'id == "{doc_id}"')
        return True

    async def delete_by_stream(self, stream_id: str) -> int:
        """Delete all documents for a stream."""
        collection = self._get_collection()
        collection.load()

        # Get count first
        results = collection.query(
            expr=f'stream_id == "{stream_id}"',
            output_fields=["id"],
        )
        count = len(results)  # type: ignore[arg-type]

        if count > 0:
            collection.delete(f'stream_id == "{stream_id}"')

        return count

    async def clear(self) -> int:
        """Clear all documents."""
        collection = self._get_collection()
        collection.flush()

        count = collection.num_entities

        # Drop and recreate collection
        utility.drop_collection(self._collection_name, using=self._alias)
        self._collection = None
        self._ensure_collection()

        return count

    async def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        collection = self._get_collection()
        collection.flush()

        return {
            "name": self._collection_name,
            "num_entities": collection.num_entities,
        }
