"""Integration tests for Neo4j graph database.

These tests require a running Neo4j instance.
Start Neo4j with: docker-compose up -d neo4j

Run with: pytest tests/integration/test_neo4j_integration.py -v
"""

from __future__ import annotations

import os
import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db.neo4j_client import Neo4jClient, Neo4jConfig, GraphNode


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


# Skip all tests if Neo4j is not available
pytestmark = pytest.mark.skipif(
    not is_neo4j_available(),
    reason="Neo4j is not available"
)


@pytest.fixture
def neo4j_config() -> "Neo4jConfig":
    """Create Neo4j configuration."""
    from src.db.neo4j_client import Neo4jConfig

    return Neo4jConfig(
        host=os.environ.get("NEO4J_HOST", "localhost"),
        port=int(os.environ.get("NEO4J_BOLT_PORT", "7687")),
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password"),
        database="neo4j",
    )


@pytest.fixture
async def neo4j_client(neo4j_config: "Neo4jConfig") -> "Neo4jClient":
    """Create and connect Neo4j client."""
    from src.db.neo4j_client import Neo4jClient

    client = Neo4jClient(neo4j_config)
    await client.connect()

    # Clear database before tests
    await client.clear_database()

    yield client

    # Cleanup after tests
    await client.clear_database()
    await client.close()


class TestNeo4jClientConnection:
    """Test Neo4j connection management."""

    async def test_connect_and_close(self, neo4j_config: "Neo4jConfig") -> None:
        """Test connecting and closing connection."""
        from src.db.neo4j_client import Neo4jClient

        client = Neo4jClient(neo4j_config)
        await client.connect()

        # Should be connected
        assert client._driver is not None

        await client.close()

        # Should be disconnected
        assert client._driver is None

    async def test_execute_simple_query(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test executing a simple Cypher query."""
        result = await neo4j_client.execute_query("RETURN 1 + 1 as sum")

        assert len(result) == 1
        assert result[0]["sum"] == 2

    async def test_create_indexes(self, neo4j_client: "Neo4jClient") -> None:
        """Test creating indexes."""
        # Should not raise
        await neo4j_client.create_indexes()

        # Verify indexes exist
        result = await neo4j_client.execute_query("SHOW INDEXES")
        assert len(result) > 0


class TestNeo4jClientNodeOperations:
    """Test node CRUD operations."""

    async def test_create_node(self, neo4j_client: "Neo4jClient") -> None:
        """Test creating a node."""
        node = await neo4j_client.create_node(
            labels=["VideoChunk"],
            properties={
                "chunk_id": "chunk_001",
                "stream_id": "test_stream",
                "chunk_idx": 0,
                "start_time": "00:00:00",
                "end_time": "00:01:00",
                "caption": "Test caption",
            }
        )

        assert node.id is not None
        assert "VideoChunk" in node.labels
        assert node.properties["chunk_id"] == "chunk_001"

    async def test_create_node_with_multiple_labels(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test creating a node with multiple labels."""
        node = await neo4j_client.create_node(
            labels=["Entity", "Person"],
            properties={
                "name": "Worker",
                "type": "PERSON",
            }
        )

        assert "Entity" in node.labels
        assert "Person" in node.labels

    async def test_get_node_by_id(self, neo4j_client: "Neo4jClient") -> None:
        """Test retrieving a node by ID."""
        # Create a node
        created = await neo4j_client.create_node(
            labels=["VideoChunk"],
            properties={
                "id": "get_test_001",
                "chunk_id": "chunk_001",
                "stream_id": "test_stream",
            }
        )

        # Retrieve it
        retrieved = await neo4j_client.get_node_by_id("get_test_001")

        assert retrieved is not None
        assert retrieved.id == "get_test_001"
        assert retrieved.properties["chunk_id"] == "chunk_001"

    async def test_get_node_by_id_not_found(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test retrieving a non-existent node."""
        result = await neo4j_client.get_node_by_id("nonexistent_id")

        assert result is None

    async def test_find_nodes_by_label(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test finding nodes by label."""
        # Create multiple nodes
        for i in range(5):
            await neo4j_client.create_node(
                labels=["Entity"],
                properties={
                    "entity_id": f"entity_{i:03d}",
                    "name": f"Entity {i}",
                    "type": "OBJECT",
                }
            )

        # Find all Entity nodes
        nodes = await neo4j_client.find_nodes("Entity")

        assert len(nodes) == 5

    async def test_find_nodes_with_properties(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test finding nodes with property filters."""
        # Create nodes with different types
        await neo4j_client.create_node(
            labels=["Entity"],
            properties={"name": "Forklift", "type": "VEHICLE"}
        )
        await neo4j_client.create_node(
            labels=["Entity"],
            properties={"name": "Worker", "type": "PERSON"}
        )
        await neo4j_client.create_node(
            labels=["Entity"],
            properties={"name": "Truck", "type": "VEHICLE"}
        )

        # Find only VEHICLE entities
        vehicles = await neo4j_client.find_nodes(
            "Entity",
            properties={"type": "VEHICLE"}
        )

        assert len(vehicles) == 2
        assert all(n.properties["type"] == "VEHICLE" for n in vehicles)

    async def test_delete_node(self, neo4j_client: "Neo4jClient") -> None:
        """Test deleting a node."""
        # Create a node
        await neo4j_client.create_node(
            labels=["Entity"],
            properties={"id": "delete_test_001", "name": "Test"}
        )

        # Verify it exists
        assert await neo4j_client.get_node_by_id("delete_test_001") is not None

        # Delete it
        result = await neo4j_client.delete_node("delete_test_001")
        assert result is True

        # Verify it's gone
        assert await neo4j_client.get_node_by_id("delete_test_001") is None


class TestNeo4jClientRelationships:
    """Test relationship operations."""

    async def test_create_relationship(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test creating a relationship between nodes."""
        # Create two nodes
        chunk = await neo4j_client.create_node(
            labels=["VideoChunk"],
            properties={"id": "chunk_rel_001", "chunk_idx": 0}
        )
        entity = await neo4j_client.create_node(
            labels=["Entity"],
            properties={"id": "entity_rel_001", "name": "Forklift"}
        )

        # Create relationship
        rel = await neo4j_client.create_relationship(
            start_node_id="chunk_rel_001",
            end_node_id="entity_rel_001",
            rel_type="CONTAINS",
            properties={"confidence": 0.95}
        )

        assert rel.id is not None
        assert rel.type == "CONTAINS"
        assert rel.properties["confidence"] == 0.95

    async def test_create_follows_relationship(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test creating FOLLOWS relationship between chunks."""
        # Create sequential chunks
        chunk1 = await neo4j_client.create_node(
            labels=["VideoChunk"],
            properties={"id": "chunk_seq_001", "chunk_idx": 0}
        )
        chunk2 = await neo4j_client.create_node(
            labels=["VideoChunk"],
            properties={"id": "chunk_seq_002", "chunk_idx": 1}
        )

        # Create FOLLOWS relationship
        rel = await neo4j_client.create_relationship(
            start_node_id="chunk_seq_001",
            end_node_id="chunk_seq_002",
            rel_type="FOLLOWS"
        )

        assert rel.type == "FOLLOWS"

    async def test_get_neighbors_outgoing(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test getting outgoing neighbors."""
        # Create chunk with entities
        await neo4j_client.create_node(
            labels=["VideoChunk"],
            properties={"id": "chunk_neighbors_001", "chunk_idx": 0}
        )
        await neo4j_client.create_node(
            labels=["Entity"],
            properties={"id": "entity_neighbors_001", "name": "Forklift"}
        )
        await neo4j_client.create_node(
            labels=["Entity"],
            properties={"id": "entity_neighbors_002", "name": "Worker"}
        )

        # Create relationships
        await neo4j_client.create_relationship(
            "chunk_neighbors_001", "entity_neighbors_001", "CONTAINS"
        )
        await neo4j_client.create_relationship(
            "chunk_neighbors_001", "entity_neighbors_002", "CONTAINS"
        )

        # Get outgoing neighbors
        neighbors = await neo4j_client.get_neighbors(
            "chunk_neighbors_001",
            rel_type="CONTAINS",
            direction="out"
        )

        assert len(neighbors) == 2
        entity_names = [n[1].properties["name"] for n in neighbors]
        assert "Forklift" in entity_names
        assert "Worker" in entity_names

    async def test_get_neighbors_incoming(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test getting incoming neighbors."""
        # Create entity contained in multiple chunks
        await neo4j_client.create_node(
            labels=["Entity"],
            properties={"id": "entity_incoming_001", "name": "Forklift"}
        )
        await neo4j_client.create_node(
            labels=["VideoChunk"],
            properties={"id": "chunk_incoming_001", "chunk_idx": 0}
        )
        await neo4j_client.create_node(
            labels=["VideoChunk"],
            properties={"id": "chunk_incoming_002", "chunk_idx": 1}
        )

        # Create relationships
        await neo4j_client.create_relationship(
            "chunk_incoming_001", "entity_incoming_001", "CONTAINS"
        )
        await neo4j_client.create_relationship(
            "chunk_incoming_002", "entity_incoming_001", "CONTAINS"
        )

        # Get incoming neighbors (chunks that contain this entity)
        neighbors = await neo4j_client.get_neighbors(
            "entity_incoming_001",
            rel_type="CONTAINS",
            direction="in"
        )

        assert len(neighbors) == 2


class TestNeo4jClientStreamOperations:
    """Test stream-level operations."""

    async def test_delete_by_stream(self, neo4j_client: "Neo4jClient") -> None:
        """Test deleting all nodes for a stream."""
        # Create nodes for target stream
        for i in range(5):
            await neo4j_client.create_node(
                labels=["VideoChunk"],
                properties={
                    "id": f"stream_del_{i:03d}",
                    "stream_id": "delete_stream",
                    "chunk_idx": i,
                }
            )

        # Create node in different stream
        await neo4j_client.create_node(
            labels=["VideoChunk"],
            properties={
                "id": "other_stream_001",
                "stream_id": "other_stream",
                "chunk_idx": 0,
            }
        )

        # Delete by stream
        deleted = await neo4j_client.delete_by_stream("delete_stream")

        assert deleted == 5

        # Verify other stream node still exists
        other = await neo4j_client.get_node_by_id("other_stream_001")
        assert other is not None

    async def test_clear_database(self, neo4j_client: "Neo4jClient") -> None:
        """Test clearing entire database."""
        # Create some nodes
        for i in range(10):
            await neo4j_client.create_node(
                labels=["VideoChunk"],
                properties={"id": f"clear_test_{i:03d}"}
            )

        # Clear database
        deleted = await neo4j_client.clear_database()

        assert deleted == 10

        # Verify database is empty
        result = await neo4j_client.execute_query("MATCH (n) RETURN count(n) as count")
        assert result[0]["count"] == 0


class TestNeo4jClientGraphTraversal:
    """Test graph traversal operations."""

    async def test_traverse_temporal_chain(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test traversing temporal FOLLOWS chain."""
        # Create chain of chunks
        chunk_ids = []
        for i in range(5):
            chunk = await neo4j_client.create_node(
                labels=["VideoChunk"],
                properties={
                    "id": f"chain_{i:03d}",
                    "stream_id": "chain_stream",
                    "chunk_idx": i,
                    "start_time": f"00:{i:02d}:00",
                }
            )
            chunk_ids.append(f"chain_{i:03d}")

        # Create FOLLOWS relationships
        for i in range(len(chunk_ids) - 1):
            await neo4j_client.create_relationship(
                chunk_ids[i],
                chunk_ids[i + 1],
                "FOLLOWS"
            )

        # Traverse from first chunk
        query = """
        MATCH path = (start:VideoChunk {id: $start_id})-[:FOLLOWS*]->(end:VideoChunk)
        RETURN length(path) as depth, end.id as end_id
        ORDER BY depth
        """
        results = await neo4j_client.execute_query(
            query,
            {"start_id": "chain_000"}
        )

        assert len(results) == 4  # 4 chunks after the first
        assert results[-1]["end_id"] == "chain_004"

    async def test_find_entities_in_chunk(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test finding all entities in a chunk."""
        # Create chunk with multiple entities
        await neo4j_client.create_node(
            labels=["VideoChunk"],
            properties={
                "id": "chunk_entities_001",
                "caption": "Forklift and worker in warehouse",
            }
        )

        entities = ["Forklift", "Worker", "Warehouse"]
        for i, name in enumerate(entities):
            await neo4j_client.create_node(
                labels=["Entity"],
                properties={"id": f"ent_{i:03d}", "name": name}
            )
            await neo4j_client.create_relationship(
                "chunk_entities_001",
                f"ent_{i:03d}",
                "CONTAINS"
            )

        # Query for entities in chunk
        query = """
        MATCH (c:VideoChunk {id: $chunk_id})-[:CONTAINS]->(e:Entity)
        RETURN e.name as name
        """
        results = await neo4j_client.execute_query(
            query,
            {"chunk_id": "chunk_entities_001"}
        )

        names = [r["name"] for r in results]
        assert len(names) == 3
        assert set(names) == {"Forklift", "Worker", "Warehouse"}

    async def test_find_entity_across_chunks(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test finding an entity across multiple chunks."""
        # Create entity
        await neo4j_client.create_node(
            labels=["Entity"],
            properties={"id": "forklift_001", "name": "Red Forklift"}
        )

        # Create chunks that contain the entity
        for i in range(3):
            await neo4j_client.create_node(
                labels=["VideoChunk"],
                properties={
                    "id": f"chunk_forklift_{i:03d}",
                    "chunk_idx": i,
                    "start_time": f"00:{i:02d}:00",
                }
            )
            await neo4j_client.create_relationship(
                f"chunk_forklift_{i:03d}",
                "forklift_001",
                "CONTAINS"
            )

        # Query for chunks containing the entity
        query = """
        MATCH (c:VideoChunk)-[:CONTAINS]->(e:Entity {name: $entity_name})
        RETURN c.id as chunk_id, c.start_time as start_time
        ORDER BY c.chunk_idx
        """
        results = await neo4j_client.execute_query(
            query,
            {"entity_name": "Red Forklift"}
        )

        assert len(results) == 3
        assert results[0]["start_time"] == "00:00:00"
        assert results[2]["start_time"] == "00:02:00"


class TestNeo4jClientWithGraphIngestion:
    """Integration tests with GraphIngestion component."""

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
    async def test_full_ingestion_workflow(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test full graph ingestion workflow."""
        from src.db.graph_ingestion import GraphIngestion
        from src.models.gemini.gemini_llm import GeminiLLM
        from src.models.gemini.gemini_embeddings import GeminiEmbeddings

        api_key = os.environ["GEMINI_API_KEY"]
        llm = GeminiLLM(api_key=api_key)
        embeddings = GeminiEmbeddings(api_key=api_key)

        ingestion = GraphIngestion(
            neo4j_client=neo4j_client,
            llm=llm,
            embeddings=embeddings,
        )

        # Ingest a chunk
        result = await ingestion.ingest_chunk(
            chunk_id="ingestion_test_001",
            stream_id="ingestion_stream",
            chunk_idx=0,
            caption="""
            00:00:05-00:00:15: A yellow forklift enters the warehouse.
            00:00:15-00:00:30: A worker in a blue vest approaches the forklift.
            00:00:30-00:00:45: The forklift picks up a pallet of boxes.
            """,
            start_time="00:00:00",
            end_time="00:01:00",
            embedding_id="milvus_001",
        )

        assert result is not None
        assert result.get("chunk_created", False)

        # Verify chunk was created
        chunk = await neo4j_client.get_node_by_id("ingestion_test_001")
        assert chunk is not None
        assert "VideoChunk" in chunk.labels

        # Verify entities were extracted
        entities = await neo4j_client.find_nodes("Entity")
        assert len(entities) > 0

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
    async def test_entity_extraction(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test entity extraction from captions."""
        from src.db.graph_ingestion import GraphIngestion
        from src.models.gemini.gemini_llm import GeminiLLM
        from src.models.gemini.gemini_embeddings import GeminiEmbeddings

        api_key = os.environ["GEMINI_API_KEY"]
        llm = GeminiLLM(api_key=api_key)
        embeddings = GeminiEmbeddings(api_key=api_key)

        ingestion = GraphIngestion(
            neo4j_client=neo4j_client,
            llm=llm,
            embeddings=embeddings,
        )

        caption = "A red forklift carrying pallets moves toward the loading dock."

        entities = await ingestion.extract_entities(caption)

        assert len(entities) > 0
        entity_names = [e.name.lower() for e in entities]
        # Should extract forklift-related entity
        assert any("forklift" in name for name in entity_names)


class TestNeo4jClientPerformance:
    """Performance tests for Neo4j operations."""

    async def test_bulk_node_creation(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test performance of bulk node creation."""
        import time

        start_time = time.time()

        # Create 100 nodes
        for i in range(100):
            await neo4j_client.create_node(
                labels=["VideoChunk"],
                properties={
                    "id": f"perf_{i:04d}",
                    "stream_id": "perf_stream",
                    "chunk_idx": i,
                    "caption": f"Caption for chunk {i}",
                }
            )

        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 30 seconds)
        assert elapsed < 30

        # Verify all nodes were created
        nodes = await neo4j_client.find_nodes(
            "VideoChunk",
            properties={"stream_id": "perf_stream"},
            limit=200
        )
        assert len(nodes) == 100

    async def test_relationship_creation_performance(
        self,
        neo4j_client: "Neo4jClient"
    ) -> None:
        """Test performance of relationship creation."""
        import time

        # Create nodes first
        node_ids = []
        for i in range(50):
            await neo4j_client.create_node(
                labels=["VideoChunk"],
                properties={"id": f"rel_perf_{i:04d}", "chunk_idx": i}
            )
            node_ids.append(f"rel_perf_{i:04d}")

        start_time = time.time()

        # Create chain of relationships
        for i in range(len(node_ids) - 1):
            await neo4j_client.create_relationship(
                node_ids[i],
                node_ids[i + 1],
                "FOLLOWS"
            )

        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 15 seconds)
        assert elapsed < 15
