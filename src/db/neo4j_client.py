"""Neo4j graph database client for entity and relationship management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""

    host: str = "localhost"
    port: int = 7687
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    @property
    def uri(self) -> str:
        """Get connection URI."""
        return f"bolt://{self.host}:{self.port}"


@dataclass
class GraphNode:
    """A node in the graph."""

    id: str
    labels: list[str]
    properties: dict[str, Any]


@dataclass
class GraphRelationship:
    """A relationship between nodes."""

    id: str
    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


class Neo4jClient:
    """Client for Neo4j graph database operations."""

    def __init__(self, config: Neo4jConfig) -> None:
        """
        Initialize Neo4j client.

        Args:
            config: Neo4j connection configuration
        """
        self._config = config
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self._config.uri,
            auth=(self._config.username, self._config.password),
        )
        # Verify connectivity
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        """Close connection to Neo4j."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    def _get_driver(self) -> AsyncDriver:
        """Get driver, raising error if not connected."""
        if self._driver is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._driver

    async def execute_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Query results as list of dictionaries
        """
        driver = self._get_driver()

        async with driver.session(database=self._config.database) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def create_node(
        self,
        labels: list[str],
        properties: dict[str, Any],
    ) -> GraphNode:
        """
        Create a new node.

        Args:
            labels: Node labels (e.g., ["VideoChunk"])
            properties: Node properties

        Returns:
            Created node
        """
        # Generate ID if not provided
        if "id" not in properties:
            properties["id"] = str(uuid.uuid4())

        labels_str = ":".join(labels)
        query = f"""
        CREATE (n:{labels_str} $props)
        RETURN n, elementId(n) as element_id
        """

        results = await self.execute_query(query, {"props": properties})

        if not results:
            raise RuntimeError("Failed to create node")

        node_data = results[0]["n"]
        return GraphNode(
            id=properties["id"],
            labels=labels,
            properties=dict(node_data),
        )

    async def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        rel_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> GraphRelationship:
        """
        Create a relationship between nodes.

        Args:
            start_node_id: Source node ID
            end_node_id: Target node ID
            rel_type: Relationship type (e.g., "FOLLOWS")
            properties: Relationship properties

        Returns:
            Created relationship
        """
        properties = properties or {}
        rel_id = str(uuid.uuid4())
        properties["id"] = rel_id

        query = f"""
        MATCH (a {{id: $start_id}}), (b {{id: $end_id}})
        CREATE (a)-[r:{rel_type} $props]->(b)
        RETURN r, elementId(r) as element_id
        """

        results = await self.execute_query(
            query,
            {
                "start_id": start_node_id,
                "end_id": end_node_id,
                "props": properties,
            },
        )

        if not results:
            raise RuntimeError(
                f"Failed to create relationship. Nodes may not exist: {start_node_id}, {end_node_id}"
            )

        return GraphRelationship(
            id=rel_id,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            type=rel_type,
            properties=properties,
        )

    async def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID."""
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) as labels
        """

        results = await self.execute_query(query, {"node_id": node_id})

        if not results:
            return None

        node_data = results[0]["n"]
        labels = results[0]["labels"]

        return GraphNode(
            id=node_id,
            labels=labels,
            properties=dict(node_data),
        )

    async def find_nodes(
        self,
        label: str,
        properties: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[GraphNode]:
        """
        Find nodes by label and properties.

        Args:
            label: Node label to search
            properties: Property filters
            limit: Maximum results

        Returns:
            Matching nodes
        """
        where_clause = ""
        params: dict[str, Any] = {"limit": limit}

        if properties:
            conditions = []
            for i, (key, value) in enumerate(properties.items()):
                param_name = f"prop_{i}"
                conditions.append(f"n.{key} = ${param_name}")
                params[param_name] = value
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        MATCH (n:{label})
        {where_clause}
        RETURN n, labels(n) as labels
        LIMIT $limit
        """

        results = await self.execute_query(query, params)

        nodes = []
        for r in results:
            node_data = r["n"]
            nodes.append(GraphNode(
                id=node_data.get("id", ""),
                labels=r["labels"],
                properties=dict(node_data),
            ))

        return nodes

    async def get_neighbors(
        self,
        node_id: str,
        rel_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 100,
    ) -> list[tuple[GraphRelationship, GraphNode]]:
        """
        Get neighboring nodes.

        Args:
            node_id: Starting node ID
            rel_type: Filter by relationship type
            direction: "in", "out", or "both"
            limit: Maximum results

        Returns:
            List of (relationship, neighbor_node) tuples
        """
        rel_pattern = f":{rel_type}" if rel_type else ""

        if direction == "out":
            query = f"""
            MATCH (n {{id: $node_id}})-[r{rel_pattern}]->(m)
            RETURN r, m, labels(m) as labels
            LIMIT $limit
            """
        elif direction == "in":
            query = f"""
            MATCH (n {{id: $node_id}})<-[r{rel_pattern}]-(m)
            RETURN r, m, labels(m) as labels
            LIMIT $limit
            """
        else:  # both
            query = f"""
            MATCH (n {{id: $node_id}})-[r{rel_pattern}]-(m)
            RETURN r, m, labels(m) as labels
            LIMIT $limit
            """

        results = await self.execute_query(query, {"node_id": node_id, "limit": limit})

        neighbors = []
        for r in results:
            rel_data = r["r"]
            node_data = r["m"]

            relationship = GraphRelationship(
                id=rel_data.get("id", ""),
                start_node_id=node_id,
                end_node_id=node_data.get("id", ""),
                type=rel_data.type,
                properties=dict(rel_data),
            )

            node = GraphNode(
                id=node_data.get("id", ""),
                labels=r["labels"],
                properties=dict(node_data),
            )

            neighbors.append((relationship, node))

        return neighbors

    async def delete_node(self, node_id: str, detach: bool = True) -> bool:
        """
        Delete a node.

        Args:
            node_id: Node to delete
            detach: Also delete relationships

        Returns:
            True if deleted
        """
        if detach:
            query = """
            MATCH (n {id: $node_id})
            DETACH DELETE n
            RETURN count(n) as deleted
            """
        else:
            query = """
            MATCH (n {id: $node_id})
            DELETE n
            RETURN count(n) as deleted
            """

        results = await self.execute_query(query, {"node_id": node_id})
        return results[0]["deleted"] > 0 if results else False

    async def delete_by_stream(self, stream_id: str) -> int:
        """Delete all nodes for a stream. Returns count deleted."""
        query = """
        MATCH (n {stream_id: $stream_id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """

        results = await self.execute_query(query, {"stream_id": stream_id})
        return results[0]["deleted"] if results else 0

    async def clear_database(self) -> int:
        """Clear all nodes and relationships. Returns count deleted."""
        query = """
        MATCH (n)
        DETACH DELETE n
        RETURN count(n) as deleted
        """

        results = await self.execute_query(query, {})
        return results[0]["deleted"] if results else 0

    async def create_indexes(self) -> None:
        """Create recommended indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:VideoChunk) ON (c.chunk_id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:VideoChunk) ON (c.stream_id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:VideoChunk) ON (c.chunk_idx)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.entity_id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX IF NOT EXISTS FOR (ev:Event) ON (ev.event_id)",
            "CREATE INDEX IF NOT EXISTS FOR (ev:Event) ON (ev.event_type)",
        ]

        for index_query in indexes:
            await self.execute_query(index_query)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique node/relationship ID."""
        return str(uuid.uuid4())
