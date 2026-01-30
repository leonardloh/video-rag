"""Integration tests for Gemini API components.

These tests require a valid GEMINI_API_KEY environment variable.
They test real API calls to verify the integration works correctly.

Run with: pytest tests/integration/test_gemini_integration.py -v
"""

from __future__ import annotations

import os
import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.gemini.gemini_embeddings import GeminiEmbeddings
    from src.models.gemini.gemini_llm import GeminiLLM


# Skip all tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not set"
)


@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key


@pytest.fixture
def embeddings(api_key: str) -> "GeminiEmbeddings":
    """Create GeminiEmbeddings instance."""
    from src.models.gemini.gemini_embeddings import GeminiEmbeddings
    return GeminiEmbeddings(api_key=api_key)


@pytest.fixture
def llm(api_key: str) -> "GeminiLLM":
    """Create GeminiLLM instance."""
    from src.models.gemini.gemini_llm import GeminiLLM
    return GeminiLLM(api_key=api_key)


class TestGeminiEmbeddingsIntegration:
    """Integration tests for GeminiEmbeddings."""

    async def test_embed_single_text(self, embeddings: "GeminiEmbeddings") -> None:
        """Test embedding a single text."""
        text = "A forklift moves across the warehouse floor."

        embedding = await embeddings.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768  # text-embedding-004 dimensions
        assert all(isinstance(x, float) for x in embedding)

    async def test_embed_query(self, embeddings: "GeminiEmbeddings") -> None:
        """Test embedding a search query."""
        query = "What is the forklift doing?"

        embedding = await embeddings.embed_query(query)

        assert isinstance(embedding, list)
        assert len(embedding) == 768

    async def test_embed_document(self, embeddings: "GeminiEmbeddings") -> None:
        """Test embedding a document."""
        document = """
        At 00:00:05, a forklift enters the frame from the left side.
        The operator is wearing a yellow safety vest.
        At 00:00:15, the forklift picks up a pallet of boxes.
        """

        embedding = await embeddings.embed_document(document)

        assert isinstance(embedding, list)
        assert len(embedding) == 768

    async def test_embed_batch(self, embeddings: "GeminiEmbeddings") -> None:
        """Test embedding multiple texts in batch."""
        texts = [
            "A worker walks across the warehouse.",
            "A forklift carries pallets.",
            "Safety cones mark the work area.",
        ]

        result = await embeddings.embed_batch(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        for emb in result:
            assert len(emb) == 768

    async def test_embed_documents(self, embeddings: "GeminiEmbeddings") -> None:
        """Test embedding multiple documents."""
        documents = [
            "Chunk 1: Worker enters the scene.",
            "Chunk 2: Forklift moves to loading dock.",
            "Chunk 3: Pallets are loaded onto truck.",
        ]

        result = await embeddings.embed_documents(documents)

        assert isinstance(result, list)
        assert len(result) == 3

    async def test_semantic_similarity(self, embeddings: "GeminiEmbeddings") -> None:
        """Test that semantically similar texts have high similarity."""
        text1 = "A forklift moves pallets in the warehouse."
        text2 = "A lift truck transports cargo in the storage facility."
        text3 = "The weather is sunny today."

        emb1 = await embeddings.embed_text(text1)
        emb2 = await embeddings.embed_text(text2)
        emb3 = await embeddings.embed_text(text3)

        # Similar texts should have higher similarity
        sim_similar = embeddings.cosine_similarity(emb1, emb2)
        sim_different = embeddings.cosine_similarity(emb1, emb3)

        assert sim_similar > sim_different
        assert sim_similar > 0.5  # Similar texts should have reasonable similarity

    async def test_query_document_retrieval(self, embeddings: "GeminiEmbeddings") -> None:
        """Test query-document retrieval scenario."""
        from src.models.gemini.gemini_embeddings import TaskType

        # Simulate documents
        documents = [
            "A forklift operator loads boxes onto a truck.",
            "Workers take a lunch break in the cafeteria.",
            "Security guard patrols the parking lot.",
        ]

        # Embed documents
        doc_embeddings = await embeddings.embed_batch(
            documents,
            task_type=TaskType.RETRIEVAL_DOCUMENT
        )

        # Embed query
        query = "forklift loading cargo"
        query_embedding = await embeddings.embed_text(
            query,
            task_type=TaskType.RETRIEVAL_QUERY
        )

        # Find most similar document
        similarities = [
            embeddings.cosine_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]

        # First document should be most similar to the query
        assert similarities[0] == max(similarities)


class TestGeminiLLMIntegration:
    """Integration tests for GeminiLLM."""

    async def test_generate_simple(self, llm: "GeminiLLM") -> None:
        """Test simple text generation."""
        prompt = "What is 2 + 2? Answer with just the number."

        result = await llm.generate(prompt)

        assert result.text is not None
        assert "4" in result.text
        assert result.usage.total_tokens > 0

    async def test_generate_with_system_prompt(self, llm: "GeminiLLM") -> None:
        """Test generation with system prompt."""
        system_prompt = "You are a helpful assistant that responds in JSON format."
        prompt = "List three colors."

        result = await llm.generate(prompt, system_prompt=system_prompt)

        assert result.text is not None
        # Should contain some JSON-like structure
        assert "{" in result.text or "[" in result.text

    async def test_chat_single_turn(self, llm: "GeminiLLM") -> None:
        """Test single-turn chat."""
        from src.models.gemini.gemini_llm import Message

        messages = [
            Message(role="user", content="Hello, how are you?")
        ]

        result = await llm.chat(messages)

        assert result.text is not None
        assert len(result.text) > 0

    async def test_chat_multi_turn(self, llm: "GeminiLLM") -> None:
        """Test multi-turn chat."""
        from src.models.gemini.gemini_llm import Message

        messages = [
            Message(role="user", content="My name is Alice."),
            Message(role="assistant", content="Hello Alice! Nice to meet you."),
            Message(role="user", content="What is my name?"),
        ]

        result = await llm.chat(messages)

        assert result.text is not None
        assert "Alice" in result.text

    async def test_summarize_captions(self, llm: "GeminiLLM") -> None:
        """Test caption summarization."""
        captions = [
            "00:00:00-00:01:00: A forklift enters the warehouse from the main entrance.",
            "00:01:00-00:02:00: The forklift picks up a pallet of boxes.",
            "00:02:00-00:03:00: The forklift moves to the loading dock.",
            "00:03:00-00:04:00: Workers help unload the pallet onto a truck.",
        ]

        prompt_template = """Summarize the following video captions into a brief paragraph:

{captions}

Summary:"""

        result = await llm.summarize_captions(captions, prompt_template)

        assert result is not None
        assert len(result) > 0
        # Should mention key elements
        assert any(word in result.lower() for word in ["forklift", "warehouse", "pallet", "truck"])

    async def test_aggregate_summaries(self, llm: "GeminiLLM") -> None:
        """Test summary aggregation."""
        summaries = [
            "In the first segment, a forklift enters and picks up cargo.",
            "In the second segment, workers load boxes onto a truck.",
            "In the third segment, the truck departs from the loading dock.",
        ]

        aggregation_prompt = """Combine the following summaries into a single coherent summary:

{summaries}

Combined summary:"""

        result = await llm.aggregate_summaries(summaries, aggregation_prompt)

        assert result is not None
        assert len(result) > 0

    async def test_check_notification_detected(self, llm: "GeminiLLM") -> None:
        """Test notification check when event is detected."""
        caption = "A person falls down near the forklift. Workers rush to help."
        events = ["fall", "accident", "injury"]
        notification_prompt = """Check if the following caption contains any of these events: {events}

Caption: {caption}

If any event is detected, respond with "DETECTED:" followed by the event name.
If no events are detected, respond with "NOT DETECTED"."""

        should_notify, detected, explanation = await llm.check_notification(
            caption, events, notification_prompt
        )

        # Should detect fall-related event
        assert should_notify is True
        assert len(detected) > 0

    async def test_check_notification_not_detected(self, llm: "GeminiLLM") -> None:
        """Test notification check when no event is detected."""
        caption = "A forklift moves pallets across the warehouse floor normally."
        events = ["fire", "explosion", "flood"]
        notification_prompt = """Check if the following caption contains any of these events: {events}

Caption: {caption}

If any event is detected, respond with "DETECTED:" followed by the event name.
If no events are detected, respond with "NOT DETECTED"."""

        should_notify, detected, explanation = await llm.check_notification(
            caption, events, notification_prompt
        )

        # Should not detect any events
        assert should_notify is False
        assert len(detected) == 0

    async def test_generation_config_override(self, llm: "GeminiLLM") -> None:
        """Test generation with custom config."""
        from src.models.gemini.gemini_llm import LLMGenerationConfig

        config = LLMGenerationConfig(
            temperature=0.1,
            max_output_tokens=50,
        )

        prompt = "Write a very short sentence about warehouses."

        result = await llm.generate(prompt, generation_config=config)

        assert result.text is not None
        # With low max_output_tokens, response should be short
        assert len(result.text) < 500


class TestGeminiEmbeddingsLLMIntegration:
    """Integration tests combining embeddings and LLM."""

    async def test_rag_workflow(
        self,
        embeddings: "GeminiEmbeddings",
        llm: "GeminiLLM"
    ) -> None:
        """Test a complete RAG workflow."""
        # Simulate video captions as documents
        documents = [
            "00:00:00-00:01:00: A red forklift enters the warehouse.",
            "00:01:00-00:02:00: The forklift picks up a pallet of electronics.",
            "00:02:00-00:03:00: A worker in a blue vest guides the forklift.",
            "00:03:00-00:04:00: The forklift moves to the shipping area.",
            "00:04:00-00:05:00: Boxes are loaded onto a delivery truck.",
        ]

        # Embed all documents
        doc_embeddings = await embeddings.embed_documents(documents)

        # User query
        query = "What color was the forklift?"
        query_embedding = await embeddings.embed_query(query)

        # Find most relevant documents
        similarities = [
            (i, embeddings.cosine_similarity(query_embedding, doc_emb))
            for i, doc_emb in enumerate(doc_embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top 2 relevant documents
        top_docs = [documents[i] for i, _ in similarities[:2]]

        # Generate answer using LLM
        context = "\n".join(top_docs)
        prompt = f"""Based on the following video captions, answer the question.

Captions:
{context}

Question: {query}

Answer:"""

        result = await llm.generate(prompt)

        assert result.text is not None
        assert "red" in result.text.lower()

    async def test_entity_extraction_workflow(
        self,
        embeddings: "GeminiEmbeddings",
        llm: "GeminiLLM"
    ) -> None:
        """Test entity extraction from video captions."""
        caption = """
        00:00:05-00:00:15: A yellow forklift enters from the east entrance.
        00:00:15-00:00:30: Worker John in a blue safety vest approaches.
        00:00:30-00:00:45: The forklift picks up a pallet of computer monitors.
        """

        prompt = """Extract all entities from the following video caption.
Return as a JSON array with objects containing "name", "type" (PERSON, VEHICLE, OBJECT, LOCATION).

Caption:
{caption}

Entities:""".format(caption=caption)

        result = await llm.generate(prompt)

        assert result.text is not None
        # Should extract key entities
        text_lower = result.text.lower()
        assert any(word in text_lower for word in ["forklift", "worker", "john", "pallet"])


class TestGeminiErrorHandling:
    """Test error handling for Gemini API."""

    async def test_empty_text_embedding(self, embeddings: "GeminiEmbeddings") -> None:
        """Test handling of empty text."""
        # Empty text should still work (API may return zeros or handle gracefully)
        try:
            result = await embeddings.embed_text("")
            assert isinstance(result, list)
        except Exception as e:
            # Some error is acceptable for empty input
            assert "empty" in str(e).lower() or "invalid" in str(e).lower()

    async def test_very_long_text_truncation(self, embeddings: "GeminiEmbeddings") -> None:
        """Test that very long text is truncated."""
        # Create text longer than MAX_TEXT_LENGTH
        long_text = "word " * 10000  # ~50000 characters

        # Should not raise, should truncate
        result = await embeddings.embed_text(long_text)

        assert isinstance(result, list)
        assert len(result) == 768

    async def test_llm_empty_prompt(self, llm: "GeminiLLM") -> None:
        """Test LLM with minimal prompt."""
        # Very short prompt should still work
        result = await llm.generate("Hi")

        assert result.text is not None


class TestGeminiRateLimiting:
    """Test rate limiting behavior."""

    async def test_concurrent_embeddings(self, embeddings: "GeminiEmbeddings") -> None:
        """Test concurrent embedding requests."""
        import asyncio

        texts = [f"Document number {i}" for i in range(5)]

        # Run concurrently
        tasks = [embeddings.embed_text(text) for text in texts]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert len(result) == 768

    async def test_batch_vs_individual(self, embeddings: "GeminiEmbeddings") -> None:
        """Compare batch embedding vs individual calls."""
        texts = [
            "First document about warehouses.",
            "Second document about forklifts.",
            "Third document about safety.",
        ]

        # Batch embedding
        batch_results = await embeddings.embed_batch(texts)

        # Individual embeddings
        individual_results = []
        for text in texts:
            emb = await embeddings.embed_text(text)
            individual_results.append(emb)

        # Results should be consistent (same dimensions)
        assert len(batch_results) == len(individual_results)
        for batch_emb, ind_emb in zip(batch_results, individual_results):
            assert len(batch_emb) == len(ind_emb) == 768
