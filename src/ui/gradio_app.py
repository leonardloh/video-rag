"""Gradio UI for Video Search and Summarization (VSS) PoC."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import gradio as gr

logger = logging.getLogger(__name__)

# Global state
_handler: Optional[Any] = None
_current_request: Optional[Any] = None
_config: dict[str, Any] = {}


def get_config() -> dict[str, Any]:
    """Load configuration."""
    global _config
    if not _config:
        try:
            from ..config import load_config

            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
            if config_path.exists():
                _config = load_config(str(config_path))
            else:
                _config = {}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            _config = {}
    return _config


async def initialize_handler(
    enable_cv: bool = False,
    enable_milvus: bool = False,
    enable_neo4j: bool = False,
) -> Any:
    """Initialize the ViaStreamHandler."""
    from ..asset_manager import AssetManager
    from ..models.gemini.gemini_embeddings import GeminiEmbeddings
    from ..models.gemini.gemini_file_manager import GeminiFileManager
    from ..models.gemini.gemini_llm import GeminiLLM
    from ..models.gemini.gemini_vlm import GeminiVLM
    from ..via_stream_handler import ViaStreamHandler

    config = get_config()

    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    # Initialize Gemini components
    gemini_config = config.get("gemini", {})

    file_manager = GeminiFileManager(
        api_key=api_key,
        max_concurrent_uploads=config.get("processing", {}).get("max_concurrent_chunks", 3),
    )

    vlm = GeminiVLM(
        api_key=api_key,
        model=gemini_config.get("vlm", {}).get("model", "gemini-2.0-flash"),
    )

    llm = GeminiLLM(
        api_key=api_key,
        model=gemini_config.get("llm", {}).get("model", "gemini-2.0-flash"),
    )

    embeddings = GeminiEmbeddings(
        api_key=api_key,
        model=gemini_config.get("embeddings", {}).get("model", "text-embedding-004"),
    )

    # Initialize asset manager
    asset_manager = AssetManager()

    # Initialize optional components
    yolo_pipeline = None
    milvus_client = None
    neo4j_client = None

    if enable_cv:
        try:
            from ..cv_pipeline.yolo_pipeline import YOLOPipeline

            cv_config = config.get("cv_pipeline", {}).get("yolo", {})
            yolo_pipeline = YOLOPipeline(
                model=cv_config.get("model", os.environ.get("YOLO_MODEL", "yolov8n-seg")),
                confidence=cv_config.get(
                    "confidence", float(os.environ.get("YOLO_CONFIDENCE", "0.5"))
                ),
                device=cv_config.get("device", os.environ.get("YOLO_DEVICE", "cpu")),
            )
            logger.info("YOLO pipeline initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO pipeline: {e}")

    if enable_milvus:
        try:
            from ..db.milvus_client import MilvusClient, MilvusConfig

            milvus_cfg = config.get("rag", {}).get("vector_db", {})
            milvus_config_obj = MilvusConfig(
                host=milvus_cfg.get("host", os.environ.get("MILVUS_HOST", "localhost")),
                port=milvus_cfg.get(
                    "port", int(os.environ.get("MILVUS_PORT", "19530"))
                ),
                collection_name=milvus_cfg.get("collection_name", "vss_poc_captions"),
            )
            milvus_client = MilvusClient(config=milvus_config_obj)
            logger.info("Milvus client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Milvus client: {e}")

    if enable_neo4j:
        try:
            from ..db.neo4j_client import Neo4jClient, Neo4jConfig

            neo4j_cfg = config.get("rag", {}).get("graph_db", {})
            neo4j_config_obj = Neo4jConfig(
                host=neo4j_cfg.get("host", os.environ.get("NEO4J_HOST", "localhost")),
                port=neo4j_cfg.get(
                    "port", int(os.environ.get("NEO4J_BOLT_PORT", "7687"))
                ),
                username=neo4j_cfg.get(
                    "username", os.environ.get("NEO4J_USERNAME", "neo4j")
                ),
                password=neo4j_cfg.get(
                    "password", os.environ.get("NEO4J_PASSWORD", "")
                ),
            )
            neo4j_client = Neo4jClient(config=neo4j_config_obj)
            logger.info("Neo4j client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Neo4j client: {e}")

    return ViaStreamHandler(
        asset_manager=asset_manager,
        gemini_file_manager=file_manager,
        gemini_vlm=vlm,
        gemini_llm=llm,
        gemini_embeddings=embeddings,
        yolo_pipeline=yolo_pipeline,
        milvus_client=milvus_client,
        neo4j_client=neo4j_client,
        config=config,
    )


def get_default_prompt() -> str:
    """Get default VLM prompt."""
    prompt_path = Path(__file__).parent.parent.parent / "config" / "prompts" / "caption.txt"
    if prompt_path.exists():
        with open(prompt_path) as f:
            return f.read().strip()
    return """Analyze this video segment and provide detailed captions of all events.
For each distinct event or action, provide:
- Start and end timestamps in format <HH:MM:SS>
- Detailed description of what is happening
- Any relevant objects, people, or text visible
Focus on actions, interactions, and notable occurrences."""


def get_chat_prompt() -> str:
    """Get chat system prompt."""
    prompt_path = Path(__file__).parent.parent.parent / "config" / "prompts" / "chat.txt"
    if prompt_path.exists():
        with open(prompt_path) as f:
            return f.read().strip()
    return """You are a video analysis assistant. You have access to detailed captions,
object detection data, and a knowledge graph of entities and events from a video.
Answer questions accurately based on the provided context.
If information is not available in the context, say so clearly.
Always reference specific timestamps when relevant."""


def check_db_status() -> tuple[str, str]:
    """Check database connection status."""
    milvus_status = "Not configured"
    neo4j_status = "Not configured"

    # Check Milvus
    milvus_host = os.environ.get("MILVUS_HOST")
    if milvus_host:
        try:
            from pymilvus import connections

            connections.connect(
                alias="status_check",
                host=milvus_host,
                port=int(os.environ.get("MILVUS_PORT", "19530")),
            )
            connections.disconnect("status_check")
            milvus_status = f"Connected ({milvus_host})"
        except Exception as e:
            milvus_status = f"Error: {e}"

    # Check Neo4j
    neo4j_host = os.environ.get("NEO4J_HOST")
    if neo4j_host:
        try:
            from neo4j import GraphDatabase

            uri = f"bolt://{neo4j_host}:{os.environ.get('NEO4J_BOLT_PORT', '7687')}"
            driver = GraphDatabase.driver(
                uri,
                auth=(
                    os.environ.get("NEO4J_USERNAME", "neo4j"),
                    os.environ.get("NEO4J_PASSWORD", ""),
                ),
            )
            driver.verify_connectivity()
            driver.close()
            neo4j_status = f"Connected ({neo4j_host})"
        except Exception as e:
            neo4j_status = f"Error: {e}"

    return milvus_status, neo4j_status


async def process_video_async(
    video_path: str,
    vlm_prompt: str,
    chunk_duration: int,
    enable_cv: bool,
    enable_milvus: bool,
    enable_neo4j: bool,
    progress: gr.Progress,
) -> tuple[str, str, str]:
    """Process video asynchronously."""
    global _handler, _current_request

    if not video_path:
        return "No video uploaded", "", ""

    try:
        # Initialize handler
        progress(0, desc="Initializing components...")
        _handler = await initialize_handler(
            enable_cv=enable_cv,
            enable_milvus=enable_milvus,
            enable_neo4j=enable_neo4j,
        )

        # Process video
        captions_text = ""

        def on_chunk_complete(result: Any) -> None:
            nonlocal captions_text
            if result.vlm_response:
                captions_text += f"\n\n### Chunk {result.chunk.chunkIdx} [{result.chunk.start_ntp} - {result.chunk.end_ntp}]\n"
                captions_text += result.vlm_response
                if result.cv_metadata:
                    captions_text += f"\n\n**CV Detection:** {json.dumps(result.cv_metadata.get('class_counts', {}))}"

        def on_progress(prog: float) -> None:
            progress(prog / 100, desc=f"Processing video... {prog:.1f}%")

        progress(0.05, desc="Processing video...")
        _current_request = await _handler.process_video(
            file_path=video_path,
            vlm_prompt=vlm_prompt,
            chunk_duration=chunk_duration,
            chunk_overlap=2,
            enable_cv_pipeline=enable_cv,
            on_chunk_complete=on_chunk_complete,
            on_progress=on_progress,
        )

        status = f"Status: {_current_request.status.value}\n"
        status += f"Chunks processed: {len(_current_request.processed_chunks)}/{_current_request.chunk_count}\n"
        status += f"Request ID: {_current_request.request_id}"

        # Build result JSON
        result_json = {
            "request_id": _current_request.request_id,
            "stream_id": _current_request.stream_id,
            "status": _current_request.status.value,
            "chunk_count": _current_request.chunk_count,
            "captions": [
                {
                    "chunk_idx": r.chunk.chunkIdx,
                    "start_time": r.chunk.start_ntp,
                    "end_time": r.chunk.end_ntp,
                    "caption": r.vlm_response,
                    "cv_metadata": r.cv_metadata,
                }
                for r in _current_request.processed_chunks
                if r.vlm_response
            ],
        }

        return status, captions_text.strip(), json.dumps(result_json, indent=2)

    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        return f"Error: {e}", "", ""


def process_video(
    video: str,
    vlm_prompt: str,
    chunk_duration: int,
    enable_cv: bool,
    enable_milvus: bool,
    enable_neo4j: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str]:
    """Process video (sync wrapper)."""
    return asyncio.run(
        process_video_async(
            video_path=video,
            vlm_prompt=vlm_prompt,
            chunk_duration=chunk_duration,
            enable_cv=enable_cv,
            enable_milvus=enable_milvus,
            enable_neo4j=enable_neo4j,
            progress=progress,
        )
    )


async def generate_summary_async() -> str:
    """Generate summary asynchronously."""
    global _handler, _current_request

    if not _handler or not _current_request:
        return "No video has been processed yet. Please process a video first."

    try:
        summary = await _handler.summarize(
            request_id=_current_request.request_id,
            batch_size=6,
        )
        return summary
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return f"Error generating summary: {e}"


def generate_summary() -> str:
    """Generate summary (sync wrapper)."""
    return asyncio.run(generate_summary_async())


async def chat_async(
    message: str,
    history: list[tuple[str, str]],
) -> str:
    """Chat about the video asynchronously."""
    global _handler, _current_request

    if not _handler or not _current_request:
        return "No video has been processed yet. Please process a video first."

    if not message.strip():
        return ""

    try:
        # Convert history to expected format
        chat_history = []
        for user_msg, assistant_msg in history:
            chat_history.append({"role": "user", "content": user_msg})
            chat_history.append({"role": "assistant", "content": assistant_msg})

        answer = await _handler.chat(
            request_id=_current_request.request_id,
            question=message,
            chat_history=chat_history if chat_history else None,
            system_prompt=get_chat_prompt(),
        )
        return answer
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return f"Error: {e}"


def chat(message: str, history: list[tuple[str, str]]) -> str:
    """Chat about the video (sync wrapper)."""
    return asyncio.run(chat_async(message, history))


def export_results() -> Optional[str]:
    """Export results as JSON file."""
    global _current_request

    if not _current_request:
        return None

    result = {
        "request_id": _current_request.request_id,
        "stream_id": _current_request.stream_id,
        "status": _current_request.status.value,
        "chunk_count": _current_request.chunk_count,
        "captions": [
            {
                "chunk_idx": r.chunk.chunkIdx,
                "start_time": r.chunk.start_ntp,
                "end_time": r.chunk.end_ntp,
                "caption": r.vlm_response,
                "cv_metadata": r.cv_metadata,
            }
            for r in _current_request.processed_chunks
            if r.vlm_response
        ],
    }

    # Save to temp file
    output_path = Path("/tmp") / f"vss_results_{_current_request.request_id}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return str(output_path)


def refresh_db_status() -> tuple[str, str]:
    """Refresh database status."""
    return check_db_status()


def create_app() -> gr.Blocks:
    """Create the Gradio application."""
    with gr.Blocks(
        title="VSS PoC - Video Search and Summarization",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # Video Search and Summarization (VSS) PoC

            Upload a video to analyze it using Gemini VLM, generate captions, and ask questions about the content.
            """
        )

        with gr.Tabs():
            # Tab 1: Video Processing
            with gr.TabItem("Process Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Upload Video",
                            sources=["upload"],
                        )

                        with gr.Accordion("Configuration", open=False):
                            vlm_prompt = gr.Textbox(
                                label="VLM Prompt",
                                value=get_default_prompt(),
                                lines=5,
                            )
                            chunk_duration = gr.Slider(
                                label="Chunk Duration (seconds)",
                                minimum=10,
                                maximum=300,
                                value=60,
                                step=10,
                            )

                        with gr.Accordion("Options", open=True):
                            enable_cv = gr.Checkbox(
                                label="Enable CV Pipeline (YOLO)",
                                value=False,
                            )
                            enable_milvus = gr.Checkbox(
                                label="Enable Milvus (Vector DB)",
                                value=False,
                            )
                            enable_neo4j = gr.Checkbox(
                                label="Enable Neo4j (Graph DB)",
                                value=False,
                            )

                        process_btn = gr.Button("Process Video", variant="primary")

                    with gr.Column(scale=2):
                        status_output = gr.Textbox(
                            label="Processing Status",
                            lines=3,
                            interactive=False,
                        )
                        captions_output = gr.Markdown(
                            label="Captions",
                        )

                process_btn.click(
                    fn=process_video,
                    inputs=[
                        video_input,
                        vlm_prompt,
                        chunk_duration,
                        enable_cv,
                        enable_milvus,
                        enable_neo4j,
                    ],
                    outputs=[status_output, captions_output, gr.State()],
                )

            # Tab 2: Summary
            with gr.TabItem("Summary"):
                summary_btn = gr.Button("Generate Summary", variant="primary")
                summary_output = gr.Markdown(label="Video Summary")

                summary_btn.click(
                    fn=generate_summary,
                    inputs=[],
                    outputs=[summary_output],
                )

            # Tab 3: Chat
            with gr.TabItem("Chat"):
                gr.Markdown(
                    """
                    Ask questions about the processed video. The assistant will use the
                    generated captions and context to answer your questions.
                    """
                )
                chatbot = gr.ChatInterface(
                    fn=chat,
                    examples=[
                        "What happens in the video?",
                        "Are there any people in the video?",
                        "What objects are visible?",
                        "Describe the main events in chronological order.",
                    ],
                )

            # Tab 4: Export
            with gr.TabItem("Export"):
                gr.Markdown("Export the processing results as a JSON file.")
                export_btn = gr.Button("Export Results", variant="primary")
                export_file = gr.File(label="Download Results")

                export_btn.click(
                    fn=export_results,
                    inputs=[],
                    outputs=[export_file],
                )

            # Tab 5: Status
            with gr.TabItem("Status"):
                gr.Markdown("## Database Connection Status")

                with gr.Row():
                    milvus_status = gr.Textbox(
                        label="Milvus Status",
                        interactive=False,
                    )
                    neo4j_status = gr.Textbox(
                        label="Neo4j Status",
                        interactive=False,
                    )

                refresh_btn = gr.Button("Refresh Status")
                refresh_btn.click(
                    fn=refresh_db_status,
                    inputs=[],
                    outputs=[milvus_status, neo4j_status],
                )

                # Initial status check
                app.load(
                    fn=refresh_db_status,
                    inputs=[],
                    outputs=[milvus_status, neo4j_status],
                )

                gr.Markdown(
                    """
                    ## Environment Variables

                    | Variable | Value |
                    |----------|-------|
                    | GEMINI_API_KEY | {"Set" if os.environ.get("GEMINI_API_KEY") else "Not set"} |
                    | MILVUS_HOST | {os.environ.get("MILVUS_HOST", "Not set")} |
                    | NEO4J_HOST | {os.environ.get("NEO4J_HOST", "Not set")} |
                    | YOLO_DEVICE | {os.environ.get("YOLO_DEVICE", "cpu")} |
                    """
                )

    return app


def main() -> None:
    """Run the Gradio application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = create_app()
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )


if __name__ == "__main__":
    main()
