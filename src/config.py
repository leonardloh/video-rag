"""Configuration loader for VSS PoC."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


def _env_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    """Custom YAML constructor for !ENV tag."""
    value = loader.construct_scalar(node)

    # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
    pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"
    match = re.match(pattern, value)

    if match:
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default or "")

    return value


def _include_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    """Custom YAML constructor for !include tag."""
    filename = loader.construct_scalar(node)
    base_path = Path(loader.stream.name).parent if hasattr(loader.stream, "name") else Path.cwd()
    file_path = base_path / filename

    if file_path.exists():
        return file_path.read_text()
    return ""


# Register custom constructors
yaml.SafeLoader.add_constructor("!ENV", _env_constructor)
yaml.SafeLoader.add_constructor("!include", _include_constructor)


def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file (uses default if None)

    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()

    # Default config path
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config or {}


def get_prompt(prompt_name: str, prompts_dir: Optional[str] = None) -> str:
    """
    Load a prompt from file.

    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        prompts_dir: Directory containing prompts (uses default if None)

    Returns:
        Prompt text
    """
    if prompts_dir is None:
        prompts_dir = str(Path(__file__).parent.parent / "config" / "prompts")

    prompt_path = Path(prompts_dir) / f"{prompt_name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text()


def validate_config(config: dict[str, Any]) -> list[str]:
    """
    Validate configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required Gemini config
    gemini = config.get("gemini", {})
    if not gemini.get("api_key"):
        errors.append("Missing gemini.api_key (set GEMINI_API_KEY environment variable)")

    # Check RAG config
    rag = config.get("rag", {})
    if rag.get("enabled", False):
        vector_db = rag.get("vector_db", {})
        if vector_db.get("enabled", False):
            if not vector_db.get("host"):
                errors.append("Missing rag.vector_db.host")

        graph_db = rag.get("graph_db", {})
        if graph_db.get("enabled", False):
            if not graph_db.get("host"):
                errors.append("Missing rag.graph_db.host")
            if not graph_db.get("password"):
                errors.append("Missing rag.graph_db.password")

    return errors


def build_tools_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Build tools configuration from main config.

    Args:
        config: Main configuration dictionary

    Returns:
        Tools configuration dictionary
    """
    return config.get("tools", {})


def build_functions_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Build functions configuration from main config.

    Args:
        config: Main configuration dictionary

    Returns:
        Functions configuration dictionary
    """
    return config.get("functions", )


def get_gemini_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Get Gemini-specific configuration.

    Args:
        config: Main configuration dictionary

    Returns:
        Gemini configuration
    """
    return config.get("gemini", {})


def get_cv_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Get CV pipeline configuration.

    Args:
        config: Main configuration dictionary

    Returns:
        CV pipeline configuration
    """
    return config.get("cv_pipeline", {})


def get_rag_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Get RAG configuration.

    Args:
        config: Main configuration dictionary

    Returns:
        RAG configuration
    """
    return config.get("rag", {})


def get_processing_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Get processing configuration.

    Args:
        config: Main configuration dictionary

    Returns:
        Processing configuration
    """
    return config.get("processing", {})
