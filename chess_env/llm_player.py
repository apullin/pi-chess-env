"""
LLM player wrapper for chess.

Provides a unified interface for LLM-based chess players using local inference.
Supports:
- Ollama for easy local model serving
- vLLM for high-performance inference with continuous batching

Usage:
    from chess_env.llm_player import create_player

    # Ollama (simple)
    player = create_player("ollama", model="qwen2.5:0.5b")

    # vLLM (high performance)
    player = create_player("vllm", model="meta-llama/Llama-3.1-8B-Instruct", base_url="http://localhost:8000/v1")

    response = player.generate(observation, system_prompt)
"""

from dataclasses import dataclass
from typing import Protocol, Optional
import re
import time


@dataclass
class LLMConfig:
    """Configuration for LLM player."""
    model: str = "qwen2.5:0.5b"
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 60.0
    # vLLM specific
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"  # vLLM doesn't require auth by default


class LLMPlayer(Protocol):
    """Protocol for LLM players."""

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate a response given a prompt and system prompt."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...


class OllamaPlayer:
    """
    Ollama-based LLM player.

    Requires ollama to be installed and running:
        brew install ollama
        ollama serve
        ollama pull qwen2.5:0.5b
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None
        self._last_response_time: float = 0.0

    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def last_response_time(self) -> float:
        """Time taken for last generation in seconds."""
        return self._last_response_time

    def _ensure_client(self):
        """Lazily initialize ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama
            except ImportError:
                raise ImportError(
                    "ollama package not installed. Install with: pip install ollama\n"
                    "Also ensure ollama is running: ollama serve"
                )

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        Generate a response using Ollama.

        Args:
            prompt: The user prompt (board observation)
            system_prompt: System instructions for the model

        Returns:
            Model's response text
        """
        self._ensure_client()

        start_time = time.time()

        try:
            response = self._client.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            )

            self._last_response_time = time.time() - start_time
            return response["message"]["content"]

        except Exception as e:
            self._last_response_time = time.time() - start_time
            raise RuntimeError(f"Ollama generation failed: {e}")

    def is_available(self) -> bool:
        """Check if the model is available."""
        self._ensure_client()
        try:
            response = self._client.list()
            # Handle both old dict format and new ListResponse format
            if hasattr(response, 'models'):
                model_names = [m.model for m in response.models]
            else:
                model_names = [m.get("name", m.get("model", "")) for m in response.get("models", [])]
            # Check if our model (or a variant) is available
            return any(self.config.model in name for name in model_names)
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False


class VLLMPlayer:
    """
    vLLM-based LLM player using OpenAI-compatible API.

    vLLM provides high-performance inference with:
    - Continuous batching for better throughput
    - PagedAttention for efficient memory usage
    - OpenAI-compatible API

    Start vLLM server:
        vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

    Or with tensor parallelism for large models:
        vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 2
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None
        self._last_response_time: float = 0.0

    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def last_response_time(self) -> float:
        """Time taken for last generation in seconds."""
        return self._last_response_time

    def _ensure_client(self):
        """Lazily initialize OpenAI client for vLLM."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai\n"
                    "Also ensure vLLM server is running: vllm serve <model>"
                )

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        Generate a response using vLLM.

        Args:
            prompt: The user prompt (board observation)
            system_prompt: System instructions for the model

        Returns:
            Model's response text
        """
        self._ensure_client()

        start_time = time.time()

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            self._last_response_time = time.time() - start_time
            return response.choices[0].message.content

        except Exception as e:
            self._last_response_time = time.time() - start_time
            raise RuntimeError(f"vLLM generation failed: {e}")

    def is_available(self) -> bool:
        """Check if the vLLM server is reachable and model is loaded."""
        self._ensure_client()
        try:
            models = self._client.models.list()
            model_ids = [m.id for m in models.data]
            return any(self.config.model in mid for mid in model_ids)
        except Exception as e:
            print(f"Error checking vLLM availability: {e}")
            return False


def parse_move_from_response(response: str) -> Optional[str]:
    """
    Extract move from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        Extracted move in SAN notation, or None if not found
    """
    # Try to find <move>...</move> tag
    match = re.search(r"<move>(.*?)</move>", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def create_player(
    backend: str = "ollama",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    base_url: Optional[str] = None,
    **kwargs
) -> LLMPlayer:
    """
    Factory function to create LLM players.

    Args:
        backend: LLM backend to use ("ollama" or "vllm")
        model: Model name (default depends on backend)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        base_url: API base URL (for vllm backend)

    Returns:
        LLMPlayer instance

    Examples:
        # Ollama (simple local inference)
        player = create_player("ollama", model="qwen2.5:0.5b")

        # vLLM (high-performance inference)
        player = create_player("vllm", model="meta-llama/Llama-3.1-8B-Instruct")
    """
    if backend == "ollama":
        config = LLMConfig(
            model=model or "qwen2.5:0.5b",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return OllamaPlayer(config)

    elif backend == "vllm":
        config = LLMConfig(
            model=model or "meta-llama/Llama-3.1-8B-Instruct",
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url or "http://localhost:8000/v1",
            **kwargs
        )
        return VLLMPlayer(config)

    raise ValueError(f"Unknown backend: {backend}. Supported: ollama, vllm")


# Model recommendations for benchmarking - organized by size tier
# GH200 has 96GB VRAM, can run up to ~70B models

RECOMMENDED_MODELS = {
    # Tier 1: Small (< 4B) - fast iteration, local testing
    "qwen2.5:0.5b": {
        "params": "0.5B",
        "size_gb": 0.4,
        "description": "Smallest/fastest, good for testing",
        "tier": "small",
    },
    "qwen2.5:1.5b": {
        "params": "1.5B",
        "size_gb": 1.0,
        "description": "Mid-size Qwen",
        "tier": "small",
    },
    "llama3.2:1b": {
        "params": "1B",
        "size_gb": 1.3,
        "description": "Small Llama",
        "tier": "small",
    },
    "llama3.2:3b": {
        "params": "3B",
        "size_gb": 2.0,
        "description": "Larger Llama, good local baseline",
        "tier": "small",
    },
    "phi3:mini": {
        "params": "3.8B",
        "size_gb": 2.3,
        "description": "Microsoft Phi-3, strong for size",
        "tier": "small",
    },
    "gemma3:4b": {
        "params": "4B",
        "size_gb": 3.0,
        "description": "Google Gemma 3, efficient",
        "tier": "small",
    },
    # Tier 2: Medium (4-15B) - good balance
    "qwen2.5:7b": {
        "params": "7B",
        "size_gb": 4.7,
        "description": "Qwen 2.5 7B, solid mid-range",
        "tier": "medium",
    },
    "llama3.1:8b": {
        "params": "8B",
        "size_gb": 4.9,
        "description": "Llama 3.1 8B, strong baseline",
        "tier": "medium",
    },
    "gemma3:12b": {
        "params": "12B",
        "size_gb": 8.1,
        "description": "Google Gemma 3 12B, very capable",
        "tier": "medium",
    },
    "qwen2.5:14b": {
        "params": "14B",
        "size_gb": 9.0,
        "description": "Qwen 2.5 14B",
        "tier": "medium",
    },
    # Tier 3: Large (15-35B) - needs beefy GPU
    "gemma3:27b": {
        "params": "27B",
        "size_gb": 17.0,
        "description": "Google Gemma 3 27B, very strong",
        "tier": "large",
    },
    "qwen2.5:32b": {
        "params": "32B",
        "size_gb": 20.0,
        "description": "Qwen 2.5 32B",
        "tier": "large",
    },
    # Tier 4: XL (70B+) - GH200 territory
    "llama3.1:70b": {
        "params": "70B",
        "size_gb": 43.0,
        "description": "Llama 3.1 70B, flagship open model",
        "tier": "xl",
    },
    "llama3.3:70b": {
        "params": "70B",
        "size_gb": 43.0,
        "description": "Llama 3.3 70B, latest flagship",
        "tier": "xl",
    },
    "qwen2.5:72b": {
        "params": "72B",
        "size_gb": 47.0,
        "description": "Qwen 2.5 72B, very strong",
        "tier": "xl",
    },
}

# Reasoning models - use with --reasoning flag for higher token limits
# These models do chain-of-thought before answering
REASONING_MODELS = {
    # Small reasoning
    "deepseek-r1:1.5b": {
        "params": "1.5B",
        "size_gb": 1.1,
        "description": "DeepSeek-R1 distilled, smallest",
        "tier": "small",
    },
    "deepseek-r1:7b": {
        "params": "7B",
        "size_gb": 4.7,
        "description": "DeepSeek-R1 distilled, good balance",
        "tier": "medium",
    },
    "deepseek-r1:8b": {
        "params": "8B",
        "size_gb": 4.9,
        "description": "DeepSeek-R1 distilled from Llama",
        "tier": "medium",
    },
    # Medium reasoning
    "deepseek-r1:14b": {
        "params": "14B",
        "size_gb": 9.0,
        "description": "DeepSeek-R1 distilled, stronger",
        "tier": "medium",
    },
    # Large reasoning
    "deepseek-r1:32b": {
        "params": "32B",
        "size_gb": 20.0,
        "description": "DeepSeek-R1 distilled, very capable",
        "tier": "large",
    },
    "qwq:32b": {
        "params": "32B",
        "size_gb": 20.0,
        "description": "Qwen QwQ 32B, strong reasoning",
        "tier": "large",
    },
    # XL reasoning - GH200 territory
    "deepseek-r1:70b": {
        "params": "70B",
        "size_gb": 43.0,
        "description": "DeepSeek-R1 distilled 70B, flagship",
        "tier": "xl",
    },
}
