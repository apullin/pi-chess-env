"""
LLM player wrapper for chess.

Provides a unified interface for LLM-based chess players using local inference.
Currently supports Ollama for local model serving.

Usage:
    from chess_env.llm_player import create_player

    player = create_player("ollama", model="qwen2.5:0.5b")
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
    **kwargs
) -> LLMPlayer:
    """
    Factory function to create LLM players.

    Args:
        backend: LLM backend to use ("ollama")
        model: Model name (default depends on backend)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        LLMPlayer instance

    Example:
        player = create_player("ollama", model="qwen2.5:0.5b")
    """
    if backend == "ollama":
        config = LLMConfig(
            model=model or "qwen2.5:0.5b",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return OllamaPlayer(config)

    raise ValueError(f"Unknown backend: {backend}. Supported: ollama")


# Model recommendations for benchmarking
RECOMMENDED_MODELS = {
    "qwen2.5:0.5b": {
        "params": "0.5B",
        "size_gb": 0.4,
        "description": "Smallest/fastest, good for testing",
    },
    "qwen2.5:1.5b": {
        "params": "1.5B",
        "size_gb": 1.0,
        "description": "Mid-size Qwen, better reasoning",
    },
    "llama3.2:1b": {
        "params": "1B",
        "size_gb": 1.3,
        "description": "Small Llama, different architecture",
    },
    "llama3.2:3b": {
        "params": "3B",
        "size_gb": 2.0,
        "description": "Larger Llama, better performance",
    },
    "phi3:mini": {
        "params": "3.8B",
        "size_gb": 2.3,
        "description": "Microsoft Phi-3, strong for size",
    },
    "gemma3:4b": {
        "params": "4B",
        "size_gb": 3.0,
        "description": "Google Gemma 3, efficient architecture",
    },
    "gemma3:12b": {
        "params": "12B",
        "size_gb": 8.1,
        "description": "Google Gemma 3, very strong for size",
    },
}

# Reasoning models - use with --reasoning flag for higher token limits
REASONING_MODELS = {
    "deepseek-r1:1.5b": {
        "params": "1.5B",
        "size_gb": 1.1,
        "description": "DeepSeek-R1 distilled, smallest reasoning model",
    },
    "deepseek-r1:7b": {
        "params": "7B",
        "size_gb": 4.7,
        "description": "DeepSeek-R1 distilled, good balance",
    },
    "deepseek-r1:8b": {
        "params": "8B",
        "size_gb": 4.9,
        "description": "DeepSeek-R1 distilled from Llama",
    },
    "deepseek-r1:14b": {
        "params": "14B",
        "size_gb": 9.0,
        "description": "DeepSeek-R1 distilled, stronger reasoning",
    },
}
