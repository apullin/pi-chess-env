"""
Prime Intellect Environments Hub Entry Point.

This file is the entry point for the Prime Intellect Environments Hub.
When you run `prime env push`, this file is what gets executed.

The Hub calls `load_environment()` to get your environment.
"""

from chess_env.verifiers_env import load_environment

# This is required by the Hub
__all__ = ["load_environment"]
