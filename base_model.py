"""
Abstract base class for reasoning model implementations.
Each model must implement the `generate` method.
"""

from abc import ABC, abstractmethod


class BaseReasoningModel(ABC):
    """Interface for a reasoning model that produces (think, sql) pairs."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> tuple[str, str]:
        """
        Send the prompt to the model and return (think, sql).

        Args:
            prompt: The fully-formed prompt string including schema and question.

        Returns:
            A tuple of (think_text, sql_text).
            - think_text: the chain-of-thought reasoning string.
            - sql_text: the generated SQL query string.
        """
        ...

    @property
    def name(self) -> str:
        """Short name used for output file naming, e.g. 'gemini', 'deepseek'."""
        return self.model_name
