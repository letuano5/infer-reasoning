"""
DeepSeek Reasoner model implementation using OpenAI-compatible API.
"""

import re
import os
import logging

from openai import OpenAI
from dotenv import load_dotenv

from base_model import BaseReasoningModel

load_dotenv()
logger = logging.getLogger(__name__)


class DeepSeekModel(BaseReasoningModel):
    """DeepSeek Reasoner model (native reasoning via reasoning_content)."""

    def __init__(self):
        super().__init__("deepseek")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set in .env")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.model_id = "deepseek-reasoner"

    def generate(self, prompt: str) -> tuple[str, str]:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
        )

        choice = response.choices[0].message

        # Log native reasoning for debug
        native_reasoning = getattr(choice, "reasoning_content", "") or ""
        if native_reasoning:
            logger.debug(
                "Native reasoning (first 500 chars): %s",
                native_reasoning[:500],
            )

        content = choice.content or ""

        # Extract formatted <think> (5-step reasoning) from response content
        think_text = self._extract_think(content)
        # Extract <sql> from response content
        sql = self._extract_sql(content)

        return think_text, sql

    @staticmethod
    def _extract_sql(text: str) -> str:
        match = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: try to find SQL in code blocks
        match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        logger.warning("Could not extract SQL from response, returning raw text")
        return text.strip()

    @staticmethod
    def _extract_think(text: str) -> str:
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
