"""
Gemini model implementation using google-genai SDK with thinking enabled.
"""

import re
import os
import logging

from google import genai
from google.genai import types
from dotenv import load_dotenv

from base_model import BaseReasoningModel

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiModel(BaseReasoningModel):
    """Gemini reasoning model (thinking mode enabled)."""

    def __init__(self):
        super().__init__("gemini")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-pro" # "gemini-3.1-pro-preview"

    def generate(self, prompt: str) -> tuple[str, str]:
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            # config=types.GenerateContentConfig(
            #     thinking_config=types.ThinkingConfig(thinking_level="high")
            # ),
        )

        native_thinking = []
        response_parts = []

        for part in response.candidates[0].content.parts:
            if part.thought:
                native_thinking.append(part.text)
            else:
                response_parts.append(part.text)

        if native_thinking:
            logger.debug(
                "Native thinking (first 500 chars): %s",
                "\n".join(native_thinking)[:500],
            )

        response_text = "\n".join(response_parts)

        # Extract formatted <think> (5-step reasoning) from response content
        think_text = self._extract_think(response_text)
        # Extract <sql> from response content
        sql = self._extract_sql(response_text)

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
        # Last resort: return the whole response text
        logger.warning("Could not extract SQL from response, returning raw text")
        return text.strip()

    @staticmethod
    def _extract_think(text: str) -> str:
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
