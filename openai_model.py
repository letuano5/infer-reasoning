"""
OpenAI model implementation with prompt caching optimization.

OpenAI's prompt caching is automatic — it caches the longest common prefix 
of messages (≥1024 tokens) that repeat across requests.  By splitting the 
prompt into a long system message (instructions + schema) and a short user
message (just the question), we maximize cache hits for requests sharing 
the same schema.

Usage in runner.py:
    The model's `generate` method accepts both `prompt` (full) for 
    backward compatibility AND `schema`+`question` separately for 
    the cache-optimised path.
"""

import re
import os
import logging

from openai import OpenAI
from dotenv import load_dotenv

from base_model import BaseReasoningModel
from prompt import prompt_template

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt: everything EXCEPT the question.
# The {schema} placeholder will be filled per-schema; the question goes
# into the user message.
# ---------------------------------------------------------------------------
_SYSTEM_TEMPLATE = """\
Bạn là công cụ "SQLite Text-to-SQL Reasoner" chuyên nghiệp.

=== INPUT ===
SCHEMA:
{{
    {schema}
}}

=== NHIỆM VỤ ===
Sinh (1) chuỗi suy luận theo bước và (2) một câu SQL chạy được trên SQLite.

=== RÀNG BUỘC BẮT BUỘC ===
- Chỉ dùng bảng/cột có trong schema. Không bịa tên cột.
- Dialect: SQLite. Không dùng ILIKE, FULL OUTER JOIN, hàm riêng MySQL/PostgreSQL.
- Khi JOIN từ 2 bảng trở lên: luôn đặt alias bảng, prefix đầy đủ cho mọi tên cột.
- Nếu khái niệm trong câu hỏi không khớp tên cột, chọn cột gần nghĩa nhất và ghi rõ giả định.
- Dùng CTE (WITH) khi logic tái sử dụng hoặc subquery lồng > 2 tầng.

=== BỘ BƯỚC SUY LUẬN (LUÔN VIẾT ĐỦ 5 MỤC) ===

1) Mục tiêu & grain
   Tóm tắt mục tiêu và loại câu trả lời: liệt kê / thống kê / tồn tại / so sánh / top-k
   Xác định grain (1 dòng = 1 gì?).

2) Gắn khái niệm vào schema
   Map từng khái niệm trong câu hỏi → bảng.cột cụ thể.
   Nếu ambiguous, ghi rõ giả định tối thiểu.

3) Khung truy vấn
   Anchor table, có/không CTE, các field sẽ SELECT.

4) Module logic (chỉ ghi module thật sự cần)
   - JOIN: đường join + khóa.
   - AGG: hàm + GROUP BY / HAVING.
   - TEMPORAL: cột thời gian + kỹ thuật (LAG/LEAD/self-join).
   - SET: UNION/INTERSECT/EXCEPT.
   - RANK: ORDER BY / LIMIT / window function.
   - DEDUP: DISTINCT / COUNT(DISTINCT) nếu có nguy cơ nhân bản.

5) Hoàn thiện & kiểm tra
   - ORDER BY / LIMIT nếu cần.
   - NULL handling: COALESCE/IFNULL đúng chỗ chưa?
   - COUNT(*) vs COUNT(col) đúng chưa?
   - Tên cột tồn tại trong schema, SQLite compatible?

=== OUTPUT FORMAT ===
<think>
1. ...
2. ...
3. ...
4. ...
5. ...
</think>

<sql>
-- một câu SQL duy nhất (có thể dùng WITH), không comment giải thích trong này
</sql>"""


class OpenAIModel(BaseReasoningModel):
    """OpenAI model with prompt caching optimization.

    OpenAI automatically caches the longest matching prefix of the message
    list.  By placing the long, schema-specific instructions in the system
    message and only putting the short question in the user message, all
    questions that share the same schema will hit the prompt cache after
    the first request (given the system message exceeds 1024 tokens).
    """

    def __init__(self, model_id: str = "o3"):
        super().__init__("openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        # Cache built system messages per schema_id to avoid re-formatting
        self._system_cache: dict[str, str] = {}

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate(self, prompt: str) -> tuple[str, str]:
        """Fallback: send the whole prompt as a single user message.

        This path does NOT benefit from prompt caching but keeps
        backward compatibility with runner.py's existing call pattern.
        """
        response = self.client.responses.create(
            model=self.model_id,
            input=[{"role": "user", "content": prompt}],
        )
        content = response.output_text or ""

        self._log_reasoning(response)
        self._log_cache_stats(response)

        think_text = self._extract_think(content)
        sql = self._extract_sql(content)
        return think_text, sql

    def generate_with_cache(self, schema: str, question: str) -> tuple[str, str]:
        """Cache-optimized path: split system (instructions+schema) / user (question).

        The system message contains the full instructions and the schema,
        which is identical for every question in the same schema_id.
        The user message contains only the question, which is short.

        OpenAI will automatically cache the system message prefix when it
        is ≥ 1024 tokens and the same prefix is sent in subsequent requests.
        """
        system_msg = self._get_system_message(schema)

        response = self.client.responses.create(
            model=self.model_id,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"QUESTION:\n{{\n    {question}\n}}"},
            ],
        )

        content = response.output_text or ""

        self._log_reasoning(response)
        self._log_cache_stats(response)

        think_text = self._extract_think(content)
        sql = self._extract_sql(content)
        return think_text, sql

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _get_system_message(self, schema: str) -> str:
        """Build (and locally cache) the system message for a given schema."""
        if schema not in self._system_cache:
            self._system_cache[schema] = _SYSTEM_TEMPLATE.format(schema=schema)
        return self._system_cache[schema]

    @staticmethod
    def _log_reasoning(response) -> None:
        """Log native reasoning summary from o3/o4 reasoning models."""
        for item in getattr(response, "output", []):
            if getattr(item, "type", "") == "reasoning":
                summaries = getattr(item, "summary", [])
                if summaries:
                    text = " ".join(s.get("text", "") for s in summaries if isinstance(s, dict))
                    if text:
                        logger.debug(
                            "Native reasoning (first 500 chars): %s",
                            text[:500],
                        )

    @staticmethod
    def _log_cache_stats(response) -> None:
        """Log prompt caching statistics from the response usage object."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        # Responses API: usage.input_tokens / usage.input_tokens_details.cached_tokens
        cached_details = getattr(usage, "input_tokens_details", None)
        if cached_details is None:
            return
        cached_tokens = getattr(cached_details, "cached_tokens", 0) or 0
        total_input = getattr(usage, "input_tokens", 0) or 0
        if cached_tokens > 0:
            pct = (cached_tokens / total_input * 100) if total_input else 0
            logger.info(
                "Prompt cache hit: %d / %d tokens (%.1f%%)",
                cached_tokens, total_input, pct,
            )

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
