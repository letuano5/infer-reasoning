"""
Batch API provider implementations for Gemini, Claude, and OpenAI.

Each provider class exposes a uniform interface:
    submit(requests)   -> batch_id / job_name
    poll(batch_id)     -> status string
    get_results(batch_id) -> dict[custom_id, response_text]

The batch logic is fully isolated here so that runner_batch.py
only deals with prompt building, CSV I/O, and result parsing.
"""

import json
import logging
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("batch_providers")

POLL_INTERVAL_SECONDS = 60  # how often to check batch status


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseBatchProvider(ABC):
    """Uniform interface for all batch providers."""

    name: str  # e.g. "gemini", "claude", "openai"

    @abstractmethod
    def submit(self, requests: list[dict]) -> str:
        """Submit a batch of requests.

        Args:
            requests: list of dicts with keys:
                - custom_id: str   (unique identifier for matching results)
                - prompt: str      (the full prompt string)

        Returns:
            A batch identifier string (job name / batch id).
        """
        ...

    @abstractmethod
    def poll(self, batch_id: str) -> str:
        """Check the current status of a batch.

        Returns one of: "processing", "completed", "failed", "expired".
        """
        ...

    @abstractmethod
    def get_results(self, batch_id: str) -> dict[str, str]:
        """Retrieve results after a batch has completed.

        Returns:
            dict mapping custom_id -> raw response text (the model's full text output).
        """
        ...

    def wait_for_completion(self, batch_id: str, poll_interval: int = POLL_INTERVAL_SECONDS) -> str:
        """Block until the batch finishes. Returns final status."""
        logger.info(f"[{self.name}] Waiting for batch {batch_id} ...")
        while True:
            status = self.poll(batch_id)
            if status in ("completed", "failed", "expired"):
                logger.info(f"[{self.name}] Batch {batch_id} finished with status: {status}")
                return status
            logger.info(f"[{self.name}] Batch {batch_id} status: {status} — sleeping {poll_interval}s")
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Gemini Batch Provider (google-genai SDK)
# ---------------------------------------------------------------------------
class GeminiBatchProvider(BaseBatchProvider):
    """
    Uses the Gemini Batch API via google-genai SDK.
    - Inline requests for <=200k items.
    - Polls via client.batches.get().
    - Results from inlined_responses.
    """

    name = "gemini"

    def __init__(self):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3.1-pro-preview"

    def submit(self, requests: list[dict]) -> str:
        """Submit inline batch requests to Gemini."""
        inline_requests = []
        for req in requests:
            inline_requests.append({
                "contents": [{
                    "parts": [{"text": req["prompt"]}],
                    "role": "user",
                }],
            })

        batch_job = self.client.batches.create(
            model=self.model_id,
            src=inline_requests,
            config={
                "display_name": f"vitext2sql-batch-{int(time.time())}",
            },
        )
        job_name = batch_job.name
        logger.info(f"[{self.name}] Created batch job: {job_name} ({len(requests)} requests)")
        return job_name

    def poll(self, batch_id: str) -> str:
        batch_job = self.client.batches.get(name=batch_id)
        state = batch_job.state.name
        state_map = {
            "JOB_STATE_PENDING": "processing",
            "JOB_STATE_RUNNING": "processing",
            "JOB_STATE_SUCCEEDED": "completed",
            "JOB_STATE_FAILED": "failed",
            "JOB_STATE_CANCELLED": "failed",
            "JOB_STATE_EXPIRED": "expired",
        }
        return state_map.get(state, "processing")

    def get_results(self, batch_id: str) -> dict[str, str]:
        batch_job = self.client.batches.get(name=batch_id)
        results = {}

        if batch_job.dest and batch_job.dest.inlined_responses:
            for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                custom_id = str(i)  # inline responses are ordered same as input
                if inline_response.response:
                    try:
                        # Gemini response: extract text from parts
                        parts = inline_response.response.candidates[0].content.parts
                        response_parts = []
                        for part in parts:
                            if not getattr(part, "thought", False):
                                response_parts.append(part.text)
                        results[custom_id] = "\n".join(response_parts)
                    except (AttributeError, IndexError) as e:
                        logger.warning(f"[{self.name}] Failed to parse response {i}: {e}")
                        results[custom_id] = ""
                elif inline_response.error:
                    logger.warning(f"[{self.name}] Request {i} errored: {inline_response.error}")
                    results[custom_id] = ""
        elif batch_job.dest and batch_job.dest.file_name:
            # File-based results
            file_content = self.client.files.download(file=batch_job.dest.file_name)
            content_str = file_content.decode("utf-8")
            for i, line in enumerate(content_str.strip().split("\n")):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    # File responses use key field
                    key = obj.get("key", str(i))
                    resp = obj.get("response", {})
                    candidates = resp.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        text_parts = [p.get("text", "") for p in parts]
                        results[key] = "\n".join(text_parts)
                    else:
                        results[key] = ""
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"[{self.name}] Failed to parse result line {i}: {e}")
                    results[str(i)] = ""

        logger.info(f"[{self.name}] Retrieved {len(results)} results from batch {batch_id}")
        return results


# ---------------------------------------------------------------------------
# Claude (Anthropic) Batch Provider
# ---------------------------------------------------------------------------
class ClaudeBatchProvider(BaseBatchProvider):
    """
    Uses the Anthropic Message Batches API.
    - Submits via client.messages.batches.create()
    - Polls via client.messages.batches.retrieve()
    - Results streamed via client.messages.batches.results()
    """

    name = "claude"

    def __init__(self):
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = "claude-sonnet-4-20250514"

    def submit(self, requests: list[dict]) -> str:
        batch_requests = []
        for req in requests:
            batch_requests.append({
                "custom_id": req["custom_id"],
                "params": {
                    "model": self.model_id,
                    "max_tokens": 8192,
                    "messages": [
                        {"role": "user", "content": req["prompt"]},
                    ],
                },
            })

        message_batch = self.client.messages.batches.create(requests=batch_requests)
        batch_id = message_batch.id
        logger.info(f"[{self.name}] Created batch: {batch_id} ({len(requests)} requests)")
        return batch_id

    def poll(self, batch_id: str) -> str:
        message_batch = self.client.messages.batches.retrieve(batch_id)
        status = message_batch.processing_status
        if status == "ended":
            return "completed"
        elif status == "in_progress":
            return "processing"
        else:
            return "processing"

    def get_results(self, batch_id: str) -> dict[str, str]:
        results = {}
        for result in self.client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            if result.result.type == "succeeded":
                message = result.result.message
                # Concatenate all text blocks
                text_parts = []
                for block in message.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                results[custom_id] = "\n".join(text_parts)
            else:
                logger.warning(
                    f"[{self.name}] Request {custom_id} result type: {result.result.type}"
                )
                results[custom_id] = ""

        logger.info(f"[{self.name}] Retrieved {len(results)} results from batch {batch_id}")
        return results


# ---------------------------------------------------------------------------
# OpenAI Batch Provider
# ---------------------------------------------------------------------------
class OpenAIBatchProvider(BaseBatchProvider):
    """
    Uses the OpenAI Batch API.
    - Creates a JSONL file, uploads it, then creates a batch.
    - Polls via client.batches.retrieve().
    - Results downloaded from output file.
    """

    name = "openai"

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI(api_key=api_key)
        self.model_id = "gpt-5.2"

    def submit(self, requests: list[dict]) -> str:
        # 1. Build JSONL content
        tasks = []
        for req in requests:
            task = {
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_id,
                    "messages": [
                        {"role": "user", "content": req["prompt"]},
                    ],
                    "max_tokens": 1000,
                },
            }
            tasks.append(task)

        # 2. Write to temp JSONL file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        try:
            for task in tasks:
                tmp.write(json.dumps(task,ensure_ascii=False) + "\n")
            tmp.close()

            # 3. Upload file
            with open(tmp.name, "rb") as f:
                batch_file = self.client.files.create(file=f, purpose="batch")

            logger.info(f"[{self.name}] Uploaded batch file: {batch_file.id}")

            # 4. Create batch job
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            logger.info(f"[{self.name}] Created batch: {batch_job.id} ({len(requests)} requests)")
            return batch_job.id
        finally:
            os.unlink(tmp.name)

    def poll(self, batch_id: str) -> str:
        batch_job = self.client.batches.retrieve(batch_id)
        status = batch_job.status
        status_map = {
            "validating": "processing",
            "in_progress": "processing",
            "finalizing": "processing",
            "completed": "completed",
            "failed": "failed",
            "expired": "expired",
            "cancelling": "processing",
            "cancelled": "failed",
        }
        return status_map.get(status, "processing")

    def get_results(self, batch_id: str) -> dict[str, str]:
        batch_job = self.client.batches.retrieve(batch_id)
        if not batch_job.output_file_id:
            logger.error(f"[{self.name}] No output file for batch {batch_id}")
            return {}

        # Download result file
        file_content = self.client.files.content(batch_job.output_file_id).content
        results = {}
        for line in file_content.decode("utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                custom_id = obj["custom_id"]
                response_body = obj.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    results[custom_id] = text
                else:
                    results[custom_id] = ""
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[{self.name}] Failed to parse result: {e}")

        logger.info(f"[{self.name}] Retrieved {len(results)} results from batch {batch_id}")
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_batch_provider(name: str) -> BaseBatchProvider:
    """Create a batch provider by name."""
    providers = {
        "gemini": GeminiBatchProvider,
        "claude": ClaudeBatchProvider,
        "openai": OpenAIBatchProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown batch provider: {name}. Available: {list(providers.keys())}")
    return providers[name]()
