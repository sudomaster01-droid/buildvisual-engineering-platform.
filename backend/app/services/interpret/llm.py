"""Language-model clients used for interpretation and optional classification.

Blueprint p.25: model-specific prompts and wire formats stay behind an interface
so nothing else in the application knows which vendor is in use. All three
adapters below expose the same ``generate_json`` contract and return parsed JSON
or raise; the caller validates the JSON against our own schema and falls back
deterministically if it is unusable.

Keys are read from configuration on the server only. Nothing here is reachable
from the browser.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from ...config import get_settings
from ...errors import CapabilityUnavailable, InterpretationError

log = logging.getLogger("buildvisual.llm")

REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=60.0, pool=10.0)


@dataclass
class LlmUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""


class LLMClient(Protocol):
    provider: str
    model: str
    last_usage: LlmUsage

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        images: list[tuple[bytes, str]] | None = None,
        max_output_tokens: int = 4096,
    ) -> dict: ...


def _extract_json(text: str) -> dict:
    """Models occasionally wrap JSON in prose or fences; recover the object."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise InterpretationError(
                "The interpretation model returned malformed JSON."
            ) from exc
    raise InterpretationError("The interpretation model returned no JSON object.")


class GeminiClient:
    provider = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.last_usage = LlmUsage(provider=self.provider, model=model)

    def generate_json(self, *, system, user, images=None, max_output_tokens=4096) -> dict:
        parts: list[dict] = [{"text": user}]
        for data, mime in (images or []):
            parts.append({
                "inline_data": {
                    "mime_type": mime,
                    "data": base64.b64encode(data).decode("ascii"),
                }
            })
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                url, headers={"x-goog-api-key": self.api_key}, json=payload
            )
        if response.status_code >= 400:
            raise InterpretationError(
                "The interpretation model rejected the request.",
                details={"status": response.status_code, "provider": self.provider},
            )
        data = response.json()
        usage = data.get("usageMetadata", {})
        self.last_usage = LlmUsage(
            input_tokens=int(usage.get("promptTokenCount", 0)),
            output_tokens=int(usage.get("candidatesTokenCount", 0)),
            model=self.model, provider=self.provider,
        )
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            blocked = data.get("promptFeedback", {}).get("blockReason")
            raise InterpretationError(
                "The interpretation model returned no content.",
                details={"blocked": blocked},
            ) from exc
        return _extract_json(text)


class OpenAIClient:
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        provider: str = "openai",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.last_usage = LlmUsage(provider=provider, model=model)

    def generate_json(self, *, system, user, images=None, max_output_tokens=4096) -> dict:
        content: list[dict] = [{"type": "text", "text": user}]
        for data, mime in (images or []):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
                },
            })
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
        if response.status_code >= 400:
            raise InterpretationError(
                "The interpretation model rejected the request.",
                details={"status": response.status_code, "provider": self.provider},
            )
        data = response.json()
        usage = data.get("usage", {})
        self.last_usage = LlmUsage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            model=self.model, provider=self.provider,
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise InterpretationError("The interpretation model returned no content.") from exc
        return _extract_json(text)


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.last_usage = LlmUsage(provider=self.provider, model=model)

    def generate_json(self, *, system, user, images=None, max_output_tokens=4096) -> dict:
        content: list[dict] = []
        for data, mime in (images or []):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64", "media_type": mime,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            })
        content.append({"type": "text", "text": user})
        payload = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "temperature": 0.1,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
        if response.status_code >= 400:
            raise InterpretationError(
                "The interpretation model rejected the request.",
                details={"status": response.status_code, "provider": self.provider},
            )
        data = response.json()
        usage = data.get("usage", {})
        self.last_usage = LlmUsage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            model=self.model, provider=self.provider,
        )
        blocks = data.get("content") or []
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        if not text:
            raise InterpretationError("The interpretation model returned no content.")
        return _extract_json(text)


DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-5",
}


def build_llm_client() -> LLMClient:
    settings = get_settings()
    provider = settings.resolved_llm_provider
    if provider == "none":
        raise CapabilityUnavailable(
            "No vision/language model is configured, so drawings cannot be interpreted. "
            "Type or speak the requirements instead, or configure a local model endpoint."
        )
    if provider == "local_openai":
        if not settings.llm_model:
            raise CapabilityUnavailable(
                "The local vision model name is missing from LLM_MODEL."
            )
        return OpenAIClient(
            api_key=settings.llm_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            provider="local_openai",
        )
    key = settings.llm_key
    if not key:
        raise CapabilityUnavailable(
            "The language-model API key is missing, so drawings cannot be interpreted."
        )
    model = settings.llm_model or DEFAULT_MODELS.get(provider, "")
    if provider == "gemini":
        return GeminiClient(key, model)
    if provider == "openai":
        return OpenAIClient(key, model)
    if provider == "anthropic":
        return AnthropicClient(key, model)
    raise CapabilityUnavailable(f"Unsupported LLM provider: {provider}")


def llm_available() -> bool:
    settings = get_settings()
    if settings.resolved_llm_provider == "local_openai":
        return bool(settings.llm_model and settings.llm_base_url)
    return settings.resolved_llm_provider != "none" and bool(settings.llm_key)
