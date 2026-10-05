"""LLM backends returning schema-constrained JSON.

* :class:`AnthropicBackend` -- Claude via the official ``anthropic`` SDK
  (default model ``claude-opus-5-5``, adaptive thinking, structured JSON output,
  streaming, prompt caching of the stable system prompt, and server-side refusal
  fallbacks enabled).
* :class:`CommandBackend` -- any local or self-hosted model behind a shell
  command that reads a JSON request on stdin and prints JSON on stdout.
* :class:`ScriptedBackend` -- canned responses for tests and offline demos.

Select with ``--llm anthropic``, ``--llm anthropic:claude-sonnet-5-5`` or
``--llm "cmd:python my_llm.py"`` (or the ``TRAINMEDIC_LLM`` environment variable).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Callable, Protocol

DEFAULT_MODEL = "claude-opus-5-5"


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class LLMBackend(Protocol):
    name: str

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any], *, max_tokens: int = 32000) -> dict[str, Any]:
        ...


class AnthropicBackend:
    def __init__(
        self,
        model: str | None = None,
        *,
        effort: str | None = None,
        max_tokens: int = 32000,
        fallbacks: bool = True,
        client: Any = None,
    ) -> None:
        self.model = model or os.environ.get("TRAINMEDIC_LLM_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.environ.get("TRAINMEDIC_LLM_EFFORT", "high")
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self._client = client
        self.name = f"anthropic:{self.model}"
        self.usage: list[dict[str, Any]] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise LLMError("the Anthropic backend needs `pip install trainmedic[llm]`") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any], *, max_tokens: int | None = None) -> dict[str, Any]:
        import anthropic

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        }
        try:
            if self.fallbacks:
                stream_ctx = self.client.beta.messages.stream(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                stream_ctx = self.client.messages.stream(**kwargs)
            with stream_ctx as stream:
                msg = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"model {self.model!r} not found") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("rate limited by the Anthropic API; retry later") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("could not reach the Anthropic API") from exc
        usage = getattr(msg, "usage", None)
        if usage is not None:
            self.usage.append({"input": getattr(usage, "input_tokens", None), "output": getattr(usage, "output_tokens", None),
                               "cache_read": getattr(usage, "cache_read_input_tokens", None)})
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            raise LLMRefusal(f"model declined the request ({getattr(details, 'category', None)})")
        if msg.stop_reason == "max_tokens":
            raise LLMError("response truncated (max_tokens); increase max_tokens")
        text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise LLMError("model returned no text block")
        return parse_json(text)


class CommandBackend:
    """Shell-command backend: stdin gets {"system", "prompt", "schema"}; stdout must be JSON."""

    def __init__(self, command: str, timeout: float = 900.0) -> None:
        self.command = command
        self.timeout = timeout
        self.name = f"cmd:{command}"

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any], *, max_tokens: int = 32000) -> dict[str, Any]:
        payload = json.dumps({"system": system, "prompt": prompt, "schema": schema, "max_tokens": max_tokens})
        try:
            proc = subprocess.run(self.command, shell=True, input=payload, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"LLM command timed out after {self.timeout}s") from exc
        if proc.returncode != 0:
            raise LLMError(f"LLM command failed ({proc.returncode}): {proc.stderr.strip()[-500:]}")
        return parse_json(proc.stdout)


class ScriptedBackend:
    """Returns pre-programmed responses (a list consumed in order, or a function)."""

    def __init__(self, responses: list[dict[str, Any]] | Callable[[str, str, dict], dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.name = "scripted"

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any], *, max_tokens: int = 32000) -> dict[str, Any]:
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        if callable(self.responses):
            return self.responses(system, prompt, schema)
        if not self.responses:
            raise LLMError("scripted backend has no responses left")
        return self.responses.pop(0)


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise LLMError("could not parse JSON from model output")


def get_backend(spec: str | None = None) -> LLMBackend | None:
    spec = spec if spec is not None else os.environ.get("TRAINMEDIC_LLM")
    if not spec or spec.lower() in ("none", "off", "rules"):
        return None
    if spec.startswith("anthropic"):
        _, _, model = spec.partition(":")
        return AnthropicBackend(model or None)
    if spec.startswith("cmd:"):
        return CommandBackend(spec[4:])
    raise ValueError(f"unknown LLM backend spec {spec!r} (use 'anthropic[:MODEL]', 'cmd:COMMAND' or 'none')")
