"""LLM clients behind one interface.

Messages are kept in the OpenAI chat format throughout (system / user /
assistant-with-tool_calls / tool). `OpenAICompatLLM` sends them as-is to any
OpenAI-compatible endpoint (DeepInfra, which FinXray already pays for, OpenAI,
OpenRouter, ...). `AnthropicLLM` converts them to the Claude Messages API.
Either returns an `LLMReply`, so the agent never knows which one it is using.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text


@dataclass
class LLMReply:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMError(RuntimeError):
    pass


class OpenAICompatLLM:
    def __init__(self, s: Settings, session: Optional[requests.Session] = None):
        if not s.llm_api_key:
            raise LLMError("LLM_API_KEY (or DEEPINFRA_API_KEY) is not set")
        self.s = s
        self.url = s.llm_base_url.rstrip("/") + "/chat/completions"
        self.http = session or requests.Session()

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None) -> LLMReply:
        payload: Dict[str, Any] = {"model": self.s.model, "messages": messages,
                                   "temperature": self.s.llm_temperature,
                                   "max_tokens": self.s.llm_max_tokens}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {self.s.llm_api_key}", "Content-Type": "application/json"}
        last = None
        for attempt in range(3):
            try:
                r = self.http.post(self.url, json=payload, headers=headers, timeout=self.s.llm_timeout_s)
            except requests.RequestException as e:
                last = str(e)
            else:
                if r.status_code == 200:
                    return self._parse(r.json())
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (408, 429, 500, 502, 503, 504):
                    break
            time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM request failed: {last}")

    @staticmethod
    def _parse(body: Dict[str, Any]) -> LLMReply:
        try:
            msg = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected LLM response: {str(body)[:300]}") from e
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            calls.append(ToolCall(id=tc.get("id") or f"call_{i}", name=fn.get("name", ""),
                                  arguments=args if isinstance(args, str) else json.dumps(args or {})))
        usage = body.get("usage") or {}
        return LLMReply(content=(msg.get("content") or "").strip(), tool_calls=calls,
                        prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0))


class AnthropicLLM:
    def __init__(self, s: Settings, client=None):
        if client is None:
            if not s.llm_api_key:
                raise LLMError("LLM_API_KEY (or ANTHROPIC_API_KEY) is not set")
            import anthropic
            client = anthropic.Anthropic(api_key=s.llm_api_key, timeout=s.llm_timeout_s, max_retries=2)
        self.s, self.client = s, client

    @staticmethod
    def convert(messages: List[Dict[str, Any]]):
        """OpenAI-format messages -> (system, Claude messages)."""
        system, out = [], []
        for m in messages:
            role = m.get("role")
            if role == "system":
                system.append(m.get("content") or "")
            elif role == "user":
                out.append({"role": "user", "content": m.get("content") or ""})
            elif role == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": fn.get("name"), "input": args})
                out.append({"role": "assistant", "content": blocks or ""})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"),
                         "content": m.get("content") or ""}
                # Consecutive tool results must share ONE user turn.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and all(b.get("type") == "tool_result" for b in out[-1]["content"]):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return "\n\n".join(system), out

    @staticmethod
    def convert_tools(tools: Optional[List[Dict]]):
        return [{"name": t["function"]["name"], "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]} for t in tools or []]

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None) -> LLMReply:
        system, msgs = self.convert(messages)
        kwargs: Dict[str, Any] = {"model": self.s.model, "max_tokens": self.s.llm_max_tokens,
                                  "temperature": self.s.llm_temperature, "system": system,
                                  "messages": msgs}
        if tools:
            kwargs["tools"] = self.convert_tools(tools)
        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as e:
            raise LLMError(f"LLM request failed: {e}") from e
        text, calls = [], []
        for b in resp.content:
            kind = getattr(b, "type", None)
            if kind == "text":
                text.append(b.text)
            elif kind == "tool_use":
                calls.append(ToolCall(id=b.id, name=b.name, arguments=json.dumps(b.input or {})))
        usage = getattr(resp, "usage", None)
        return LLMReply(content="".join(text).strip(), tool_calls=calls,
                        prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                        completion_tokens=int(getattr(usage, "output_tokens", 0) or 0))


def make_llm(s: Settings):
    if s.llm_provider == "anthropic":
        return AnthropicLLM(s)
    return OpenAICompatLLM(s)
