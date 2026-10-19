import json

from gq_bot.config import Settings
from gq_bot.llm import AnthropicLLM, OpenAICompatLLM


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._b, self.text = status, body, json.dumps(body)

    def json(self):
        return self._b


class FakeHTTP:
    def __init__(self, responses):
        self.responses, self.payloads = list(responses), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.payloads.append(json)
        return self.responses.pop(0)


def test_openai_compat_parses_tool_calls_and_retries_5xx(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    body = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "t1", "function": {"name": "get_snapshot", "arguments": '{"ticker":"AAPL"}'}}]}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3}}
    http = FakeHTTP([FakeResp(503, {}), FakeResp(200, body)])
    llm = OpenAICompatLLM(Settings(llm_api_key="k"), session=http)
    r = llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert r.tool_calls[0].name == "get_snapshot" and r.content == "" and r.prompt_tokens == 11
    assert http.payloads[-1]["tool_choice"] == "auto" and http.payloads[-1]["model"] == "google/gemini-2.5-flash"


def test_anthropic_conversion_merges_consecutive_tool_results():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "x", "arguments": '{"ticker":"AAPL"}'}},
                {"id": "b", "type": "function", "function": {"name": "y", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "content": "2"}]
    system, out = AnthropicLLM.convert(msgs)
    assert system == "S" and [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][0] == {"type": "tool_use", "id": "a", "name": "x", "input": {"ticker": "AAPL"}}
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["a", "b"]


def test_anthropic_default_model():
    assert Settings(llm_provider="anthropic").model == "claude-haiku-5-5"
    assert Settings(llm_model="custom").model == "custom"
