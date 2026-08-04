"""
test_generation_backend.py
-----------------------------
AgentRouter/agentrouter.org is dev sandbox ki allowed-domains list mein
NAHI hai, isliye maine (Claude) usse live call NAHI kiya — ye tests
OpenAI-compatible contract ko mock kar ke check karte hain ke request
sahi shape mein ban rahi hai aur response sahi parse ho rahi hai.

>>> DEPLOY SE PEHLE, ek real key (regenerated — purani wali chat mein
>>> paste hone ki wajah se already compromise maani jani chahiye) ke
>>> saath khud ek live smoke-test zaroor chalayein.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from generation_backend import (
    GeminiGenerationBackend,
    OpenAICompatibleGenerationBackend,
    _strip_markdown_fences,
    get_generation_backend,
)


class DummySchema(BaseModel):
    english: str
    grounding: str


class TestGeminiGenerationBackend:
    def test_calls_interactions_create_with_wrapped_response_format(self):
        client = MagicMock()
        fake_interaction = MagicMock()
        fake_interaction.output_text = json.dumps({"english": "hi", "grounding": "direct_from_notes"})
        client.interactions.create.return_value = fake_interaction

        backend = GeminiGenerationBackend(client=client, model="gemini-3.6-flash")
        result = backend.generate("system", "prompt", DummySchema)

        assert result.english == "hi"
        _, kwargs = client.interactions.create.call_args
        # FIX regression test: response_format ab list-wrapped {type, mime_type, schema}
        # honi chahiye, raw dict nahi (jo docs ke contract se match nahi karta tha)
        assert isinstance(kwargs["response_format"], list)
        assert kwargs["response_format"][0]["type"] == "text"
        assert kwargs["response_format"][0]["mime_type"] == "application/json"
        assert "schema" in kwargs["response_format"][0]
        assert kwargs["response_format"][0]["schema"] == DummySchema.model_json_schema()


class TestStripMarkdownFences:
    def test_strips_json_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert _strip_markdown_fences(text) == '{"a": 1}'

    def test_strips_plain_fence(self):
        text = '```\n{"a": 1}\n```'
        assert _strip_markdown_fences(text) == '{"a": 1}'

    def test_leaves_unfenced_text_alone(self):
        text = '{"a": 1}'
        assert _strip_markdown_fences(text) == '{"a": 1}'


class TestOpenAICompatibleGenerationBackend:
    def test_builds_correct_request_and_parses_response(self):
        backend = OpenAICompatibleGenerationBackend(
            api_key="sk-test", base_url="https://agentrouter.org/v1", model="claude-sonnet-4-5-20250929"
        )
        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json.return_value = {
            "choices": [{"message": {"content": '{"english": "hi", "grounding": "direct_from_notes"}'}}]
        }

        with patch("requests.post", return_value=fake_response) as mock_post:
            result = backend.generate("system", "prompt", DummySchema)

        assert result.english == "hi"
        called_url = mock_post.call_args[0][0]
        assert called_url == "https://agentrouter.org/v1/chat/completions"
        sent_json = mock_post.call_args[1]["json"]
        assert sent_json["model"] == "claude-sonnet-4-5-20250929"
        assert sent_json["response_format"] == {"type": "json_object"}
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-test"

    def test_handles_markdown_fenced_response(self):
        backend = OpenAICompatibleGenerationBackend(
            api_key="sk-test", base_url="https://agentrouter.org/v1", model="some-model"
        )
        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fenced_content = '```json\n{"english": "hi", "grounding": "direct_from_notes"}\n```'
        fake_response.json.return_value = {"choices": [{"message": {"content": fenced_content}}]}

        with patch("requests.post", return_value=fake_response):
            result = backend.generate("system", "prompt", DummySchema)
        assert result.english == "hi"

    def test_retries_then_raises_on_persistent_failure(self):
        backend = OpenAICompatibleGenerationBackend(
            api_key="sk-test", base_url="https://agentrouter.org/v1", model="some-model", retries=2
        )
        with patch("requests.post", side_effect=Exception("connection error")):
            with patch("time.sleep"):
                with pytest.raises(Exception, match="connection error"):
                    backend.generate("system", "prompt", DummySchema)


class TestGetGenerationBackendFactory:
    def test_gemini_default(self):
        backend = get_generation_backend("gemini", client=MagicMock())
        assert isinstance(backend, GeminiGenerationBackend)

    def test_known_gateway_shortcut_resolves_base_url(self):
        backend = get_generation_backend("agentrouter", api_key="sk-test", model="claude-sonnet-4-5-20250929")
        assert isinstance(backend, OpenAICompatibleGenerationBackend)
        assert backend._base_url == "https://agentrouter.org/v1"

    def test_missing_api_key_raises_helpful_error(self):
        with pytest.raises(ValueError, match="api_key"):
            get_generation_backend("agentrouter", api_key=None, model="some-model")

    def test_missing_model_raises_helpful_error(self):
        with pytest.raises(ValueError, match="model"):
            get_generation_backend("agentrouter", api_key="sk-test", model=None)

    def test_unknown_provider_without_base_url_raises(self):
        with pytest.raises(ValueError):
            get_generation_backend("totally-unknown-provider", api_key="sk-test", model="m")
