"""
test_embedding_backend.py
----------------------------
Ye sandbox mein live Gemini API call NAHI kar sakta (network restricted
hai is dev environment mein) — isliye client ko mock karte hain aur
sirf ye verify karte hain ke:
  1. Sahi arguments ke saath call ho raha hai (model, task_type).
  2. Response se embedding sahi tarah nikal rahi hai.
  3. Retry logic kaam kar rahi hai.
  4. Factory function (get_backend) sahi backend class laut a raha hai.

Deploy se pehle ek chhota manual smoke-test zaroor chalayein (README.md
mein "Verify before you deploy" section dekhein) — mocks sirf ye
guarantee dete hain ke code logically sahi hai, live API contract
guarantee nahi karte.
"""

from unittest.mock import MagicMock, patch

import pytest

from embedding_backend import GeminiEmbeddingBackend, get_backend


class TestGeminiEmbeddingBackend:
    def _make_fake_client(self, values=(0.1, 0.2, 0.3)):
        client = MagicMock()
        fake_embedding = MagicMock()
        fake_embedding.values = list(values)
        fake_result = MagicMock()
        fake_result.embeddings = [fake_embedding]
        client.models.embed_content.return_value = fake_result
        return client

    def test_embed_query_calls_with_retrieval_query_task_type(self):
        client = self._make_fake_client()
        backend = GeminiEmbeddingBackend(client=client, retries=1)
        vec = backend.embed_query("what is a derivative?")
        assert vec == [0.1, 0.2, 0.3]
        _, kwargs = client.models.embed_content.call_args
        assert kwargs["config"].task_type == "RETRIEVAL_QUERY"
        assert kwargs["model"] == "gemini-embedding-001"

    def test_embed_document_calls_with_retrieval_document_task_type(self):
        client = self._make_fake_client()
        backend = GeminiEmbeddingBackend(client=client, retries=1)
        backend.embed_document("Definition: a derivative measures rate of change.")
        _, kwargs = client.models.embed_content.call_args
        assert kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"

    def test_retries_on_failure_then_succeeds(self):
        client = MagicMock()
        fake_embedding = MagicMock()
        fake_embedding.values = [0.5, 0.5]
        fake_result = MagicMock()
        fake_result.embeddings = [fake_embedding]
        client.models.embed_content.side_effect = [Exception("transient error"), fake_result]

        backend = GeminiEmbeddingBackend(client=client, retries=2)
        with patch("time.sleep"):  # test ko slow na karein
            vec = backend.embed_query("test")
        assert vec == [0.5, 0.5]
        assert client.models.embed_content.call_count == 2

    def test_raises_after_exhausting_retries(self):
        client = MagicMock()
        client.models.embed_content.side_effect = Exception("persistent error")
        backend = GeminiEmbeddingBackend(client=client, retries=2)
        with patch("time.sleep"):
            with pytest.raises(Exception, match="persistent error"):
                backend.embed_query("test")


class TestGetBackendFactory:
    def test_gemini_provider_requires_client(self):
        with pytest.raises(ValueError):
            get_backend("gemini", client=None)

    def test_gemini_provider_returns_gemini_backend(self):
        backend = get_backend("gemini", client=MagicMock())
        assert isinstance(backend, GeminiEmbeddingBackend)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            get_backend("some-made-up-provider")

    def test_local_provider_without_dependency_raises_helpful_error(self):
        # sentence-transformers is sandbox mein installed nahi hai (bhaari
        # dependency, network-restricted) — isliye ye confirm karta hai ke
        # error message helpful hai, crash silent/confusing nahi.
        import importlib.util

        if importlib.util.find_spec("sentence_transformers") is not None:
            pytest.skip("sentence-transformers installed hai is environment mein — skip.")
        with pytest.raises(ImportError, match="requirements-local-embeddings"):
            get_backend("local")
