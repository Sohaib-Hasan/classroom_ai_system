"""
generation_backend.py
------------------------
embedding_backend.py ki tarah, is module ka maqsad hai GENERATION provider
ko bhi "pluggable" banana — taake:

  1. Default/recommended path Gemini rahe (native structured-output support,
     free tier, aur ab tak jo bhi testing hui hai wo isi par hui hai).
  2. Agar kabhi Gemini ka free-tier quota khatam ho jaye (jaise exam week
     mein bhari traffic ke doran), ek FALLBACK provider switch-on kiya ja
     sake — jaise AgentRouter (agentrouter.org) — bina app.py ka code
     chhede.

IMPORTANT — AgentRouter (ya koi bhi third-party "gateway") ke baare mein:
  - Ye ek unverified, non-profit third-party proxy hai jo aapki requests
    asal model provider (Anthropic/OpenAI/etc.) tak forward karta hai.
    Iski koi published privacy/data-retention policy maujood nahi — student
    sawal-jawab is gateway se guzrenge, jo aapke direct control mein nahi.
  - Isse "primary/only" backbone na banayein — sirf FALLBACK ke tor par
    use karein jab Gemini quota khatam ho jaye. Isliye is file mein
    GENERATION_PROVIDER = "gemini" hi default hai.
  - Maine (Claude) is integration ko sandbox ki network restrictions ki
    wajah se live test NAHI kar saka (agentrouter.org allowed domains
    mein nahi tha) — sirf OpenAI-compatible /v1/chat/completions ke
    standard, well-documented contract ke mutabiq likha hai. Deploy karne
    se PEHLE khud ek chhota manual test zaroor karein:
        python3 verify_fallback_provider.py
  - Apni key kabhi bhi chat mein ya code mein hardcode karke commit na
    karein — sirf config.py (gitignored) ya Streamlit Secrets mein.
"""

from __future__ import annotations

import abc
import json
import re
import time
from typing import Type, TypeVar

from pydantic import BaseModel

from core import repair_json_escaping

T = TypeVar("T", bound=BaseModel)


class GenerationBackend(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def generate(self, system_instruction: str, prompt: str, response_schema: Type[T]) -> T:
        """Ek structured (pydantic-validated) jawab generate karta hai."""


def _strip_markdown_fences(text: str) -> str:
    """Kuch models (khaaskar non-OpenAI models jo OpenAI-compatible gateway
    ke peeche hain) JSON-mode maangne ke bawajood ```json ... ``` fences
    mein wrap kar dete hain — isse defensively hata dete hain."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


class GeminiGenerationBackend(GenerationBackend):
    """Default/recommended — Gemini ki Interactions API, native JSON-schema
    structured output ke saath."""

    name = "gemini"

    def __init__(self, client, model: str = "gemini-3.6-flash"):
        self._client = client
        self._model = model

    def generate(self, system_instruction: str, prompt: str, response_schema: Type[T]) -> T:
        interaction = self._client.interactions.create(
            model=self._model,
            system_instruction=system_instruction,
            input=prompt,
            # FIX: Gemini docs ke mutabiq response_format ek list honi
            # chahiye jismein {"type","mime_type","schema"} keys hon —
            # pehle raw schema seedha pass ho rahi thi.
            response_format=[
                {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": response_schema.model_json_schema(),
                }
            ],
        )
        # FIX (production bug, Aug 2026): Gemini kabhi LaTeX commands
        # (\times, \buildrel, waghera) mein backslash double-escape
        # karna bhool jata hai — is se jawab corrupt ho jata hai (kabhi
        # silently, kabhi crash ke saath). Parse karne se pehle repair
        # karte hain — dekhein core.repair_json_escaping().
        repaired_text = repair_json_escaping(interaction.output_text)
        return response_schema.model_validate_json(repaired_text)


class OpenAICompatibleGenerationBackend(GenerationBackend):
    """Kisi bhi OpenAI-Chat-Completions-compatible gateway ke liye — jaise
    AgentRouter (agentrouter.org/v1), ya kal ko OpenRouter/Groq bhi isi
    class se chal jayenge, bas base_url/model badalna hoga.

    NOTE: har underlying model strict JSON-schema enforcement support
    nahi karta (ye AgentRouter jaisi gateway ke peeche 30+ alag models
    ho sakte hain) — isliye hum widely-supported "JSON mode"
    (`response_format: {"type": "json_object"}`) use karte hain, aur
    schema ko system-instruction mein explicitly bhi likh dete hain taake
    model ko poori tarah pata ho kya chahiye. Agar phir bhi parsing fail
    ho (kabhi kabhi kisi model se markdown-fenced JSON aa jata hai), ek
    retry try karte hain."""

    name = "openai_compatible"

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 40, retries: int = 2):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._retries = retries

    def generate(self, system_instruction: str, prompt: str, response_schema: Type[T]) -> T:
        import requests  # lazy import — sirf tab chahiye jab ye backend use ho

        schema = response_schema.model_json_schema()
        full_system = (
            f"{system_instruction}\n\n"
            "You MUST reply with ONLY a single valid JSON object matching this "
            "JSON Schema, and nothing else — no markdown code fences, no "
            f"commentary before or after:\n{json.dumps(schema)}"
        )

        last_error = None
        for attempt in range(self._retries):
            try:
                resp = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": full_system},
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.2,
                    },
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                content = _strip_markdown_fences(content)
                # FIX (production bug, Aug 2026): same LaTeX-backslash
                # escaping issue can happen with any underlying model —
                # dekhein core.repair_json_escaping().
                content = repair_json_escaping(content)
                return response_schema.model_validate_json(content)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self._retries - 1:
                    time.sleep(2)
        raise last_error


# Known OpenAI-compatible gateways — base_url shortcuts taake config.py
# mein poora URL yaad na rakhna pade. Agar koi aur gateway use karna ho,
# GENERATION_PROVIDER="openai_compatible" ke saath GENERATION_BASE_URL
# seedha config.py mein de sakte hain.
KNOWN_GATEWAYS = {
    "agentrouter": "https://agentrouter.org/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
}


def get_generation_backend(provider: str, **kwargs) -> GenerationBackend:
    provider = (provider or "gemini").lower().strip()
    if provider == "gemini":
        return GeminiGenerationBackend(client=kwargs["client"], model=kwargs.get("model", "gemini-3.6-flash"))

    base_url = kwargs.get("base_url") or KNOWN_GATEWAYS.get(provider)
    if base_url is None:
        raise ValueError(
            f"Unknown provider {provider!r} — ya to 'gemini', ya "
            f"{list(KNOWN_GATEWAYS)} mein se ek, ya 'openai_compatible' "
            "(base_url khud dein)."
        )
    if "api_key" not in kwargs or not kwargs["api_key"]:
        raise ValueError(f"{provider} ke liye api_key chahiye (config.py mein set karein).")
    if "model" not in kwargs or not kwargs["model"]:
        raise ValueError(
            f"{provider} ke liye model naam chahiye — apne AgentRouter console "
            "mein dekh kar config.py mein GENERATION_MODEL_FALLBACK set karein "
            "(e.g. 'claude-sonnet-4-5-20250929' ya 'gpt-4o-mini')."
        )
    return OpenAICompatibleGenerationBackend(api_key=kwargs["api_key"], base_url=base_url, model=kwargs["model"])
