"""
verify_fallback_provider.py
------------------------------
Ye ek MANUAL smoke-test hai — pytest suite ka hissa nahi (kyunke ye
asli internet aur asli API key maangta hai). Maine (Claude) is
integration ko development sandbox mein live test NAHI kiya, kyunke
agentrouter.org wahan allowed domains mein nahi tha — code OpenAI-
compatible /v1/chat/completions ke documented contract ke mutabiq likha
gaya hai, lekin deploy se pehle isse khud ek dafa chalana zaroori hai.

Chalane se pehle:
    1. config.py mein GENERATION_FALLBACK_PROVIDER, _API_KEY, _MODEL set karein.
       (Agar aapne apni key kabhi chat/screenshot/kisi aur jagah share ki
       thi, pehle agentrouter.org/console/token pe jaa kar use REVOKE aur
       naya banayein — purani ab compromised maani jani chahiye.)
    2. Apne AgentRouter console mein confirm karein ke aapne jo model-naam
       diya hai wo actually available hai (jaise "claude-sonnet-4-5-20250929"
       ya "gpt-4o-mini") — console ki model list dekhein.

Chalana:
    python3 verify_fallback_provider.py
"""

import sys

from config import GENERATION_FALLBACK_API_KEY, GENERATION_FALLBACK_MODEL, GENERATION_FALLBACK_PROVIDER
from core import TutorAnswer
from generation_backend import get_generation_backend


def main():
    if not GENERATION_FALLBACK_PROVIDER:
        print("config.py mein GENERATION_FALLBACK_PROVIDER set nahi hai — kuch test karne ko nahi hai.")
        sys.exit(0)

    print(f"Testing fallback provider: {GENERATION_FALLBACK_PROVIDER} (model: {GENERATION_FALLBACK_MODEL})\n")

    backend = get_generation_backend(
        GENERATION_FALLBACK_PROVIDER,
        api_key=GENERATION_FALLBACK_API_KEY,
        model=GENERATION_FALLBACK_MODEL,
    )

    try:
        result = backend.generate(
            system_instruction="You are a helpful assistant. Reply with valid JSON only.",
            prompt="What is 2 + 2? Explain briefly.",
            response_schema=TutorAnswer,
        )
        print("✅ Success! Parsed response:")
        print(f"   english: {result.english[:200]}")
        print(f"   grounding: {result.grounding}")
    except Exception as e:
        print(f"❌ Failed: {e!r}")
        print(
            "\nCommon causes:\n"
            "  - Model naam galat hai (apne AgentRouter console mein exact "
            "naam confirm karein)\n"
            "  - API key invalid/revoked hai\n"
            "  - Provider ka underlying model strict JSON mode support "
            "nahi karta — kisi doosre model se try karein\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
