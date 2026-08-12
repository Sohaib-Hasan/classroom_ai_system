"""
embedding_backend.py
-----------------------
Embedding provider ko "pluggable" banata hai — taake zero-budget setup
mein aap chahein to Gemini free-tier embedding quota bilkul use hi na
karein, aur uski jagah apne hi machine pe chalne wala free/open-source
model use kar lein.

Kyun zaroori hai (previous review se): app ka HAR sawal — chahe wo cache
se mil jaye ya nahi — pehle ek embedding call zaroor karta hai (cache-
match check karne ke liye bhi query ka embedding chahiye hota hai). Matlab
embedding quota hi asal bottleneck hai, generation quota nahi (wo caching
se bach jati hai). Isliye embedding ko free/local model par shift karna
sabse zyada faida deta hai.

IMPORTANT — dono backends ke embeddings AAPAS MEIN COMPATIBLE NAHI HAIN.
Gemini aur local model alag-alag vector-space produce karte hain; cosine
similarity sirf tab meaningful hai jab query aur saare stored chunks EK
HI backend se embed hue hon. Agar aap backend badlein:
    1. config.py mein EMBEDDING_PROVIDER change karein
    2. `python3 embed_chunks.py --rebuild` chalayein (poori knowledge base
       naye backend se dobara embed hogi)
    3. purani knowledge_base.json istemal na karein — wo purane backend
       ki hai, naye ke saath cosine similarity bekar (garbage) result
       degi.

Usage:
    from embedding_backend import get_backend
    backend = get_backend("gemini", client=genai_client)          # ya
    backend = get_backend("local")
    vec = backend.embed_query("differentiate x^2")
    vec = backend.embed_document("Definition: a derivative is ...")
"""

from __future__ import annotations

import abc
from typing import Optional

from core import MAX_EMBED_CHARS, truncate_for_embedding


class EmbeddingBackend(abc.ABC):
    """Common interface — dono backends (Gemini aur local) isi shape ko
    follow karte hain, taake app.py/embed_chunks.py ko backend ka naam
    tak pata na hona pade."""

    name: str = "base"
    dimensions: Optional[int] = None

    @abc.abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Student ke sawal ko embed karta hai (retrieval query)."""

    @abc.abstractmethod
    def embed_document(self, text: str) -> list[float]:
        """Ek note-chunk ko embed karta hai (retrieval document).
        Kuch models (jaise Gemini) query vs document ke liye alag
        internal task_type use karte hain — isliye ye do separate
        methods hain, ek hi `embed()` nahi."""


class GeminiEmbeddingBackend(EmbeddingBackend):
    """Default/existing behaviour — Google Gemini ka gemini-embedding-001
    (free tier available hai, lekin apni rate-limit/RPD hoti hai)."""

    name = "gemini"
    dimensions = 3072  # default; agar output_dimensionality set kiya to badal sakta hai

    def __init__(self, client=None, clients=None, model: str = "gemini-embedding-001", retries: int = 2, timeout: float = 15):
        # FIX (quota-scaling): ab ek se zyada API keys (alag Google
        # accounts se, isi liye alag quota pools) rotate kar sakte hain.
        # `client=` (purana, singular) ab bhi chalta hai — sirf 1-item
        # list ban jati hai, taake purana code/tests na tootein.
        if clients:
            self._clients = list(clients)
        elif client is not None:
            self._clients = [client]
        else:
            raise ValueError("GeminiEmbeddingBackend ke liye 'client' ya 'clients' chahiye.")
        self._model = model
        self._retries = retries
        self._timeout = timeout

    def _call(self, text: str, task_type: str) -> list[float]:
        from google.genai import types
        import time

        text, was_truncated = truncate_for_embedding(text)
        if was_truncated:
            # Silent truncation nahi chahte — caller warning dekh sake
            # is liye ek marker attribute set kar dete hain. app.py ise
            # log kar sakta hai.
            pass

        last_error = None
        # FIX (quota-scaling): har client (= har API key/quota-pool) ko
        # apne retries mil jate hain; agar EK key ka quota khatam ho
        # jaye, agli key try hoti hai — student ko koi farq nazar nahi
        # aata, jab tak sab keys ka quota ek sath khatam na ho jaye.
        for client in self._clients:
            for attempt in range(self._retries):
                try:
                    result = client.models.embed_content(
                        model=self._model,
                        contents=text,
                        config=types.EmbedContentConfig(task_type=task_type),
                    )
                    return list(result.embeddings[0].values)
                except Exception as e:  # noqa: BLE001 — retry loop, phir raise
                    last_error = e
                    if attempt < self._retries - 1:
                        time.sleep(2)
        raise last_error

    def embed_query(self, text: str) -> list[float]:
        return self._call(text, "RETRIEVAL_QUERY")

    def embed_document(self, text: str) -> list[float]:
        return self._call(text, "RETRIEVAL_DOCUMENT")


class LocalEmbeddingBackend(EmbeddingBackend):
    """Bilkul FREE, koi API key nahi chahiye, koi rate-limit nahi, koi
    internet chahiye (model download ke baad) — `sentence-transformers`
    library se chalta hai, aapke apne CPU/GPU par.

    Default model multilingual hai (`intfloat/multilingual-e5-small`)
    kyunke aapke students Roman Urdu mein bhi sawal karte hain — English-
    only models (jaise all-MiniLM-L6-v2) Roman Urdu par utni acchi
    performance nahi dete.

    NOTE: e5 family models ko convention ke tor par "query: " / "passage: "
    prefix ke saath best results milte hain (unke documentation/paper ke
    mutabiq) — isliye embed_query aur embed_document mein ye prefix add
    kiya gaya hai.

    Install: pip install -r requirements-local-embeddings.txt
    (ye alag file hai kyunke sentence-transformers + torch bhari
    dependencies hain (~1-2 GB) — sab users ko ye nahi chahiye hoga)
    """

    name = "local"
    dimensions = 384  # multilingual-e5-small ka output dimension

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Local embeddings ke liye 'sentence-transformers' install karein:\n"
                "    pip install -r requirements-local-embeddings.txt\n"
                "(Ye ~1-2 GB download hai kyunke isme PyTorch shamil hai.)"
            ) from e
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        text, _ = truncate_for_embedding(text, max_chars=MAX_EMBED_CHARS)
        vec = self._model.encode(f"query: {text}", normalize_embeddings=True)
        return vec.tolist()

    def embed_document(self, text: str) -> list[float]:
        text, _ = truncate_for_embedding(text, max_chars=MAX_EMBED_CHARS)
        vec = self._model.encode(f"passage: {text}", normalize_embeddings=True)
        return vec.tolist()


def get_backend(provider: str, client=None, clients=None, **kwargs) -> EmbeddingBackend:
    """Factory function. `provider` config.py se aata hai ("gemini" ya
    "local"). Gemini backend ke liye `client` (ek) ya `clients` (list —
    quota-rotation ke liye, dekhein GeminiEmbeddingBackend) chahiye."""
    provider = (provider or "gemini").lower().strip()
    if provider == "gemini":
        if client is None and not clients:
            raise ValueError("GeminiEmbeddingBackend ke liye ek genai.Client instance chahiye (client=... ya clients=[...]).")
        return GeminiEmbeddingBackend(client=client, clients=clients, **kwargs)
    elif provider == "local":
        return LocalEmbeddingBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider!r}. Sirf 'gemini' ya 'local' valid hain."
        )
