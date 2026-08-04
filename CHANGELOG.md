# Changelog — bug-fix pass (Aug 2026)

Ye sab fixes ek skeptical-engineer review ke baad kiye gaye — har ek ka
evidence hai (test, ya live simulation), guesswork nahi. Har fix ko
`tests/` mein ek regression test se cover kiya gaya hai.

## Critical fixes

**1. `response_format` Gemini Interactions API ke contract se match nahi
karta tha**
Pehle: `response_format=TutorAnswer.model_json_schema()` (raw dict).
Ab: `response_format=[{"type": "text", "mime_type": "application/json",
"schema": ...}]` — Google ki official migration docs ke exact example ke
mutabiq. Verified: `tests/test_generation_backend.py::test_calls_interactions_create_with_wrapped_response_format`,
aur live simulation mein `interactions.create` call ke args directly
inspect kiye gaye.

**2. SymPy verification sirf positive numbers se sample karta tha —
domain-sensitive galtiyan pakadta nahi tha**
Reproduced: `verify_computation('sqrt(x**2)', 'x')` → `True` (GALAT —
sirf x>=0 ke liye sach hai). Fix: sampling ab negative, positive, aur
near-zero — teenon regions se hoti hai. Verified:
`tests/test_core.py::TestVerifyComputationDomainBug` (20 trials, kam se
kam ek False expected).

**3. Cross-course fallback AI ko galat (apne kam-confidence course ke)
context ke saath call kar deta tha, quota waste karte hue**
Fix: `core.decide_retrieval_strategy()` — agar doosra course clearly
(diff > 0.15) behtar match karta ho, AI ko call hi nahi karte, seedha
redirect dikhate hain. Verified live (AppTest simulation):
`interactions.create` call na hone ki confirm ki gayi jab course clearly
galat tha.

## Scalability fixes

**4. Cache flat-JSON mein thi — har save par poori file (sab embeddings
samet) rewrite hoti thi**
Fix: SQLite (`cache_store.py`) — incremental inserts. Verified:
`tests/test_cache_store.py`, aur live simulation mein 2 alag sessions ke
beech cache-hit confirm kiya (paraphrase, same numbers → dusri baar
`interactions.create` call hi nahi hui).

**5. `requirements.txt` unpinned tha, ek actively-evolving beta SDK
(`google-genai`, jismein already ek breaking-change round ho chuka hai)
ke saath**
Fix: sab versions pin kiye (jo actually test hui).

## Security/hardening fixes

**6. PIN/teacher-password par koi rate-limiting/lockout nahi tha**
Fix: `auth_guard.py` — 5 galat attempts ke baad 60-second lockout.
Verified: `tests/test_auth_guard.py`, aur live simulation mein confirm
kiya ke 5 wrong attempts ke baad PIN input field hi gayab ho jata hai.

**7. Errors silently swallow ho rahe the — koi log nahi**
Fix: `logging_setup.py` — `logs/error.log` mein poora traceback likha
jata hai.

## Minor fixes

**8. `embed_chunks.py` mein ek chunk (2172 mein se 1) gemini-embedding-001
ki ~2048-token limit se upar tha**
Fix: `core.truncate_for_embedding()` — safe character-budget truncation,
aur console warning agar truncate hua.

**9. `chunk_notes.py` mein same-type nested boxes ka content leak ho jata
tha (discovered while writing tests — pehle se zyada messy nikla jitna
guess kiya tha)**
Fix: parsing logic nahi badli (real `.tex` fixtures nahi hain safe fix ke
liye), lekin ek loud warning add ki (`check_for_leaked_box_markup`) taake
ye chup-chaap knowledge base mein na jaye. Verified:
`tests/test_chunk_notes.py::TestSameTypeNestedBoxes`.

**10. `dashboard.py` mein `use_container_width` (Streamlit se deprecated,
already removal-date cross ho chuka tha) use ho raha tha**
Fix: `width='stretch'` mein badla — discovered via live AppTest simulation
(deprecation warning dikhi), guess se nahi.

## Naya (additive) — zero-budget resilience

- `embedding_backend.py` — pluggable embedding provider (Gemini + free
  local model via `sentence-transformers`).
- `generation_backend.py` — pluggable generation provider (Gemini +
  optional OpenAI-compatible fallback, jaise AgentRouter, sirf backup ke
  tor par).

## Verification summary (kya actually test hua)

- ✅ `pytest tests/` — 70/70 pass
- ✅ `py_compile` — sab files, koi syntax error nahi
- ✅ `pyflakes` — clean
- ✅ Live Streamlit simulation (`streamlit.testing.v1.AppTest`, mocked
  network) — PIN gate, lockout, question→answer full pipeline, cache
  hit/miss, not_found path, cross_course_redirect path, dashboard.py —
  sab 0 exceptions ke saath
- ⚠️ Live Gemini API call — NAHI test hua (sandbox mein network
  restricted, koi Google domain allowed nahi)
- ⚠️ Live AgentRouter/fallback call — NAHI test hua (same wajah). Deploy
  se pehle `python3 verify_fallback_provider.py` khud chalayein.
