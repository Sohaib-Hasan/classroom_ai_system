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

## Post-deployment fixes (Aug 2026) — real bugs found in production use

**11. LaTeX commands in AI answers corrupted to garbled text**
Reported: student saw text like "imes" instead of "×", "uildrel" instead
of part of a LaTeX command, in the Roman Urdu explanation. Root cause
(confirmed by reproduction): Gemini sometimes writes LaTeX commands
(`\times`, `\buildrel`, `\pmod`) in the JSON response without properly
double-escaping the backslash. JSON then interprets `\t`/`\b`/etc. as
actual control characters (tab, backspace), silently eating the first
letter — or, for letters that aren't valid JSON escapes, crashes the
whole answer. Fix: `core.repair_json_escaping()` repairs the raw text
before parsing (matches a curated list of known LaTeX command names,
deliberately NOT a blanket regex — a blanket approach would have
corrupted legitimate `\n` newlines followed by ordinary words, which a
test caught). Also strengthened `SYSTEM_INSTRUCTION` to ask the model to
avoid LaTeX/backslash notation in prose fields entirely, using plain
symbols (×, ≡, √) instead — defense in depth alongside the repair
function. Verified: `tests/test_core.py::TestRepairJsonEscaping`,
reproducing the exact reported corruption pattern.

**12. Teacher dashboard showed no questions despite students using the app**
Reported: dashboard empty even after students asked questions. Root
cause: `app.py` and `dashboard.py` were deployed as two SEPARATE
Streamlit Cloud apps — each gets its own isolated, ephemeral container,
so local files written by one are invisible to the other (and don't
survive restarts even for the same app). Fix: `db_connection.py` — a
pluggable connection that uses local SQLite by default (unchanged
behaviour for single-app setups) or a shared, hosted Turso database when
`TURSO_DATABASE_URL`/`TURSO_AUTH_TOKEN` are configured in BOTH apps'
secrets. `question_log_store.py` replaces the CSV-based logging (which
had the same cross-app-invisibility problem) with a SQL table on the
same shared connection. Verified: mocked unit tests for the Turso
adapter (`tests/test_db_connection.py`), plus a live simulation proving
one AppTest instance's logged question was correctly read by a
completely separate AppTest instance via a shared local file (same
mechanism Turso uses, minus the network hop).
⚠️ The actual Turso network connection was NOT live-tested (turso.tech
not reachable from the development sandbox) — run
`python3 verify_turso_connection.py` with real credentials before
relying on it.

## Verification summary (kya actually test hua)

- ✅ `pytest tests/` — 91/91 pass
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
