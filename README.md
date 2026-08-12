# Classroom AI System

A Streamlit-based doubt-clearing assistant for undergraduate math courses.
It answers strictly from your own course notes (retrieval-augmented
generation) — it does not answer from the model's general knowledge — and
gives teachers visibility into what students are actually struggling with.

The system is two separate apps sharing one knowledge base and one
activity log:

| App | Audience | Entry point |
|---|---|---|
| Student assistant | Students, PIN-gated | `app.py` |
| Teacher dashboard | Teacher, password-gated | `dashboard.py` |

## How it works, end to end

```
 .tex course notes
        │  chunk_notes.py
        ▼
 *_chunks.json  (raw chunks, one per definition/example/theorem/proof)
        │  clean_chunks.py
        ▼
 *_chunks.cleaned.json  (LaTeX formatting/decoration stripped, real math kept)
        │  embed_chunks.py
        ▼
 knowledge_base.json  (every chunk + its embedding vector)
        │
        ▼
 app.py  ──(question embedding + cosine similarity)──►  matching chunks
        │                                                       │
        └──(chunks + question → Gemini)──► answer ◄─────────────┘
        │
        └──(question + answer)──► question_log  ──► dashboard.py
```

Both apps read the same `knowledge_base.json` and, when deployed
separately (see [Shared storage](#shared-storage-turso) below), the same
`question_log` table — that's what lets the dashboard show real student
activity from the student app.

## Repository layout

| File | Purpose |
|---|---|
| `app.py` | Student-facing chat UI — the only file that calls the Gemini API live, per question |
| `dashboard.py` | Teacher-only analytics view (topics, trends, cache effectiveness, repeated-confusion signals) |
| `chunk_notes.py` | Splits `.tex` course notes into chunks (one per definition/example/theorem/proof box) |
| `clean_chunks.py` | Strips decorative LaTeX (colors, tables, icons, diagrams) out of chunk content; keeps real `$...$` math untouched. **Required** before embedding — see [Building the knowledge base](#building-the-knowledge-base) |
| `embed_chunks.py` | Embeds every cleaned chunk into `knowledge_base.json`; resumable, content-aware (re-embeds only chunks whose text actually changed) |
| `core.py` | All business logic — no `import streamlit`, so it's fully unit-testable |
| `embedding_backend.py` | Embedding provider abstraction (Gemini, or a free local model); supports multi-key rotation |
| `generation_backend.py` | Answer-generation provider abstraction (Gemini, with an optional third-party fallback); supports multi-key rotation |
| `cache_store.py` | SQLite-backed Q&A cache — repeated/rephrased questions don't re-hit the API |
| `db_connection.py` | Pluggable storage connection: local SQLite for single-app setups, or Turso for two separately-deployed apps that need to share data |
| `question_log_store.py` | Records every question asked — the dashboard's data source |
| `auth_guard.py` | Brute-force lockout for the PIN/password screens (5 attempts, 60s lockout; session-scoped) |
| `logging_setup.py` | Error logging — writes to both `logs/error.log` and stdout (visible in Streamlit Cloud's "Manage app → Logs") |
| `knowledge_base_loader.py` | Loads `knowledge_base.json` at app startup |
| `list_models.py` | Diagnostic — lists the Gemini models available to your API key |
| `verify_turso_connection.py` | Manual smoke test for the Turso connection before deploying |
| `verify_fallback_provider.py` | Manual smoke test for the optional third-party fallback provider |
| `tests/` | Automated tests (pytest) — 91 tests, covering every bug fixed so far so it can't silently regress |

## First-time setup

```bash
pip install -r requirements.txt
cp config.py.example config.py
```

Fill in `config.py`:
- `GEMINI_API_KEY` — from https://aistudio.google.com/apikey
- `CLASS_PIN` — shared PIN students use to open the app
- `TEACHER_PASSWORD` — for `dashboard.py` only

`config.py` is for **local development only** and is git-ignored. On
Streamlit Cloud, the same values go into each app's **Settings → Secrets**
instead (see [Deploying](#deploying-to-streamlit-cloud)).

## Building the knowledge base

Three steps, in order, whenever your `.tex` notes change:

```bash
python3 chunk_notes.py      # .tex notes  ->  *_chunks.json
python3 clean_chunks.py     # *_chunks.json  ->  *_chunks.cleaned.json
python3 embed_chunks.py     # *_chunks.cleaned.json  ->  knowledge_base.json
```

**Don't skip `clean_chunks.py`.** `chunk_notes.py` extracts each
definition/example/theorem box as raw `.tex` source — decorative LaTeX
(`\textcolor`, `\begin{tabular}`, `\begin{tikzpicture}`, icon macros,
`\vspace`, etc.) included. That raw text becomes part of the AI's prompt
for every question. If it isn't cleaned first, the model occasionally
echoes fragments of it back verbatim — which shows up to students as
literal LaTeX code instead of a rendered explanation. `embed_chunks.py`
reads from `*_chunks.cleaned.json`, **not** `*_chunks.json` — running it
straight after `chunk_notes.py` without cleaning in between will silently
embed the dirty content.

**Re-embedding is resumable and content-aware.** `embed_chunks.py`
identifies each chunk by `course|chapter|title|content-hash`. Run it
normally (no flags) after any edit to your notes, and it will only
re-embed chunks whose *content* actually changed — everything else keeps
its existing embedding, at no extra API quota cost:

```bash
python3 embed_chunks.py
```

Use `--rebuild` only when you deliberately want to re-embed *everything*
— for example, after switching `EMBEDDING_PROVIDER` between `"gemini"`
and `"local"` (their embeddings are not compatible with each other, so a
partial mix would silently corrupt retrieval):

```bash
python3 embed_chunks.py --rebuild
```

## Running locally

```bash
streamlit run app.py         # student chat, http://localhost:8501
streamlit run dashboard.py   # teacher dashboard, on another port
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

91 tests, covering every bug that has been found and fixed in this
project so far (see `CHANGELOG.md` for the history) — each one exists
specifically so that bug can't silently come back.

## Deploying to Streamlit Cloud

1. Push the repo to GitHub. **Never commit `config.py`** — it's
   git-ignored already; double-check before your first push.
2. On Streamlit Cloud, create one app per entry point (`app.py` and
   `dashboard.py`), pointing at the same repo.
3. In each app's **Settings → Secrets**, add the same keys as
   `config.py`, in TOML format:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   CLASS_PIN = "1234"
   TEACHER_PASSWORD = "something-strong"
   ```
   Every value must be in double quotes — an unquoted value is invalid
   TOML and Streamlit will reject the whole Secrets file with
   `Invalid format: please enter valid TOML`.
4. If `knowledge_base.json` is tracked via Git LFS (it will be once it
   grows past GitHub's ~100 MB plain-file limit — see `.gitattributes`),
   Streamlit Cloud resolves LFS automatically. It's still worth a
   one-time check after your first deploy that the app actually loaded
   the full file and not just the small LFS pointer text — an
   unexpectedly small knowledge base, or every question returning
   "not found in your notes," is the symptom to watch for.

### Shared storage (Turso)

If `app.py` and `dashboard.py` are deployed as **two separate** Streamlit
Cloud apps (two different URLs — the normal setup, and what this project
uses), the dashboard will show no data by default. Each Streamlit Cloud
app runs in its own isolated container, so the local SQLite file one app
writes to is invisible to the other.

Fix: point both apps at the same [Turso](https://turso.tech) database
(free, hosted, SQLite-compatible). In **both** apps' Secrets:

```toml
TURSO_DATABASE_URL = "https://your-db-name-your-org.turso.io"
TURSO_AUTH_TOKEN = "your-token"
```

Two details that matter and are easy to get wrong:

- **Use `https://`, not `libsql://`, for `TURSO_DATABASE_URL`.**
  `libsql://` is equivalent to `wss://` (WebSocket) under the hood. On
  regional Turso database URLs specifically, the Python client's
  WebSocket handshake can fail in sandboxed environments like Streamlit
  Cloud (`aiohttp.client_exceptions.WSServerHandshakeError`). `https://`
  uses a plain HTTP connection instead and avoids this entirely — the
  only capability it gives up is the client's `transaction()` /
  `batch()` API, which this codebase never uses, so there's no
  functional downside.
- **The key must be named `TURSO_AUTH_TOKEN`, not `TURSO_TOKEN` or
  anything else.** `db_connection.py` only activates Turso when *both*
  `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` are present; if either is
  missing (including just being misnamed), it silently falls back to
  local SQLite — no error, no crash, just a dashboard that stays empty
  with no indication why.

After setting or changing these secrets, **reboot both apps** (Manage
app → ⋮ → Reboot), not just a page refresh. `get_shared_db_connection()`
is cached per-process (`@st.cache_resource`); without a real process
restart it keeps using whatever connection it built on first run, even
after the secrets change.

To verify the connection independently of the apps (useful for
diagnosing "is it a Streamlit problem or a Turso problem"), query it
directly over HTTP — no CLI install needed:

```bash
curl -s https://your-db-name-your-org.turso.io/v2/pipeline \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"requests":[{"type":"execute","stmt":{"sql":"SELECT COUNT(*) FROM question_log"}},{"type":"close"}]}'
```

(The Turso CLI's install script does not officially support Windows
Git Bash/MINGW — it requires WSL. The HTTP API above works from any
shell and needs no local install.)

Alternatively, if you run both apps as pages of a single Streamlit app
(one deployment, one container), local storage works fine without Turso
— though it isn't guaranteed to persist across restarts on the free
tier.

## Math rendering

Answers are shown with `st.markdown()`, which renders `$...$` (inline)
and `$$...$$` (block) LaTeX math via KaTeX natively. The model is
instructed (`core.py`'s system prompt) to wrap every mathematical
expression or symbol in real `$...$` LaTeX — `$\phi(mn)$`,
`$\frac{d}{dx}$`, `$\gcd(m,n)$` — rather than spelling things out in
plain words, so students see the same typeset notation they'd see in
their printed notes.

This depends on the knowledge base being clean (see
[Building the knowledge base](#building-the-knowledge-base) above) — the
model tends to mirror the formatting style of whatever notes context
it's given, so decorative LaTeX leaking through from uncleaned chunks is
the most common cause of broken-looking answers.

## Scaling to more students: API quota

Every question makes at least one Gemini call (embedding the question,
to check the cache — this runs even on a cache hit) and, on a cache
miss, a second call (generating the answer). As enrolled students and
courses grow, a single free-tier API key's daily quota becomes the
limiting factor.

### Multi-key rotation (recommended)

Gemini's rate limits are applied **per project, not per API key** — keys
from different Google accounts draw from independent quota pools. The
app supports configuring up to three keys; it tries them in order and
moves to the next one only if a call fails, so this is completely
invisible to students:

```toml
GEMINI_API_KEY   = "key from account 1"
GEMINI_API_KEY_2 = "key from account 2"   # optional
GEMINI_API_KEY_3 = "key from account 3"   # optional
```

Leaving `GEMINI_API_KEY_2`/`_3` unset behaves exactly as with a single
key — this is fully backward compatible.

### Per-session rate limit

Independent of how many keys are configured, `app.py` caps each browser
session at 8 questions per rolling 60-second window (`QUESTION_RATE_LIMIT`
/ `QUESTION_RATE_WINDOW_SECONDS` near the top of the question-handling
code). This is a safety net against accidental double-submits or a
single session draining quota — normal usage (one question at a time)
never comes close to it.

### Caching

`cache_store.py` already means a repeated or rephrased question that
matches a previous one (same course, similar embedding) skips the
generation call entirely — in a class where many students ask similar
questions about the same material, this absorbs a significant fraction
of traffic before it ever reaches the API.

### Local embeddings (zero Gemini quota for retrieval)

Since the embedding call happens on *every* question (unlike generation,
which caching can skip), it's the bigger quota consumer. It can be moved
to a free local model instead, at no ongoing cost:

```bash
pip install -r requirements-local-embeddings.txt
```
Then set `EMBEDDING_PROVIDER = "local"` and rebuild:
```bash
python3 embed_chunks.py --rebuild
```
`--rebuild` is required here — Gemini and local-model embeddings are not
compatible with each other, and the app cannot meaningfully compare a
locally-embedded question against a Gemini-embedded knowledge base.

### Third-party fallback provider (use with caution)

A generation-only fallback (e.g. AgentRouter, OpenRouter) can be
configured to kick in automatically if every configured Gemini key fails:

```toml
GENERATION_FALLBACK_PROVIDER = "agentrouter"
GENERATION_FALLBACK_API_KEY  = "sk-..."
GENERATION_FALLBACK_MODEL    = "claude-sonnet-4-5-20250929"
```

Treat this as a **last-resort fallback, not a primary or scaling
strategy**:
- These are unverified third-party proxies with no published
  data-retention policy — student questions would pass through them,
  outside your direct control.
- The integration uses an OpenAI-compatible `/v1/chat/completions`
  contract and has not been live-tested end-to-end. Run
  `python3 verify_fallback_provider.py` yourself before relying on it.
- Never commit a fallback-provider API key. If one is ever pasted into
  a chat, screenshot, or commit, treat it as compromised and regenerate
  it immediately.

## Architecture notes

- `core.py` has no `import streamlit`, deliberately — business logic is
  fully testable without running the Streamlit app.
- The Q&A cache is SQLite (`cache_store.py`), not flat JSON — the
  original design rewrote the entire cache file on every save.
- `verify_computation()` samples the negative, positive, *and*
  near-zero domains, not just positive — catching domain-sensitive
  errors like `sqrt(x**2) == x` (only true for `x >= 0`) that a
  positive-only check would miss.
- A logging failure never hides a successfully-generated answer:
  `log_question()` runs in its own `try/except`, separate from the
  answer-display code, so a transient logging/storage issue produces a
  quiet log entry instead of silently discarding an answer the student
  already has.

## Known limitations

- **Nested same-type LaTeX boxes** (e.g. a `definitionbox` containing
  another `definitionbox`): `chunk_notes.py`'s parser doesn't cleanly
  separate these — the inner box doesn't get its own chunk, and its raw
  markup leaks into the outer chunk's content. A safety-net check
  (`check_for_leaked_box_markup`) flags this loudly whenever
  `chunk_notes.py`/`embed_chunks.py` run, so it can't silently reach
  production. If your notes have this nesting, restructure the inner
  box to a different box type as a workaround.
- `clean_chunks.py` handles the vast majority of decorative LaTeX but is
  not exhaustive — a small residual (roughly 3% of chunks, mostly
  `\textcolor`/`\cellcolor` used *inside* real math, or diagrams nested
  in unusual ways) can still need a manual look. It prints a per-file
  before/after count on every run; grep its `*.cleaned.json` output for
  `textcolor|textbf|tikzpicture` to find any that remain.
- The structural cache-safety signature (`structural_signature` in
  `core.py`) captures bracket-nesting/grouping but doesn't formally
  guarantee that two structurally-different, textually-identical inputs
  can never be confused — a very low-probability theoretical edge case.
- The third-party fallback provider integration (see above) has not
  been live-tested — verify it yourself before depending on it.

## Further reading

- `CHANGELOG.md` — full history of bugs found and fixed, each with the
  evidence (test or live reproduction) that confirmed it, and the test
  that now guards against it recurring.
