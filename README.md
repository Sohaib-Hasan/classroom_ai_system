# Classroom AI System

A Streamlit-based doubt-clearing assistant that answers questions **only from your course notes**. Instead of asking a general AI model, the system first searches your course material and provides answers grounded in your notes.

## Repository Structure

| File                          | Purpose                                                                  |
| ----------------------------- | ------------------------------------------------------------------------ |
| `app.py`                      | Student chat interface                                                   |
| `dashboard.py`                | Teacher-only analytics dashboard                                         |
| `chunk_notes.py`              | Splits `.tex` lecture notes into smaller chunks                          |
| `embed_chunks.py`             | Generates embeddings for each chunk and builds `knowledge_base.json`     |
| `core.py`                     | Pure business logic (independent of Streamlit and fully testable)        |
| `embedding_backend.py`        | Embedding provider abstraction (Gemini or a free local model)            |
| `generation_backend.py`       | Generation provider abstraction (Gemini with optional fallback provider) |
| `cache_store.py`              | SQLite-backed Q&A cache                                                  |
| `auth_guard.py`               | PIN/password brute-force protection                                      |
| `logging_setup.py`            | Error logging configuration                                              |
| `knowledge_base_loader.py`    | Loads `knowledge_base.json`                                              |
| `list_models.py`              | Diagnostic tool to list models available for your API key                |
| `verify_fallback_provider.py` | Manual smoke test for the fallback provider (e.g., AgentRouter)          |
| `tests/`                      | Automated test suite (pytest)                                            |

## Initial Setup

```bash
pip install -r requirements.txt
cp config.py.example config.py
```

Open `config.py` and provide your configuration values (at minimum `GEMINI_API_KEY`, `CLASS_PIN`, and `TEACHER_PASSWORD`).

You can obtain a Gemini API key here:

https://aistudio.google.com/apikey

To build the knowledge base from your lecture notes:

```bash
python3 chunk_notes.py
python3 embed_chunks.py
```

Run the application:

```bash
streamlit run app.py         # Student interface
streamlit run dashboard.py   # Teacher dashboard
```

## Running the Tests

```bash
pip install pytest
pytest tests/ -v
```

The repository includes **70 automated tests**, covering all major bugs identified during previous code reviews (see `CHANGELOG.md`) to help prevent regressions.

---

# Zero-Budget Setup

This project is designed to operate with **minimal or zero infrastructure cost**.

There are two components that consume API quota:

* Embeddings
* Answer generation

Both can be configured to remain free-tier friendly.

## Embeddings (Retrieval)

**Important:** Every user question requires an embedding request, even when the final answer comes from the cache, because the query embedding is needed to search the cache itself.

As a result, **embeddings—not answer generation—are usually the primary quota bottleneck.**

### Option 1 — Gemini (Default)

Free, subject to Gemini's current free-tier limits.

```python
EMBEDDING_PROVIDER = "gemini"
```

### Option 2 — Local Model (Recommended if quota becomes an issue)

A completely free local embedding model with:

* No API key
* No rate limits
* Runs entirely on your own machine

Install:

```bash
pip install -r requirements-local-embeddings.txt
```

Then set:

```python
EMBEDDING_PROVIDER = "local"
```

Rebuild the knowledge base:

```bash
python3 embed_chunks.py --rebuild
```

**Important:** Rebuilding is mandatory because Gemini embeddings and local embeddings are **not compatible** with one another.

---

## Answer Generation

The default generation provider is **Gemini**.

If Gemini becomes unavailable (for example, during heavy exam-week traffic or due to free-tier quota exhaustion), an optional fallback provider (such as AgentRouter) can be configured.

Example:

```python
GENERATION_FALLBACK_PROVIDER = "agentrouter"
GENERATION_FALLBACK_API_KEY = "sk-..."
GENERATION_FALLBACK_MODEL = "claude-sonnet-4-5-20250929"
```

### Important Notes About Third-Party Gateways

Services such as AgentRouter or OpenRouter are third-party proxy providers.

Before using them:

* Do not rely on them as your primary production backend.
* Run `python3 verify_fallback_provider.py` to verify the integration before deployment.
* Never commit API keys to Git, source code, comments, screenshots, or chat messages.
* Store secrets only in `config.py` (ignored by Git) or in Streamlit Secrets.
* If an API key is ever exposed, revoke and regenerate it immediately.

---

# Architecture Notes

* `core.py` intentionally contains **no `streamlit` imports**, allowing business logic to be tested independently.
* The cache is stored in SQLite (`cache/qa_cache.db`) rather than a JSON file, avoiding complete file rewrites after every cache update.
* `verify_computation()` validates AI-generated calculations using samples from **negative, positive, and near-zero domains**, reducing false positives that can occur with domain-sensitive mathematical expressions.

---

# Known Limitations

The following limitations are intentionally documented rather than hidden.

### Nested LaTeX Boxes

If one `definitionbox` is nested inside another `definitionbox`, the parser in `chunk_notes.py` cannot separate them cleanly.

Instead:

* the inner box is not extracted as its own chunk,
* raw LaTeX markup may leak into the parent chunk.

This behavior is covered by automated tests (`tests/test_chunk_notes.py`).

The helper `check_for_leaked_box_markup` raises a warning whenever such markup is detected during `chunk_notes.py` or `embed_chunks.py`, preventing the issue from silently entering the knowledge base.

If your notes contain nested boxes, restructure them manually (for example, by using a different environment for the inner box).

---

### Structural Cache Signature

`structural_signature` in `core.py` captures bracket nesting and expression structure.

Although it significantly improves cache safety, it cannot mathematically guarantee that every structurally different but textually similar expression will always be distinguished.

Extremely rare edge cases may still exist.

---

### Fallback Provider

The AgentRouter (or any fallback provider) integration has **not been live-tested in production**.

Always run:

```bash
python3 verify_fallback_provider.py
```

before deploying.

---

# Deployment (Streamlit Cloud)

1. Push the repository to GitHub.
   Ensure that `config.py` is **never committed** (it is already listed in `.gitignore`; verify before pushing).

2. Create a new application on Streamlit Cloud and connect it to this repository.

3. Open **Settings → Secrets** and add the same configuration variables used in `config.py`, including:

   * `GEMINI_API_KEY`
   * `CLASS_PIN`
   * `TEACHER_PASSWORD`
   * and any additional required settings.

4. If `knowledge_base.json` is stored using Git LFS, verify after deployment that Streamlit Cloud downloads the **actual file** (approximately 94 MB or larger) rather than only the Git LFS pointer.
