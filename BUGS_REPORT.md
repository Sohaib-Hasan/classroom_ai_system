# Classroom AI System — Bug Analysis (Aug 2026)

Repo analyzed: `Sohaib-Hasan/classroom_ai_system`
Apps: `plotlab-classroom.streamlit.app` (student), `plotlab-teacher.streamlit.app` (teacher)

Every finding below was reproduced directly (cloned the repo, ran the real
test suite, scanned the real production knowledge-base files) — not guessed.

---

## Bug 1 — Turso crash: `libsql-client` was never actually installed

**Root cause:** `libsql-client` (the Turso database driver) lives in
`requirements-turso.txt`, a *separate* file from `requirements.txt`.
Streamlit Community Cloud only installs from **one** dependency file —
whichever it finds first (`requirements.txt`, in this repo's case). It
never looks at `requirements-turso.txt`. So the moment
`TURSO_DATABASE_URL`/`TURSO_AUTH_TOKEN` were added to Secrets,
`db_connection.py`'s `TursoConnection.__init__` hit:

```python
import libsql_client   # ModuleNotFoundError — package was never installed
```

...crashing the whole app on every load. That's the error you saw, and
that's why removing the two secrets "fixed" it (fell back to local
SQLite, which doesn't sync between the two separately-deployed apps —
hence the teacher dashboard staying empty again).

**Proof:** installing only `requirements.txt` and running the repo's own
test suite reproduces the exact failure:
```
7 failed, 84 passed — ModuleNotFoundError: No module named 'libsql_client'
```
Installing `requirements-turso.txt` too → **91 passed, 0 failed.**

**Fix (already applied, included in this delivery):** moved
`libsql-client>=0.3.1` into `requirements.txt` itself. Nothing else
about the Turso logic was wrong — `db_connection.py`'s design (fall back
to local SQLite when Turso isn't configured) is correct.

**What you need to do:** replace `requirements.txt` in your repo with
the one attached here, push, then re-add the same
`TURSO_DATABASE_URL`/`TURSO_AUTH_TOKEN` to **both** apps' Secrets (same
values in both, per the existing docstring in `db_connection.py`). That
also resolves the teacher-dashboard-is-empty issue, since both apps will
then share the same database.

---

## Bug 2 — Raw LaTeX leaking into student answers

**Root cause:** `chunk_notes.py` (the script that turns your `.tex` notes
into `*_chunks.json`) extracts each definition/example/theorem box as
**raw LaTeX source, completely unstripped**. It only splits the file at
box/section boundaries — it never removes formatting commands like
`\textcolor`, `\textbf`, `\begin{tabular}`, `\begin{tikzpicture}`,
`\vspace`, FontAwesome icons (`\faInfoCircle` etc.), `\rowcolor`,
`\cellcolor`.

That raw content is then inserted verbatim into the AI's prompt
(`core.build_generation_prompt`) as the "notes context" for **every
question**. The system prompt only tells the model not to use
math-notation backslash commands (`\times`, `\frac`, etc.) in its *own*
final answer — it says nothing about the decorative commands sitting in
the context it's reading. When a retrieved chunk is dense with tables/
colors/diagrams, the model sometimes echoes some of that raw markup back
— and since `st.markdown()` doesn't understand `\textcolor{...}{...}` or
`\begin{tabular}`, it shows up as literal backslash-code to the student.

**Proof — this is not a rare edge case, it's close to universal:**
I scanned all 21 `*_chunks.json` files currently used by both deployed
apps for unstripped formatting commands:

| File | % of chunks affected |
|---|---|
| `calc_chapter1`–`6` | 99–100% |
| `dm_chapter1`–`5` | 67–100% |
| `nt_chapter1`–`6` | 77–91% |
| `chapter1`–`5` (Linear Algebra) | 58–67% |

Example, straight from `nt_chapter2_chunks.json` ("Prime Testing" table):
```
\textcolor{primaryblue}{\faCheckCircle\ \textbf{Summary of Definitions}}
\begin{center}\begin{tabular}{|p{3.5cm}|p{8cm}|}\hline\rowcolor{primaryblue!20}...
```
That entire block is what the AI reads as "the notes" — and it's what
can leak into an answer.

**Bonus find (same root cause, different symptom):** the auto-title
generator for "untitled" boxes (`tcolorbox`/`keypoint`) only strips
`\command` names, not their `{arguments}`. For boxes like the one above,
the **source title itself is broken** — e.g. one chunk's title in
production right now is literally:
```
{primaryblue}{\ {Summary of Definitions}} {center} {tabular}{|p{3.5cm}|p{8cm}|}
```
This shows to students directly in the "Sources used from notes"
expander in `app.py` — guaranteed garbled text, independent of what the
AI does.

**Fixes included in this delivery:**
1. `chunk_notes.py` — patched the auto-title generator to properly
   unwrap `\command{...}` (one level of nesting) instead of just
   deleting the command name. Verified against the existing test suite
   (9/9 still pass). This fixes new chunking going forward, once you
   regenerate from your `.tex` source.
2. `clean_chunks.py` — a new script that cleans the **already-generated**
   `*_chunks.json` (colors/bold → plain text, tables → Markdown tables,
   tikz diagrams → removed, lists → Markdown lists, real math left
   completely untouched). Tested against your actual 21 files:
   **before: 57–100% of chunks per file affected → after: ~3% (72 of
   ~2,260 chunks)**, mostly `\textcolor`/`\cellcolor` used *inside* real
   `$...$` math, plus a couple of `tikzpicture` blocks nested in
   `\begin{center}` that need a manual look. Run it, then grep the
   `*.cleaned.json` output for `textcolor|textbf|tikzpicture` to get the
   exact remaining list before trusting it in production — I'm handing
   you a strong first pass, not a silently-perfect one.
3. After cleaning, you'll need to re-run `embed_chunks.py` on the
   cleaned chunks to regenerate `knowledge_base.json` (needs your Gemini
   API key/local embedding setup — I don't have that, so I couldn't do
   this last step for you).

**Longer-term:** the real fix is doing this stripping *inside*
`chunk_notes.py` itself, from the original `.tex`, rather than patching
already-extracted JSON — cleaner, and it'll keep working as you add new
chapters. Worth asking your developer to fold `clean_chunks.py`'s logic
into `chunk_notes.py` directly.

---

## Worth double-checking (not confirmed, but flagged by your own README)

`knowledge_base.json` is stored via Git LFS (94MB actual file). Streamlit
Community Cloud is documented to support LFS, but there are a fair
number of real reports of LFS files silently failing to resolve on
deploy (showing up as the tiny pointer file instead of real content).
Your `README.md` already flags this as something to verify — worth
actually checking your Streamlit Cloud deploy logs to confirm the full
94MB is loading, not just the LFS pointer text.

---

## What's attached
- `requirements.txt` — fixed (Turso dependency merged in)
- `requirements-turso.txt` — updated comment, kept for local-dev docs
- `chunk_notes.py` — patched auto-title bug
- `clean_chunks.py` — chunk-content cleaner (run against your `*_chunks.json`)
