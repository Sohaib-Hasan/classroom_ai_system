# Classroom AI System

Ek Streamlit-based doubt-clearing assistant jo sirf aapke course notes se
grounded jawab deta hai — kisi bhi topic pe seedha "AI se poochna" nahi,
balke "aapke course ke notes ke andar dhoondh kar jawab dena."

## Kya-kya hai is repo mein

| File | Kya karta hai |
|---|---|
| `app.py` | Students ke liye chat UI |
| `dashboard.py` | Teacher-only analytics dashboard |
| `chunk_notes.py` | `.tex` notes ko chhote "chunks" mein todta hai |
| `embed_chunks.py` | Har chunk ka embedding banata hai (`knowledge_base.json`) |
| `core.py` | Saari pure business-logic (Streamlit se independent — testable) |
| `embedding_backend.py` | Embedding provider abstraction (Gemini ya free local model) |
| `generation_backend.py` | Generation provider abstraction (Gemini + optional fallback) |
| `cache_store.py` | SQLite-backed Q&A cache |
| `db_connection.py` | Pluggable storage connection (local SQLite ya Turso — shared data across separately-deployed apps) |
| `question_log_store.py` | Question-activity log (dashboard analytics ka data source) |
| `auth_guard.py` | PIN/password brute-force lockout logic |
| `logging_setup.py` | Error logging |
| `knowledge_base_loader.py` | `knowledge_base.json` ko load karta hai |
| `list_models.py` | Diagnostic: aapki API key ke available models dikhata hai |
| `verify_fallback_provider.py` | Manual smoke-test AgentRouter/fallback provider ke liye |
| `tests/` | Automated tests (pytest) |

## Setup (pehli baar)

```bash
pip install -r requirements.txt
cp config.py.example config.py
```

`config.py` khol kar apni values bharein (`GEMINI_API_KEY`, `CLASS_PIN`,
`TEACHER_PASSWORD` kam se kam). API key yahan se milti hai:
https://aistudio.google.com/apikey

Notes se knowledge base banane ke liye:
```bash
python3 chunk_notes.py    # .tex notes -> *_chunks.json (agar aapka apna chunking script hai)
python3 embed_chunks.py   # *_chunks.json -> knowledge_base.json
```

Chalana:
```bash
streamlit run app.py         # student chat
streamlit run dashboard.py   # teacher dashboard
```

## Tests chalana

```bash
pip install pytest
pytest tests/ -v
```

70 tests hain — inme woh sab bugs bhi cover hote hain jo pehle review mein
mile the (dekhein `CHANGELOG.md`), taake wo dobara chup-chaap wapas na aa
sakein.

## Teacher aur student app ALAG deploy kar rahe hain? (important)

Agar `app.py` aur `dashboard.py` ko do ALAG Streamlit Cloud apps ke tor
par deploy kar rahe hain (do alag URLs, jaise ye project), to **dashboard
by default khali dikhega** — har Streamlit Cloud app apna isolated
container use karta hai, local SQLite file ek doosre ko nazar nahi aati.

Fix: dono apps ki Secrets mein SAME Turso credentials daalein:
```toml
TURSO_DATABASE_URL = "libsql://your-db-name.turso.io"
TURSO_AUTH_TOKEN = "..."
```
Setup steps `db_connection.py` ke docstring mein hain (5 minute ka kaam,
free). Deploy se pehle confirm karne ke liye:
```bash
pip install -r requirements-turso.txt
python3 verify_turso_connection.py
```

Agar dono apps ko sirf EK hi Streamlit app ke tor par (multipage) chalayein,
to Turso ki zaroorat nahi — local storage kaam kar jayega, lekin restart/
sleep pe wo bhi reset ho sakti hai (Streamlit Cloud free tier ka local
storage permanent nahi hota).

## Zero-budget setup

Is system ka design hi is soch ke saath hua hai ke koi paisa kharch na ho.
Do hisse hain jahan quota lagta hai — **embedding** aur **generation** —
aur dono independently free-tier-friendly banaye ja sakte hain.

### Embedding (retrieval) — sabse zyada consume hone wala hissa

**Important fact**: har sawal — chahe cache se mile ya nahi — pehle ek
embedding call zaroor karta hai (query ko cache se match karne ke liye bhi
uska embedding chahiye hota hai). Isliye embedding hi asal bottleneck hai,
generation nahi (jo caching se bach jata hai).

Do options:
1. **Gemini (default)** — free, lekin apni RPD limit hai jo mahine dar
   mahine badalti hai. `config.py`: `EMBEDDING_PROVIDER = "gemini"`.
2. **Local/free model (recommended agar quota tight ho)** — bilkul free,
   koi rate-limit nahi, koi API key nahi, aapke apne machine par chalta
   hai:
   ```bash
   pip install -r requirements-local-embeddings.txt
   ```
   `config.py`: `EMBEDDING_PROVIDER = "local"`, phir:
   ```bash
   python3 embed_chunks.py --rebuild
   ```
   `--rebuild` **zaroori** hai — Gemini aur local model ke embeddings
   AAPAS MEIN compatible nahi hain, mix karna galat results dega.

### Generation (answer-writing)

Default: Gemini. Agar exam-week jaisi heavy traffic mein Gemini ka
free-tier quota khatam ho jaye, ek fallback provider configure kar sakte
hain (jaise AgentRouter) jo Gemini fail hone par automatically try hoga:

```python
# config.py
GENERATION_FALLBACK_PROVIDER = "agentrouter"
GENERATION_FALLBACK_API_KEY = "sk-..."
GENERATION_FALLBACK_MODEL = "claude-sonnet-4-5-20250929"  # apne console mein confirm karein
```

**Zaroori caveats third-party gateways (AgentRouter, OpenRouter, etc.) ke
baare mein:**
- Ye unverified third-party proxies hain — koi published data-retention
  policy nahi. Isse **primary/only** backbone na banayein, sirf fallback.
- Deploy se pehle `python3 verify_fallback_provider.py` chala kar khud
  confirm karein ke ye kaam kar raha hai — is integration ko live test
  nahi kiya gaya tha (development sandbox mein network restricted thi).
- **Apni API key kabhi bhi chat, code comments, ya commit mein na
  likhein** — sirf `config.py` (gitignored) ya Streamlit Secrets mein.
  Agar koi key kabhi chat/screenshot mein share ho jaye, use turant
  revoke/regenerate kar dein — wo compromised maani jani chahiye.

## Architecture notes

- `core.py` mein koi `import streamlit` nahi hai — jaan-boojh kar, taake
  business logic ko bina Streamlit chalaye test kiya ja sake.
- Cache SQLite mein hai (`cache/qa_cache.db`), flat JSON mein nahi — purana
  design har save par poori file rewrite karta tha.
- `verify_computation()` ab negative, positive, aur near-zero — teenon
  domains se sample karta hai (pehle sirf positive, jis se domain-sensitive
  galtiyan jaise `sqrt(x**2) == x` "verified True" ban jati thi).

## Known limitations (honestly documented, jaan-boojh kar fix nahi kiye)

- **Nested same-type LaTeX boxes** (ek `definitionbox` ke andar doosra
  `definitionbox`): `chunk_notes.py` ka parser inhe cleanly separate nahi
  karta — andar wale box ka apna chunk nahi banta, aur uska raw markup
  bahar wale chunk mein leak ho jata hai. Ye behavior test se confirmed
  hai (`tests/test_chunk_notes.py`). Ek safety-net warning
  (`check_for_leaked_box_markup`) ise loudly flag karti hai jab bhi
  `embed_chunks.py`/`chunk_notes.py` chale, taake ye chup-chaap knowledge
  base mein na jaye. Agar aapke notes mein aisi nesting hai, wahan manually
  restructure karein (andar wale box ko alag type mein badlein).
- **Structural cache-safety signature** (`structural_signature` in
  `core.py`) bracket-nesting/grouping capture karti hai, lekin ye
  guarantee nahi ki koi bhi do structurally-different-lekin-textually-
  identical inputs kabhi confuse nahi honge — bohat kam-probability edge
  cases theoretically ho sakte hain.
- AgentRouter/fallback provider integration live-tested nahi (upar
  dekhein) — deploy se pehle khud confirm karein.

## Deploy karna (Streamlit Cloud)

1. Repo ko GitHub par push karein (`config.py` push NA ho — `.gitignore`
   mein already hai, double-check zaroor karein).
2. Streamlit Cloud pe naya app banayein, is repo ko point karein.
3. Settings → Secrets mein `config.py` jaisi values daalein (isi naam se:
   `GEMINI_API_KEY`, `CLASS_PIN`, `TEACHER_PASSWORD`, waghera).
4. `knowledge_base.json` agar Git LFS se hai to Streamlit Cloud LFS support
   karta hai — confirm kar lein ke deploy ke baad file poori (94MB+, sirf
   LFS pointer nahi) load ho rahi hai.
