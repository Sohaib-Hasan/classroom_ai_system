"""
embed_chunks.py
-----------------
Har chapter ke chunk-files (chapter1_chunks.json, calc_chapter1_chunks.json,
...) ko utha kar, har tukde ka embedding banata hai — taake baad mein AI
dhoondh sake ke student ke sawal se kaunsa tukda match karta hai.

Pehli baar chalane se pehle install karna hai (terminal mein):
    pip install -r requirements.txt

Phir seedha chalayein:
    python3 embed_chunks.py

RESUMABLE HAI: agar beech mein rukna paड़e (internet, quota, ya kuch aur),
dobara wahi command chalayein — jo tukde pehle se ho chuke hain unhe dobara
nahi karega, sirf baaki wale karega. Progress har 25 tukdon ke baad save
hoti rehti hai, kabhi bhi rok sakte hain.

BACKEND BADALNA HO (Gemini <-> local free model)? config.py mein
EMBEDDING_PROVIDER badlein aur --rebuild ke saath chalayein:
    python3 embed_chunks.py --rebuild

--rebuild ZAROORI hai backend badalne par, kyunke alag backends ke
embeddings AAPAS MEIN COMPATIBLE NAHI hain (dekhein embedding_backend.py).
Bina --rebuild ke, script sirf "naye" chunks add karega aur purane
(doosre backend wale) embeddings ko as-is chhod dega — jo galat/mixed
knowledge base bana degi.
"""

import argparse
import glob
import hashlib
import json
import os
import time

from config import GEMINI_API_KEY
try:
    from config import EMBEDDING_PROVIDER
except ImportError:
    EMBEDDING_PROVIDER = "gemini"

from core import truncate_for_embedding
from embedding_backend import get_backend
from google import genai

OUTPUT_FILE = "knowledge_base.json"
SAVE_EVERY = 25


def load_all_chunks():
    """Sab *_chunks.json files ko dhoond kar (kisi bhi course ke) ek list mein jama karta hai."""
    all_chunks = []
    for filepath in sorted(glob.glob("*_chunks.cleaned.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    return all_chunks


def chunk_id(chunk):
    """Har tukde ki apni unique pehchan — resume/skip karte waqt yehi check hoti hai.

    FIX (quota-saving): pehle sirf course|chapter|title par based tha —
    matlab agar kisi chunk ka SIRF content badle (jaise LaTeX-cleaning se),
    title wahi rehta hai, to ye "already done" samajh kar skip ho jata
    (purana, dirty content wala embedding hi reh jata — content-fix kabhi
    apply hi nahi hota). Ab ek content-hash bhi shamil hai, is liye:
      - Content badla  -> naya hash -> "naya" chunk samjha jayega -> re-embed hoga
      - Content same   -> wahi hash -> "already done" -> skip (purana embedding safe)
    """
    content_hash = hashlib.sha256(chunk.get("content", "").encode("utf-8")).hexdigest()[:16]
    return f"{chunk.get('course','')}|{chunk['chapter']}|{chunk['title']}|{content_hash}"


def load_existing_knowledge_base():
    if os.path.isfile(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_knowledge_base(knowledge_base):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f)


def get_embedding(backend, text, retries=3):
    text, was_truncated = truncate_for_embedding(text)
    if was_truncated:
        print("  ⚠️  Ye chunk embedding model ki input-limit se bada tha — truncate kar diya gaya.")
    for attempt in range(retries):
        try:
            return backend.embed_document(text)
        except Exception as e:
            print(f"  Koshish {attempt + 1} fail hui: {e}")
            time.sleep(3)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Poori knowledge base dobara embed karo (backend badalne ke baad zaroori hai).",
    )
    args = parser.parse_args()

    client = genai.Client(api_key=GEMINI_API_KEY) if EMBEDDING_PROVIDER == "gemini" else None
    backend = get_backend(EMBEDDING_PROVIDER, client=client)
    print(f"Embedding provider: {backend.name}\n")

    all_chunks = load_all_chunks()
    knowledge_base = [] if args.rebuild else load_existing_knowledge_base()
    already_done = {chunk_id(c) for c in knowledge_base}

    remaining = [c for c in all_chunks if chunk_id(c) not in already_done]

    print(f"Total {len(all_chunks)} tukde hain, {len(already_done)} pehle se ho chuke hain.")
    print(f"Baaki {len(remaining)} tukdon ka embedding banana shuru...\n")

    if not remaining:
        print("Sab kuch pehle se ho chuka hai! Kuch karne ki zarurat nahi.")
        print("(Agar backend badla hai to `python3 embed_chunks.py --rebuild` chalayein.)")
        return

    for i, chunk in enumerate(remaining):
        text_to_embed = f"{chunk['course']} - {chunk['chapter']} - {chunk['section']} - {chunk['title']}\n{chunk['content']}"

        embedding = get_embedding(backend, text_to_embed)
        if embedding is None:
            print(f"  ⚠️  Skip ho gaya (3 koshishon ke baad bhi fail): {chunk['title']}")
            continue

        chunk_with_embedding = dict(chunk)
        chunk_with_embedding["embedding"] = embedding
        knowledge_base.append(chunk_with_embedding)

        if (i + 1) % SAVE_EVERY == 0:
            save_knowledge_base(knowledge_base)
            print(f"  {i + 1}/{len(remaining)} ho gaye... (progress save ho gayi)")

        if backend.name == "gemini":
            time.sleep(1)  # free tier ki rate-limit se bachne ke liye — local model ko iski zaroorat nahi

    save_knowledge_base(knowledge_base)
    print(f"\n✅ Done! Total {len(knowledge_base)} tukdon ke embeddings '{OUTPUT_FILE}' mein save ho gaye.")


if __name__ == "__main__":
    main()
