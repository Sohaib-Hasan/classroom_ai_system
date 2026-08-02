"""
embed_chunks.py
-----------------
Har chapter ke chunk-files (chapter1_chunks.json, calc_chapter1_chunks.json,
...) ko utha kar, har tukde ka embedding banata hai — taake baad mein AI
dhoondh sake ke student ke sawal se kaunsa tukda match karta hai.

Pehli baar chalane se pehle install karna hai (terminal mein):
    pip install google-genai

Phir seedha chalayein:
    python3 embed_chunks.py

RESUMABLE HAI: agar beech mein rukna paड़e (internet, quota, ya kuch aur),
dobara wahi command chalayein — jo tukde pehle se ho chuke hain unhe dobara
nahi karega, sirf baaki wale karega. Progress har 25 tukdon ke baad save
hoti rehti hai, kabhi bhi rok sakte hain.
"""

import json
import glob
import time
import os
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_FILE = "knowledge_base.json"
SAVE_EVERY = 25


def load_all_chunks():
    """Sab *_chunks.json files ko dhoond kar (kisi bhi course ke) ek list mein jama karta hai."""
    all_chunks = []
    for filepath in sorted(glob.glob("*_chunks.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    return all_chunks


def chunk_id(chunk):
    """Har tukde ki apni unique pehchan — resume karte waqt yehi check hoti hai."""
    return f"{chunk.get('course','')}|{chunk['chapter']}|{chunk['title']}"


def load_existing_knowledge_base():
    if os.path.isfile(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_knowledge_base(knowledge_base):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f)


def get_embedding(text, retries=3):
    for attempt in range(retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"  Koshish {attempt + 1} fail hui: {e}")
            time.sleep(3)
    return None


def main():
    all_chunks = load_all_chunks()
    knowledge_base = load_existing_knowledge_base()
    already_done = {chunk_id(c) for c in knowledge_base}

    remaining = [c for c in all_chunks if chunk_id(c) not in already_done]

    print(f"Total {len(all_chunks)} tukde hain, {len(already_done)} pehle se ho chuke hain.")
    print(f"Baaki {len(remaining)} tukdon ka embedding banana shuru...\n")

    if not remaining:
        print("Sab kuch pehle se ho chuka hai! Kuch karne ki zarurat nahi.")
        return

    for i, chunk in enumerate(remaining):
        text_to_embed = f"{chunk['course']} - {chunk['chapter']} - {chunk['section']} - {chunk['title']}\n{chunk['content']}"

        embedding = get_embedding(text_to_embed)
        if embedding is None:
            print(f"  ⚠️  Skip ho gaya (3 koshishon ke baad bhi fail): {chunk['title']}")
            continue

        chunk_with_embedding = dict(chunk)
        chunk_with_embedding["embedding"] = embedding
        knowledge_base.append(chunk_with_embedding)

        if (i + 1) % SAVE_EVERY == 0:
            save_knowledge_base(knowledge_base)
            print(f"  {i + 1}/{len(remaining)} ho gaye... (progress save ho gayi)")

        time.sleep(1)  # free tier ki rate-limit se bachne ke liye

    save_knowledge_base(knowledge_base)
    print(f"\n✅ Done! Total {len(knowledge_base)} tukdon ke embeddings '{OUTPUT_FILE}' mein save ho gaye.")


if __name__ == "__main__":
    main()
