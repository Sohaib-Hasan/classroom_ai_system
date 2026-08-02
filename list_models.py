"""
list_models.py
----------------
Agar embed_chunks.py mein "model not found" wala error aaye, to ye script
chalayein. Ye batayega ke aapki API key ke pass kaunse models available hain.

Chalane ka tareeqa:
    python3 list_models.py
"""

from google import genai
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

print("Aapki key ke liye ye models available hain:\n")
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "embedContent" in actions:
        print(f"  ✅ EMBEDDING MODEL: {m.name}")
    else:
        print(f"     {m.name}")
