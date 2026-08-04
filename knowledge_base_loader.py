"""
knowledge_base_loader.py
---------------------------
knowledge_base.json (embed_chunks.py se banti hai) ko load karta hai aur
embeddings ko ek numpy matrix mein convert karta hai — taake retrieval
(cosine similarity) fast ho.

Streamlit ke `@st.cache_resource` se wrap kiya gaya hai taake ye poore
app-process mein sirf EK BAAR load ho (94MB+ file har request pe dobara
parse karna bohat slow hota).
"""

import json

import numpy as np
import streamlit as st


@st.cache_resource
def load_knowledge_base(path: str = "knowledge_base.json"):
    with open(path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    if not kb:
        raise ValueError(
            f"'{path}' khali hai ya load nahi hui — pehle `python3 embed_chunks.py` "
            "chala kar knowledge base banayein."
        )
    embeddings_matrix = np.array([item["embedding"] for item in kb])
    courses = sorted(set(c["course"] for c in kb))
    return kb, embeddings_matrix, courses
