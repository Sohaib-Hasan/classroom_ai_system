"""
auth_guard.py
---------------
Pehle PIN/teacher-password check sirf ek `==` comparison tha — koi
attempt-limit nahi, koi lockout nahi. Chhoti class ke liye risk kam hai,
lekin agar quota-drain hi asal fikar hai (jaisa code comments mein khud
likha hai), to ek determined student PIN ko script/loop se brute-force
try kar sakta hai.

Ye module pure Python hai (Streamlit se independent) taake lockout logic
ko directly test kiya ja sake. app.py isko `st.session_state` ke saath
wire karta hai (session state hi attempt-counter store karta hai).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


@dataclass
class AttemptState:
    failed_attempts: int = 0
    locked_until: float = 0.0


def is_locked_out(state: AttemptState, now: float | None = None) -> bool:
    now = now if now is not None else time.time()
    return now < state.locked_until


def seconds_remaining(state: AttemptState, now: float | None = None) -> int:
    now = now if now is not None else time.time()
    return max(0, int(state.locked_until - now))


def record_attempt(state: AttemptState, correct: bool, now: float | None = None) -> AttemptState:
    """Ek login attempt record karta hai. Agar correct hai to counter
    reset ho jata hai. Agar galat hai aur MAX_ATTEMPTS tak pohanch jaye,
    to LOCKOUT_SECONDS ke liye lock laga deta hai."""
    now = now if now is not None else time.time()
    if correct:
        state.failed_attempts = 0
        state.locked_until = 0.0
        return state

    state.failed_attempts += 1
    if state.failed_attempts >= MAX_ATTEMPTS:
        state.locked_until = now + LOCKOUT_SECONDS
        state.failed_attempts = 0  # lockout khatam hone ke baad fresh start
    return state
