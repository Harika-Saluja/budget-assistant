"""checks.py - the 'compare against criteria' stage of a model call (Lectures 8, 9 and 11).

Lecture 9 says the often-missed last stage is checking the answer. These checks are deterministic code,
not another model, so they can be unit-tested. They catch *some* problems; they do not prove an answer is right.
"""
from __future__ import annotations

import re

AMOUNT_RE = re.compile(r"£\s?(\d[\d,]*(?:\.\d+)?)")
GUESS_WORDS = ("guess", "may be out of date", "might be out of date", "not certain", "estimate", "inferred", "assum")
CLAIM_RE = re.compile(r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+(?:just\s+)?"
                      r"(?:changed|updated|moved|transferred|cancelled|canceled|paid|set)\b", re.I)


def extract_amounts(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in AMOUNT_RE.findall(text)]


def match_figures(text: str, allowed, tol: float = 1.0):
    """Split the £ figures in `text` into those found in `allowed` (within `tol` pounds) and those not."""
    found, missing = [], []
    for amount in extract_amounts(text):
        (found if any(abs(amount - x) <= tol for x in allowed) else missing).append(amount)
    return found, missing


def reply_checks(text: str, allowed, inferred_used: bool, max_words: int = 120) -> list[dict]:
    """Return a list of {label, ok, detail}. `ok=None` means not applicable."""
    found, missing = match_figures(text, allowed)
    words = len(text.split())
    guess_flagged = any(w in text.lower() for w in GUESS_WORDS)
    claims_action = bool(CLAIM_RE.search(text))
    return [
        {"label": "Every £ figure matches your data",
         "ok": not missing,
         "detail": (f"{len(found)} matched" if not missing else
                    "not found in your data: " + ", ".join(f"£{m:g}" for m in missing))
                   if (found or missing) else "no figures quoted"},
        {"label": f"Under {max_words} words", "ok": words <= max_words, "detail": f"{words} words"},
        {"label": "A guess is described as a guess",
         "ok": (guess_flagged if inferred_used else None),
         "detail": ("an inferred note was used" if inferred_used else "no inferred note used")},
        {"label": "Doesn't claim to have changed anything", "ok": not claims_action,
         "detail": "changes only happen after you confirm"},
    ]
