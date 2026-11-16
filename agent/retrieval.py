"""Small, dependency-free TF-IDF retrieval index.

Used for (a) the policy-document knowledge source and (b) long-term memory recall.
This is plain information retrieval; it contains no business rules.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "then", "there", "this",
    "to", "was", "were", "will", "with", "not", "no", "do", "does", "should", "must", "may", "can",
}


def _stem(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    for suf in ("ations", "ation", "ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            tok = tok[: -len(suf)]
            break
    if len(tok) > 3 and tok[-1] == tok[-2] and tok[-1] not in "aeious":  # cancell -> cancel
        tok = tok[:-1]
    if len(tok) > 4 and tok.endswith("e"):  # reserve -> reserv
        tok = tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    out = []
    for tok in re.findall(r"[a-z0-9][a-z0-9_\-']*", text.lower()):
        tok = tok.strip("-'_")
        if len(tok) < 2 or tok in _STOP:
            continue
        out.append(_stem(tok))
    return out


class TfidfIndex:
    def __init__(self, docs: list[tuple[str, dict[str, Any]]]):
        self.docs = docs
        tokens = [tokenize(text) for text, _ in docs]
        df: Counter[str] = Counter()
        for toks in tokens:
            df.update(set(toks))
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vecs = [self._vec(t) for t in tokens]

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[float, str, dict[str, Any]]]:
        q = self._vec(tokenize(query))
        scored = []
        for i, vec in enumerate(self.vecs):
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [(round(s, 4), self.docs[i][0], self.docs[i][1]) for s, i in scored[:k]]
