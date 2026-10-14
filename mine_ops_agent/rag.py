"""Small, dependency-free retrieval over SOP and policy documents.

Documents are split on markdown headings and ranked with BM25. Every chunk
carries a citation (``file#Section``) so answers can point to their source.
Retrieved text is untrusted data: lines that look like instructions to the
model are dropped before they reach the prompt.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "sops"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "with",
         "at", "by", "it", "its", "as", "any", "all", "each", "must", "can", "when", "than", "what", "do", "we"}
_INJECTION = re.compile(r"ignore (all |any )?(previous|prior|above) instructions|you are now|system prompt",
                        re.IGNORECASE)


def _stem(token: str) -> str:
    """Light plural stemming so 'limit' matches 'limits' and 'requisition' matches 'requisitions'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    source: str
    title: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.title}"


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        doc_title, section, buf = path.stem, "Overview", []

        def flush() -> None:
            body = "\n".join(line for line in buf if not _INJECTION.search(line)).strip()
            if body:
                chunks.append(Chunk(path.name, section, f"{doc_title} / {section}: {body}"))

        for line in path.read_text().splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                section, buf = line[3:].strip(), []
            else:
                buf.append(line)
        flush()
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = []
        for chunk, doc, length in zip(self.chunks, self.docs, self.lengths):
            score = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    score += self.idf[t] * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, round(score, 3)))
        scored.sort(key=lambda s: s[1], reverse=True)
        if not scored:
            return []
        # Drop weak matches so citations stay relevant
        return [s for s in scored[:k] if s[1] >= 0.5 * scored[0][1]]


_default_index: BM25Index | None = None


def default_index() -> BM25Index:
    global _default_index
    if _default_index is None:
        _default_index = BM25Index(load_chunks())
    return _default_index
