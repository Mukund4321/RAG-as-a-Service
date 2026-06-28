import re
from typing import List


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Sentence-aware sliding window chunker.
    Splits on sentence boundaries, then fills windows of ~chunk_size tokens
    (approximated by word count * 1.3) with the given overlap.
    """
    sentences = _split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        words = len(sentence.split())
        if current_len + words > chunk_size and current:
            chunks.append(" ".join(current))
            # retain overlap worth of sentences from the end
            overlap_words = 0
            overlap_buf: List[str] = []
            for s in reversed(current):
                w = len(s.split())
                if overlap_words + w <= overlap:
                    overlap_buf.insert(0, s)
                    overlap_words += w
                else:
                    break
            current = overlap_buf
            current_len = overlap_words

        current.append(sentence)
        current_len += words

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]
