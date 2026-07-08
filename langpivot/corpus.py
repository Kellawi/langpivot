"""Build real parallel sentence pairs from Tatoeba exports.

Inputs (in data/): {lang}_sentences.tsv.bz2 (id, lang, text) and
{lang}-eng_links.tsv.bz2 (source_id, target_id). Output: list of
(source_text, english_text) pairs, deduped and length-filtered.
"""

from __future__ import annotations

import bz2
import random
import unicodedata


def _load_sentences(path: str) -> dict[int, str]:
    out: dict[int, str] = {}
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                try:
                    out[int(parts[0])] = parts[2]
                except ValueError:
                    continue
    return out


def build_pairs(
    src_sentences_path: str,
    eng_sentences_path: str,
    links_path: str,
    max_pairs: int = 50000,
    min_len: int = 4,
    max_len: int = 500,
    seed: int = 42,
) -> list[tuple[str, str]]:
    src = _load_sentences(src_sentences_path)
    eng = _load_sentences(eng_sentences_path)
    pairs: list[tuple[str, str]] = []
    seen: set[int] = set()  # one English pairing per source sentence
    with bz2.open(links_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                a, b = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if a in src and b in eng and a not in seen:
                s = unicodedata.normalize("NFKC", src[a]).strip()
                e = eng[b].strip()
                if min_len <= len(s) <= max_len and min_len <= len(e) <= max_len:
                    seen.add(a)
                    pairs.append((s, e))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:max_pairs]
