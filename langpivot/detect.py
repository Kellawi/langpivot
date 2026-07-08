"""Cheap script-based language detection.

Good enough to answer the one question the router asks: is this text
already in the pivot language (English/Latin), or is it Japanese / Arabic /
other? No external dependency, O(n), works on mixed text by proportion.
"""

from __future__ import annotations


def _classify_char(ch: str) -> str:
    cp = ord(ch)
    if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
        return "ja"  # hiragana / katakana are uniquely Japanese
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "cjk"  # han ideographs (Japanese or Chinese)
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
        return "ar"
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
        return "latin"
    return "other"


def detect_language(text: str) -> str:
    """Return 'ja', 'ar', 'en', or 'other' by dominant script.

    Han characters with no kana in the text are reported as 'other'
    (could be Chinese); with any kana present they count as Japanese.
    """
    counts = {"ja": 0, "cjk": 0, "ar": 0, "latin": 0, "other": 0}
    letters = 0
    for ch in text:
        cls = _classify_char(ch)
        if cls != "other" or ch.isalpha():
            letters += 1
        counts[cls] += 1
    if letters == 0:
        return "other"
    if counts["ja"] > 0:  # any kana -> Japanese (kanji join the count)
        ja = counts["ja"] + counts["cjk"]
        if ja >= 0.15 * letters:
            return "ja"
    if counts["ar"] >= 0.3 * letters:
        return "ar"
    if counts["latin"] >= 0.5 * letters:
        return "en"
    if counts["cjk"] >= 0.3 * letters:
        return "other"  # han without kana: likely Chinese — don't claim ja
    return "other"
