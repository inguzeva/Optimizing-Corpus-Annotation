from __future__ import annotations

import re
from pathlib import Path


def parse_dopslovar(path: Path) -> list[dict]:
    """
    Формат файла:
      altay_word

      russian_translation

      altay_word

      russian_translation
    """
    if not path.exists():
        return []

    raw = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", raw) if b.strip()]

    pairs: list[dict] = []
    i = 0
    while i < len(blocks):
        headword = blocks[i].strip()
        gloss = blocks[i + 1].strip() if i + 1 < len(blocks) else ""
        i += 2

        if not headword or not gloss:
            continue

        pairs.append({"headword": headword, "gloss_ru": gloss})

    return pairs
