
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]


# --- эвристики "что НЕ является началом статьи" ---
RE_SOURCE_IN_PARENS = re.compile(r"^\([А-ЯЁA-Z]{1,4}\)\s*$")  # (АБ)
RE_SOURCE_AT_LINE_START = re.compile(r"^\([А-ЯЁA-Z]{1,4}\)\s+")  # (АБ) текст...
RE_SINGLE_LETTER = re.compile(r"^[А-ЯЁA-ZӦӰӒӤӢӦӰӦӰЈҮӰӱӧ]{1}$")
RE_QUOTE_START = re.compile(r"^[«\"'“]")  # кавычки в начале строки


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs():
    (ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "interim" / "parse_diagnostics").mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def looks_like_entry_start(line: str, headword_regex: re.Pattern) -> bool:
    """
    Эвристика: похоже ли, что строка является заголовком новой словарной статьи.
    """
    if not line:
        return False

    if RE_QUOTE_START.match(line):
        return False

    # (АБ) и подобные источники не считаем заголовком статьи
    if RE_SOURCE_IN_PARENS.match(line) or RE_SOURCE_AT_LINE_START.match(line):
        return False

    # одиночные буквы (заголовки разделов)
    if RE_SINGLE_LETTER.match(line):
        return False

    # базовый матч "похоже на заголовок"
    if not headword_regex.match(line):
        return False

    # доп. страховка: слишком короткая строка типа "А" уже отфильтрована выше,
    # но ещё бывают строки длиной 2-3 символа — тоже сомнительно
    if len(line) < 2:
        return False

    return True


def guess_headword_from_headline(head_line: str) -> str:
    """
    Грубое извлечение леммы из строки заголовка:
    берём первый "токен" до пробела.
    """
    if not head_line:
        return ""
    head_line = head_line.strip()

    # убрать ведущие тире/точки, если вдруг попали
    head_line = re.sub(r"^[\-\–\—•·]+", "", head_line).strip()

    token = head_line.split(" ")[0]

    # иногда в лемме есть дефис, это нормально
    return token.strip()


def main() -> int:
    ensure_dirs()
    cfg = load_config()

    in_path = ROOT / "data" / "interim" / "lines_reflowed.jsonl"
    if not in_path.exists():
        print(f"[FAIL] Missing input: {in_path}")
        print("Run first: python scripts/20_reflow_columns.py")
        return 1

    out_path = ROOT / "data" / "interim" / "entries_raw.jsonl"
    diag_path = ROOT / "data" / "interim" / "parse_diagnostics" / "segment_stats.json"

    seg_cfg = cfg.get("segmentation", {}) or {}
    # дефолтное regex: заголовок начинается с заглавной буквы и имеет короткий "шапочный" кусок
    headword_re_str = seg_cfg.get("headword_regex") or r"^[A-ZА-ЯЁӦӰӒӤӢӦӰЈҮӰӱӧ\-][A-ZА-ЯЁӦӰӒӤӢӦӰЈҮӰӱӧ0-9\- ]{0,60}"
    headword_regex = re.compile(headword_re_str)

    min_lines_per_entry = int(seg_cfg.get("min_lines_per_entry", 1) or 1)
    skip_empty = bool(seg_cfg.get("skip_empty_lines", True))

    # статистика
    stats = {
        "entries_total": 0,
        "pages_total": 0,
        "start_lines_detected": 0,
        "skipped_empty_lines": 0,
        "skipped_source_lines": 0,
        "skipped_single_letter_lines": 0,
        "first_entry_page": None,
        "last_entry_page": None,
    }

    # текущее состояние
    current = None  # dict
    entry_id = 0

    records = list(read_jsonl(in_path))
    stats["pages_total"] = len(records)

    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Segment entries", unit="page")

    with open(out_path, "w", encoding="utf-8") as out:
        for page_rec in iterator:
            pno = int(page_rec["page_index"])
            lines = page_rec.get("lines", []) or []

            for raw_line in lines:
                line = clean_line(raw_line)

                if skip_empty and not line:
                    stats["skipped_empty_lines"] += 1
                    continue

                # отдельная статистика по скипам
                if RE_SOURCE_IN_PARENS.match(line) or RE_SOURCE_AT_LINE_START.match(line):
                    stats["skipped_source_lines"] += 1
                if RE_SINGLE_LETTER.match(line):
                    stats["skipped_single_letter_lines"] += 1

                is_start = looks_like_entry_start(line, headword_regex)
                if is_start:
                    stats["start_lines_detected"] += 1

                    # закрываем предыдущую
                    if current is not None and len(current["lines"]) >= min_lines_per_entry:
                        current["page_end"] = pno if pno >= current["page_start"] else current["page_start"]
                        current["raw_text"] = "\n".join(current["lines"])
                        out.write(json.dumps(current, ensure_ascii=False) + "\n")
                        stats["entries_total"] += 1
                        stats["first_entry_page"] = stats["first_entry_page"] if stats["first_entry_page"] is not None else current["page_start"]
                        stats["last_entry_page"] = current["page_end"]

                    # начинаем новую
                    entry_id += 1
                    current = {
                        "entry_id": entry_id,
                        "head_line": line,
                        "headword_guess": guess_headword_from_headline(line),
                        "page_start": pno,
                        "page_end": pno,
                        "lines": [line],
                        "raw_text": "",
                    }
                else:
                    # продолжаем текущую, если она уже началась
                    if current is not None:
                        current["lines"].append(line)
                    else:
                        # до первой найденной статьи — игнорируем (обычно предисловие/оглавление)
                        continue

        # дописываем последнюю
        if current is not None and len(current["lines"]) >= min_lines_per_entry:
            current["raw_text"] = "\n".join(current["lines"])
            out.write(json.dumps(current, ensure_ascii=False) + "\n")
            stats["entries_total"] += 1
            stats["first_entry_page"] = stats["first_entry_page"] if stats["first_entry_page"] is not None else current["page_start"]
            stats["last_entry_page"] = current["page_end"]

    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved entries: {out_path}")
    print(f"[OK] Segment stats: {diag_path}")
    print(f"[INFO] entries_total={stats['entries_total']} start_lines_detected={stats['start_lines_detected']} pages={stats['pages_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
