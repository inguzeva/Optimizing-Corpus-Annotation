from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CHAR_MAP = {
    "ö": "ӧ",
    "Ö": "Ӧ",
    "ÿ": "ӱ",
    "Ÿ": "Ӱ",
    "ä": "ӓ",
    "Ä": "Ӓ",
    "ü": "ӱ",
    "Ü": "Ӱ",
}

RE_HYPHEN_SPACE = re.compile(r"([A-Za-zА-Яа-яЁёӦӧӰӱҮүЈј0-9])-\s+([A-Za-zА-Яа-яЁёӦӧӰӱҮүЈј0-9])")
RE_MULTI_SPACE = re.compile(r"\s+")
RE_SOURCE_ENTRY_HEAD = re.compile(r"^\([А-ЯЁA-Z]{1,4}\)\s+")
RE_SINGLE_LETTER = re.compile(r"^[А-ЯЁA-ZӦӰӒӤӢӦӰЈҮӰӱӧ]{1}$")
RE_QUOTE_START = re.compile(r"^[«\"'“]")


def ensure_dirs():
    (ROOT / "data" / "processed" / "clean").mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def normalize_text(s: str) -> str:
    if not s:
        return ""
    for bad, good in CHAR_MAP.items():
        s = s.replace(bad, good)
    s = s.replace("\u00ad", "")
    s = RE_HYPHEN_SPACE.sub(r"\1\2", s)
    s = RE_MULTI_SPACE.sub(" ", s).strip()
    return s


def looks_like_false_entry(headword_raw: str, raw_text: str) -> bool:
    hw = (headword_raw or "").strip()
    rt = (raw_text or "").strip()

    if not hw:
        return True

    if RE_SINGLE_LETTER.match(hw):
        return True

    if hw.startswith("(") and hw.endswith(")") and len(hw) <= 6:
        return True

    if RE_QUOTE_START.match(hw):
        return True

    if RE_SOURCE_ENTRY_HEAD.match(rt):
        return True

    return False


def clean_entries(entries_rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Set[str]]:
    kept: List[Dict[str, str]] = []
    removed_ids: Set[str] = set()

    for r in entries_rows:
        entry_id = str(r.get("entry_id", "")).strip()
        headword_raw = r.get("headword_raw", "") or ""
        raw_text = r.get("raw_text", "") or ""

        if looks_like_false_entry(headword_raw, raw_text):
            removed_ids.add(entry_id)
            continue

        r2 = dict(r)
        r2["headword_raw"] = normalize_text(headword_raw)
        r2["headword_norm"] = normalize_text(r.get("headword_norm", "") or headword_raw).lower()
        r2["raw_text"] = normalize_text(raw_text)
        r2["labels"] = normalize_text(r.get("labels", "") or "")
        r2["pos_primary"] = normalize_text(r.get("pos_primary", "") or "")
        kept.append(r2)

    return kept, removed_ids


def filter_by_entry_id(rows: List[Dict[str, str]], removed_ids: Set[str], id_field: str = "entry_id") -> List[Dict[str, str]]:
    out = []
    for r in rows:
        eid = str(r.get(id_field, "")).strip()
        if eid in removed_ids or not eid:
            continue
        out.append(r)
    return out


def clean_senses(senses_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in senses_rows:
        r2 = dict(r)
        r2["gloss_ru"] = normalize_text(r.get("gloss_ru", "") or "")
        r2["gloss_ru_alt"] = normalize_text(r.get("gloss_ru_alt", "") or "")
        r2["labels"] = normalize_text(r.get("labels", "") or "")
        r2["refs"] = normalize_text(r.get("refs", "") or "")
        r2["raw_block"] = normalize_text(r.get("raw_block", "") or "")
        out.append(r2)
    return out


def clean_phrases(phrases_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in phrases_rows:
        r2 = dict(r)
        r2["phrase_alt_raw"] = normalize_text(r.get("phrase_alt_raw", "") or "")
        r2["phrase_alt_norm"] = normalize_text(r.get("phrase_alt_norm", "") or r.get("phrase_alt_raw", "") or "").lower()
        r2["phrase_ru"] = normalize_text(r.get("phrase_ru", "") or "")
        r2["labels"] = normalize_text(r.get("labels", "") or "")
        r2["raw_block"] = normalize_text(r.get("raw_block", "") or "")
        out.append(r2)
    return out


def clean_examples(examples_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in examples_rows:
        r2 = dict(r)
        r2["example_alt_raw"] = normalize_text(r.get("example_alt_raw", "") or "")
        r2["example_alt_norm"] = normalize_text(r.get("example_alt_norm", "") or r.get("example_alt_raw", "") or "").lower()
        r2["example_ru"] = normalize_text(r.get("example_ru", "") or "")
        r2["source_note"] = normalize_text(r.get("source_note", "") or "")
        r2["raw_line"] = normalize_text(r.get("raw_line", "") or "")
        out.append(r2)
    return out


def clean_abbr(abbr_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in abbr_rows:
        r2 = dict(r)
        r2["abbr"] = normalize_text(r.get("abbr", "") or "")
        r2["full"] = normalize_text(r.get("full", "") or "")
        r2["type"] = normalize_text(r.get("type", "") or "label")
        if not r2["abbr"]:
            continue
        out.append(r2)
    return out


def main() -> int:
    ensure_dirs()
    cfg = load_config()

    parsed_dir = ROOT / "data" / "processed" / "parsed"
    clean_dir = ROOT / "data" / "processed" / "clean"

    entries_in = parsed_dir / "entries.csv"
    senses_in = parsed_dir / "senses.csv"
    phrases_in = parsed_dir / "phrases.csv"
    examples_in = parsed_dir / "examples.csv"
    abbr_in = parsed_dir / "abbr_labels.csv"

    for p in [entries_in, senses_in, phrases_in, examples_in]:
        if not p.exists():
            print(f"[FAIL] Missing parsed file: {p}")
            return 1

    entries_rows = read_csv(entries_in)
    senses_rows = read_csv(senses_in)
    phrases_rows = read_csv(phrases_in) if phrases_in.exists() else []
    examples_rows = read_csv(examples_in) if examples_in.exists() else []
    abbr_rows = read_csv(abbr_in) if abbr_in.exists() else []

    entries_clean, removed_ids = clean_entries(entries_rows)

    senses_rows = filter_by_entry_id(senses_rows, removed_ids, "entry_id")
    phrases_rows = filter_by_entry_id(phrases_rows, removed_ids, "entry_id")
    examples_rows = filter_by_entry_id(examples_rows, removed_ids, "entry_id")

    senses_clean = clean_senses(senses_rows)
    phrases_clean = clean_phrases(phrases_rows)
    examples_clean = clean_examples(examples_rows)
    abbr_clean = clean_abbr(abbr_rows)

    entries_out = clean_dir / "entries_clean.csv"
    senses_out = clean_dir / "senses_clean.csv"
    phrases_out = clean_dir / "phrases_clean.csv"
    examples_out = clean_dir / "examples_clean.csv"
    abbr_out = clean_dir / "abbr_labels_clean.csv"

    entries_fields = list(entries_clean[0].keys()) if entries_clean else list(entries_rows[0].keys())
    senses_fields = list(senses_clean[0].keys()) if senses_clean else list(senses_rows[0].keys())
    phrases_fields = list(phrases_clean[0].keys()) if phrases_clean else (list(phrases_rows[0].keys()) if phrases_rows else [])
    examples_fields = list(examples_clean[0].keys()) if examples_clean else (list(examples_rows[0].keys()) if examples_rows else [])
    abbr_fields = ["abbr", "full", "type"]

    write_csv(entries_out, entries_clean, entries_fields)
    write_csv(senses_out, senses_clean, senses_fields)
    if phrases_fields:
        write_csv(phrases_out, phrases_clean, phrases_fields)
    else:
        with open(phrases_out, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=["entry_id", "phrase_id", "phrase_alt_raw", "phrase_alt_norm", "phrase_ru", "labels", "raw_block"]).writeheader()
    if examples_fields:
        write_csv(examples_out, examples_clean, examples_fields)
    else:
        with open(examples_out, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=["entry_id", "sense_id", "ex_id", "example_alt_raw", "example_alt_norm", "example_ru", "source_note", "raw_line"]).writeheader()
    write_csv(abbr_out, abbr_clean, abbr_fields)

    print("[OK] Clean CSV saved to data/processed/clean/")
    print(f"[INFO] entries: {len(entries_rows)} -> {len(entries_clean)} (removed {len(removed_ids)})")
    print(f"[INFO] senses: {len(senses_rows)} -> {len(senses_clean)}")
    print(f"[INFO] phrases: {len(phrases_rows)} -> {len(phrases_clean)}")
    print(f"[INFO] examples: {len(examples_rows)} -> {len(examples_clean)}")
    print(f"[INFO] abbr: {len(abbr_rows)} -> {len(abbr_clean)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
