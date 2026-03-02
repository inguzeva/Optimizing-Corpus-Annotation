#!/usr/bin/env python3
"""
Parse Altay-Russian dictionary PDF into structured CSVs.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pdfplumber

# Letters specific to Altay that are unlikely in Russian glosses
ALTAY_SPECIAL = set("ӧӱҥјӦӰҤЈ")

ROMAN_SET = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}

POS_PATTERNS = [
    (r"\bмежд\.", "межд."),
    (r"\bсоюз\b", "союз"),
    (r"\bчастица\b", "частица"),
    (r"\bпослелог\b", "послелог"),
    (r"\bнареч\.", "нареч."),
    (r"\bсущ\.", "сущ."),
    (r"\bгл\.", "гл."),
    (r"\bприл\.", "прил."),
    (r"\bмест\.", "мест."),
    (r"\bчисл\.", "числ."),
    (r"\bприч\.", "прич."),
    (r"\bдеепр\.", "деепр."),
    (r"\bвводн\.\s*сл\.", "вводн. сл."),
    (r"\bсравн\.\s*ст\.", "сравн. ст."),
]

SENSE_RE = re.compile(r"(?<!\d)(\d{1,2})([).])\s")


def normalize_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    # Normalize quotes and collapse whitespace
    text = text.replace("«", "").replace("»", "").replace("“", "").replace("”", "")
    text = text.replace("„", "").replace("‟", "")
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_roman(token: str) -> bool:
    return token in ROMAN_SET


def clean_token(token: str) -> str:
    return token.strip(".,;:()[]{}")


def is_upper_token(token: str) -> bool:
    token = clean_token(token)
    if not token:
        return False
    # Allow hyphens inside the token
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return False
    return token == token.upper()


def compute_column_threshold(words: List[Dict], width: float) -> float:
    if not words:
        return width / 2
    xs = [(w["x0"] + w["x1"]) / 2 for w in words]
    if len(xs) < 20:
        return width / 2
    c1, c2 = width * 0.33, width * 0.66
    for _ in range(12):
        g1 = [x for x in xs if abs(x - c1) <= abs(x - c2)]
        g2 = [x for x in xs if abs(x - c1) > abs(x - c2)]
        if g1:
            c1 = sum(g1) / len(g1)
        if g2:
            c2 = sum(g2) / len(g2)
    return (c1 + c2) / 2


def group_words_to_lines(words: List[Dict], y_tol: float = 2.0) -> List[List[Dict]]:
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: List[List[Dict]] = []
    current: List[Dict] = []
    current_top: Optional[float] = None
    for w in words_sorted:
        if current_top is None or abs(w["top"] - current_top) > y_tol:
            if current:
                lines.append(current)
            current = [w]
            current_top = w["top"]
        else:
            current.append(w)
    if current:
        lines.append(current)
    return lines


def extract_page_lines(page, page_idx: int) -> List[Dict]:
    words = page.extract_words()
    if not words:
        return []
    threshold = compute_column_threshold(words, page.width)
    left_words = [w for w in words if (w["x0"] + w["x1"]) / 2 < threshold]
    right_words = [w for w in words if (w["x0"] + w["x1"]) / 2 >= threshold]
    left_margin = min((w["x0"] for w in left_words), default=0)
    right_margin = min((w["x0"] for w in right_words), default=0)

    lines = group_words_to_lines(words)
    left_lines: List[Dict] = []
    right_lines: List[Dict] = []
    for line in lines:
        left = [w for w in line if (w["x0"] + w["x1"]) / 2 < threshold]
        right = [w for w in line if (w["x0"] + w["x1"]) / 2 >= threshold]
        if left:
            text = " ".join(w["text"] for w in sorted(left, key=lambda w: w["x0"]))
            left_lines.append(
                {
                    "page": page_idx,
                    "col": 0,
                    "text": text,
                    "x0": min(w["x0"] for w in left),
                    "top": min(w["top"] for w in left),
                    "margin": left_margin,
                }
            )
        if right:
            text = " ".join(w["text"] for w in sorted(right, key=lambda w: w["x0"]))
            right_lines.append(
                {
                    "page": page_idx,
                    "col": 1,
                    "text": text,
                    "x0": min(w["x0"] for w in right),
                    "top": min(w["top"] for w in right),
                    "margin": right_margin,
                }
            )

    left_lines.sort(key=lambda d: d["top"])
    right_lines.sort(key=lambda d: d["top"])
    return left_lines + right_lines


def is_page_number_line(text: str) -> bool:
    return bool(re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", text.strip()))


POS_TOKENS_SIMPLE = {
    "межд.",
    "союз",
    "частица",
    "послелог",
    "нареч.",
    "сущ.",
    "гл.",
    "прил.",
    "мест.",
    "числ.",
    "прич.",
    "деепр.",
    "вводн.",
    "сравн.",
}


def looks_like_pos(token: str) -> bool:
    token = token.strip().lower()
    if token in POS_TOKENS_SIMPLE:
        return True
    # handle two-token POS like \"вводн. сл.\"
    return token in {"сл."}


def parse_headword_line(text: str) -> Optional[Tuple[str, Optional[str], str]]:
    tokens = text.strip().split()
    if not tokens:
        return None
    head_tokens: List[str] = []
    idx = 0
    while idx < len(tokens):
        tok = clean_token(tokens[idx])
        # Stop before roman numerals so they can be captured as homonyms
        if head_tokens and is_roman(tok):
            break
        if is_upper_token(tok):
            head_tokens.append(tok)
            idx += 1
        else:
            break
    if not head_tokens:
        return None

    # Heuristic: avoid sentence starts like "А, ..."
    if len(head_tokens) == 1 and len(head_tokens[0]) == 1 and idx < len(tokens):
        next_tok = clean_token(tokens[idx])
        if not is_roman(next_tok) and not re.match(r"^\d+[).]?$", next_tok) and not looks_like_pos(tokens[idx]):
            return None

    homonym = None
    if idx < len(tokens):
        maybe_roman = clean_token(tokens[idx])
        if is_roman(maybe_roman):
            homonym = maybe_roman
            idx += 1

    remainder = " ".join(tokens[idx:]).strip()
    headword_raw = " ".join(head_tokens)
    return headword_raw, homonym, remainder


def extract_pos(text: str) -> str:
    for pat, label in POS_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return label
    return ""


def build_label_pattern(labels: List[str]) -> Optional[re.Pattern]:
    if not labels:
        return None
    escaped = [re.escape(lab) for lab in sorted(labels, key=len, reverse=True)]
    pattern = r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE)


def extract_labels(text: str, label_pattern: Optional[re.Pattern]) -> str:
    if not text or not label_pattern:
        return ""
    found = label_pattern.findall(text)
    # Normalize: keep original label casing as in pattern list
    unique = []
    seen = set()
    for lab in found:
        lab_norm = lab.strip()
        if lab_norm and lab_norm not in seen:
            seen.add(lab_norm)
            unique.append(lab_norm)
    return ";".join(unique)


def extract_refs(text: str) -> str:
    refs = []
    for m in re.finditer(r"\b(см\.|ср\.)\s*([^;.,]+)", text, flags=re.IGNORECASE):
        val = (m.group(1) + " " + m.group(2)).strip()
        refs.append(val)
    return ";".join(refs)


def is_source_note(src: str, source_abbr: set) -> bool:
    src = src.strip()
    if not src:
        return False
    if src in source_abbr:
        return True
    if src.startswith("Из "):
        return True
    if re.search(r"[А-ЯA-Z]\.", src):
        return True
    if src.isupper() and len(src) <= 6:
        return True
    return False


def extract_examples(block: str, source_abbr: set) -> List[Dict]:
    examples = []
    segments = [seg.strip() for seg in block.split(";") if seg.strip()]
    for seg in segments:
        m = re.search(r"\(([^)]+)\)", seg)
        if not m:
            continue
        src = m.group(1).strip()
        if not is_source_note(src, source_abbr):
            continue
        alt = seg[: m.start()].strip(" ,—–")
        ru = seg[m.end() :].strip(" ,—–")
        if not alt and not ru:
            continue
        examples.append(
            {
                "example_alt_raw": alt,
                "example_alt_norm": normalize_text(alt),
                "example_ru": ru,
                "source_note": src,
                "raw_line": seg,
            }
        )
    return examples


def segment_gloss(block: str, source_abbr: set) -> str:
    segments = [seg.strip() for seg in block.split(";") if seg.strip()]
    gloss_parts = []
    for seg in segments:
        m = re.search(r"\(([^)]+)\)", seg)
        if m and is_source_note(m.group(1), source_abbr):
            continue
        gloss_parts.append(seg)
    return "; ".join(gloss_parts).strip()


def is_mostly_russian(seg: str) -> bool:
    letters = [ch for ch in seg if ch.isalpha()]
    if not letters:
        return False
    alt = sum(1 for ch in letters if ch in ALTAY_SPECIAL)
    return (alt / len(letters)) < 0.05


def parse_senses(
    entry_id: int,
    body_text: str,
    label_pattern: Optional[re.Pattern],
    source_abbr: set,
) -> Tuple[List[Dict], List[Dict]]:
    text = re.sub(r"\s+", " ", body_text).strip()
    if not text:
        return [], []

    matches = list(SENSE_RE.finditer(text))
    blocks: List[Tuple[str, str]] = []
    if not matches:
        blocks.append(("none", text))
    else:
        prefix = text[: matches[0].start()].strip()
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            if i == 0 and prefix:
                block = (prefix + " " + block).strip()
            group = "paren" if m.group(2) == ")" else "dot"
            blocks.append((group, block))

    senses = []
    examples = []
    sense_id = 0
    for group, block in blocks:
        if not block:
            continue
        sense_id += 1
        # Remove phrases from gloss candidate
        block_for_gloss = block.split("♦")[0].strip()
        labels = extract_labels(block_for_gloss, label_pattern)
        pos = extract_pos(block_for_gloss)
        refs = extract_refs(block_for_gloss)
        gloss_full = segment_gloss(block_for_gloss, source_abbr)
        gloss_ru = ""
        if gloss_full:
            ru_parts = [seg for seg in gloss_full.split("; ") if is_mostly_russian(seg)]
            gloss_ru = "; ".join(ru_parts).strip() if ru_parts else gloss_full

        exs = extract_examples(block, source_abbr)
        for ex in exs:
            ex["entry_id"] = entry_id
            ex["sense_id"] = sense_id
            examples.append(ex)

        # Heuristic confidence
        conf = 0.4
        if group in {"paren", "dot"}:
            conf += 0.2
        if gloss_ru:
            conf += 0.2
        if labels or pos:
            conf += 0.1
        if len(block) > 80:
            conf += 0.1
        if len(block) < 10:
            conf -= 0.2
        conf = max(0.0, min(1.0, conf))

        senses.append(
            {
                "entry_id": entry_id,
                "sense_id": sense_id,
                "sense_group": group,
                "pos": pos,
                "gloss_ru": gloss_ru,
                "labels": labels,
                "refs": refs,
                "raw_block": block.strip(),
                "confidence_parse": f"{conf:.2f}",
            }
        )

    return senses, examples


def split_phrase_alt_ru(block: str) -> Tuple[str, str]:
    for sep in [";", "—", "–"]:
        if sep in block:
            alt, ru = block.split(sep, 1)
            return alt.strip(), ru.strip()
    return block.strip(), ""


def extract_phrases(entry_text: str) -> List[Dict]:
    parts = entry_text.split("♦")
    phrases = []
    for idx, part in enumerate(parts[1:], start=1):
        block = part.strip()
        if not block:
            continue
        alt, ru = split_phrase_alt_ru(block)
        phrases.append(
            {
                "phrase_id": idx,
                "phrase_alt_raw": alt,
                "phrase_alt_norm": normalize_text(alt),
                "phrase_ru": ru,
                "raw_block": block,
            }
        )
    return phrases


def parse_abbreviations(pdf) -> Tuple[List[Tuple[str, str, str]], set, set]:
    abbr_rows: List[Tuple[str, str, str]] = []
    label_set = set()
    source_set = set()
    mode = None  # None / labels / sources

    # Abbreviations are confined to the preface pages; avoid dictionary body pages.
    for page_idx in range(min(13, len(pdf.pages))):
        lines = extract_page_lines(pdf.pages[page_idx], page_idx)
        # Use top-to-bottom order to avoid column-order artifacts for headers
        lines_sorted = sorted(lines, key=lambda d: (d["top"], d["x0"]))
        for line in lines_sorted:
            text = line["text"].strip()
            if not text:
                continue
            if "УСЛОВНЫЕ СОКРАЩЕНИЯ" in text:
                mode = "labels"
                continue
            if "Обозначения" in text:
                mode = "sources"
                continue
            if "РУССКИЙ АЛФАВИТ" in text or "АЛТАЙСКИЙ АЛФАВИТ" in text:
                mode = None
                continue
            if mode is None:
                continue

            # Split potential multi-pair lines by large spaces
            parts = re.split(r"\s{2,}", text)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                m = re.match(r"(.+?)\s*[–-]\s*(.+)", part)
                if m:
                    abbr = m.group(1).strip()
                    full = m.group(2).strip()
                    typ = "label" if mode == "labels" else "source"
                    abbr_rows.append((abbr, full, typ))
                    if typ == "label":
                        label_set.add(abbr)
                    else:
                        source_set.add(abbr)
                elif mode == "sources":
                    # continuation line for sources
                    if abbr_rows and abbr_rows[-1][2] == "source":
                        prev_abbr, prev_full, _ = abbr_rows[-1]
                        abbr_rows[-1] = (prev_abbr, prev_full + " " + part, "source")
    return abbr_rows, label_set, source_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse dictionary PDF into CSV datasets.")
    parser.add_argument("--pdf", default="словарь для модели.pdf", help="Path to PDF")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--start-page", type=int, default=14, help="0-based page index for dictionary start")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages for debug")
    parser.add_argument("--header-cutoff", type=float, default=80.0, help="Ignore lines above this y position")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        abbr_rows, label_set, source_set = parse_abbreviations(pdf)

        # Remove POS tags and refs from labels
        pos_tokens = {p for _, p in POS_PATTERNS}
        pos_tokens.update({"союз", "частица", "послелог"})
        label_set = {lab for lab in label_set if lab.lower() not in pos_tokens and lab.lower() not in {"см.", "ср."}}

        label_pattern = build_label_pattern(sorted(label_set, key=len, reverse=True))

        # Write abbreviations
        with (out_dir / "abbr_labels.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["abbr", "full", "type"])
            for abbr, full, typ in abbr_rows:
                w.writerow([abbr, full, typ])

        # Prepare CSV writers
        entries_f = (out_dir / "entries.csv").open("w", encoding="utf-8", newline="")
        senses_f = (out_dir / "senses.csv").open("w", encoding="utf-8", newline="")
        phrases_f = (out_dir / "phrases.csv").open("w", encoding="utf-8", newline="")
        examples_f = (out_dir / "examples.csv").open("w", encoding="utf-8", newline="")
        raw_pages_f = (out_dir / "raw_pages.txt").open("w", encoding="utf-8")

        entries_w = csv.DictWriter(
            entries_f,
            fieldnames=[
                "entry_id",
                "headword_raw",
                "headword_norm",
                "homonym_roman",
                "pos_primary",
                "labels",
                "raw_text",
                "page_start",
                "page_end",
            ],
        )
        senses_w = csv.DictWriter(
            senses_f,
            fieldnames=[
                "entry_id",
                "sense_id",
                "sense_group",
                "pos",
                "gloss_ru",
                "labels",
                "refs",
                "raw_block",
                "confidence_parse",
            ],
        )
        phrases_w = csv.DictWriter(
            phrases_f,
            fieldnames=[
                "entry_id",
                "phrase_id",
                "phrase_alt_raw",
                "phrase_alt_norm",
                "phrase_ru",
                "raw_block",
            ],
        )
        examples_w = csv.DictWriter(
            examples_f,
            fieldnames=[
                "entry_id",
                "sense_id",
                "ex_id",
                "example_alt_raw",
                "example_alt_norm",
                "example_ru",
                "source_note",
                "raw_line",
            ],
        )

        entries_w.writeheader()
        senses_w.writeheader()
        phrases_w.writeheader()
        examples_w.writeheader()

        entry_id = 0
        current = None

        max_pages = args.max_pages if args.max_pages is not None else len(pdf.pages)

        for page_idx in range(args.start_page, min(max_pages, len(pdf.pages))):
            page = pdf.pages[page_idx]
            lines = extract_page_lines(page, page_idx)

            # Write raw page text for control
            raw_pages_f.write(f"\n=== PAGE {page_idx + 1} COL 0 ===\n")
            for line in [l for l in lines if l["col"] == 0]:
                raw_pages_f.write(line["text"] + "\n")
            raw_pages_f.write(f"\n=== PAGE {page_idx + 1} COL 1 ===\n")
            for line in [l for l in lines if l["col"] == 1]:
                raw_pages_f.write(line["text"] + "\n")

            for line in lines:
                if line["top"] < args.header_cutoff:
                    continue
                text = line["text"].strip()
                if not text or is_page_number_line(text):
                    continue

                parsed = None
                # Quick heuristic: first token uppercase and line is near column margin
                first_token = text.split()[0] if text.split() else ""
                if is_upper_token(first_token) and (line["x0"] - line["margin"]) <= 40:
                    parsed = parse_headword_line(text)

                if parsed:
                    # finalize previous entry
                    if current:
                        finalize_entry(
                            current,
                            entries_w,
                            senses_w,
                            phrases_w,
                            examples_w,
                            label_pattern,
                            source_set,
                        )
                    entry_id += 1
                    headword_raw, homonym, remainder = parsed
                    current = {
                        "entry_id": entry_id,
                        "headword_raw": headword_raw,
                        "homonym": homonym or "",
                        "header_remainder": remainder,
                        "raw_lines": [text],
                        "body_lines": [remainder] if remainder else [],
                        "page_start": page_idx + 1,
                        "page_end": page_idx + 1,
                    }
                else:
                    if current:
                        current["raw_lines"].append(text)
                        current["body_lines"].append(text)
                        current["page_end"] = page_idx + 1

        # finalize last entry
        if current:
            finalize_entry(
                current,
                entries_w,
                senses_w,
                phrases_w,
                examples_w,
                label_pattern,
                source_set,
            )

        # Close files
        entries_f.close()
        senses_f.close()
        phrases_f.close()
        examples_f.close()
        raw_pages_f.close()



def finalize_entry(
    current: Dict,
    entries_w: csv.DictWriter,
    senses_w: csv.DictWriter,
    phrases_w: csv.DictWriter,
    examples_w: csv.DictWriter,
    label_pattern: Optional[re.Pattern],
    source_set: set,
) -> None:
    entry_id = current["entry_id"]
    raw_text = "\n".join(current["raw_lines"]).strip()
    body_text = " ".join([ln for ln in current["body_lines"] if ln]).strip()

    header_rem = current.get("header_remainder", "")
    pos_primary = extract_pos(header_rem)
    if not pos_primary:
        pos_primary = extract_pos(body_text)
    labels = extract_labels(header_rem, label_pattern)

    entry_row = {
        "entry_id": entry_id,
        "headword_raw": current["headword_raw"],
        "headword_norm": normalize_text(current["headword_raw"]),
        "homonym_roman": current.get("homonym", ""),
        "pos_primary": pos_primary,
        "labels": labels,
        "raw_text": raw_text,
        "page_start": current["page_start"],
        "page_end": current["page_end"],
    }
    entries_w.writerow(entry_row)

    # Senses and examples
    senses, examples = parse_senses(entry_id, body_text, label_pattern, source_set)
    for sense in senses:
        senses_w.writerow(sense)
    ex_id = 0
    for ex in examples:
        ex_id += 1
        examples_w.writerow(
            {
                "entry_id": ex["entry_id"],
                "sense_id": ex["sense_id"],
                "ex_id": ex_id,
                "example_alt_raw": ex["example_alt_raw"],
                "example_alt_norm": ex["example_alt_norm"],
                "example_ru": ex["example_ru"],
                "source_note": ex["source_note"],
                "raw_line": ex["raw_line"],
            }
        )

    # Phrases
    phrases = extract_phrases(body_text)
    for phr in phrases:
        phr["entry_id"] = entry_id
        phrases_w.writerow(phr)


if __name__ == "__main__":
    main()
