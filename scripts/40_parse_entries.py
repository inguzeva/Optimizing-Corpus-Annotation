
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

import yaml


ROOT = Path(__file__).resolve().parents[1]


# -------------------------
# Optional reuse of app normalization
# -------------------------
def _normalize_for_match_fallback(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


try:
    # allow running from project root
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from app.services.normalize import normalize_for_match as _normalize_for_match  # type: ignore
except Exception:
    _normalize_for_match = _normalize_for_match_fallback


# -------------------------
# Regexes
# -------------------------
RE_ROMAN = re.compile(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)\b")
RE_POS = re.compile(
    r"\b(сущ\.|прил\.|гл\.|нареч\.|межд\.|местоим\.|числ\.|союз\.|част\.|предл\.|вводн\.)\b",
    re.IGNORECASE,
)

RE_LABEL_TOKEN = re.compile(r"\b[А-Яа-яA-Za-zЁёӦӧӰӱҮүЈј]+\.\b")

RE_SENSE_PAREN = re.compile(r"(?<!\d)(\d{1,2})\)")
RE_SENSE_DOT = re.compile(r"(?<!\d)(\d{1,2})\.")

RE_DIAMOND = re.compile(r"[♦◆]")

# Пример "ALT — RU" (допускаем разные тире)
RE_EXAMPLE_DASH = re.compile(r"(.+?)\s[—\-–]\s(.+)")
RE_SOURCE_TAIL = re.compile(r"\(([А-ЯЁA-Z]{1,4})\)\s*$")


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs():
    (ROOT / "data" / "processed" / "parsed").mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_spaces(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def split_head_line(head_line: str) -> Tuple[str, str, str]:
    """
    Возвращает (headword_raw, homonym_roman, rest_after_head)
    """
    head_line = clean_spaces(head_line)
    if not head_line:
        return "", "", ""

    parts = head_line.split(" ", 2)
    headword = parts[0].strip()

    tail = head_line[len(headword):].strip()
    hom = ""

    # омоним римскими I/II/... сразу после headword
    m = RE_ROMAN.match(tail)
    if m:
        hom = m.group(1)
        tail = tail[m.end():].strip()

    return headword, hom, tail


def extract_pos_and_labels(rest: str) -> Tuple[str, str, str]:
    """
    Из остатка строки заголовка пытаемся извлечь pos_primary и labels.
    Возвращает (pos_primary, labels, rest_wo_pos_labels)
    """
    rest = clean_spaces(rest)
    if not rest:
        return "", "", ""

    pos = ""
    m = RE_POS.search(rest)
    if m:
        pos = m.group(1)
        # вырежем найденный POS из строки (первое вхождение)
        rest_wo_pos = (rest[:m.start()] + " " + rest[m.end():]).strip()
    else:
        rest_wo_pos = rest

    # labels: любые "слова." до начала смысла; это грубо, но работает
    # соберём все токены вида "разг." "перен." "зоол." и т.п.
    labels_tokens = []
    for t in RE_LABEL_TOKEN.findall(rest_wo_pos):
        # исключаем POS если он тоже с точкой
        if pos and t.lower() == pos.lower():
            continue
        labels_tokens.append(t)

    # удалим label-токены из остатка
    rest_clean = rest_wo_pos
    for t in set(labels_tokens):
        rest_clean = re.sub(rf"\b{re.escape(t)}\b", " ", rest_clean)
    rest_clean = clean_spaces(rest_clean)

    labels = ";".join(sorted(set(labels_tokens), key=lambda x: x.lower()))
    return pos, labels, rest_clean


def split_senses(text: str) -> List[Dict[str, Any]]:
    """
    Делит текст на значения по маркерам "1)" и "1.".
    Возвращает список dict: {sense_id, sense_group, raw_block, confidence_parse}
    """
    t = text.strip()
    if not t:
        return []

    markers: List[Tuple[int, str, str]] = []  # (pos, sense_id, group)
    for m in RE_SENSE_PAREN.finditer(t):
        markers.append((m.start(), m.group(1), "paren"))
    for m in RE_SENSE_DOT.finditer(t):
        markers.append((m.start(), m.group(1), "dot"))

    markers.sort(key=lambda x: x[0])

    # если маркеров нет — одно значение
    if not markers:
        return [{
            "sense_id": "1",
            "sense_group": "none",
            "raw_block": t,
            "confidence_parse": "0.4",
        }]

    # если первый маркер не в начале, префикс считаем как sense 1 (none), если он не пустой
    senses: List[Dict[str, Any]] = []
    if markers[0][0] > 0:
        prefix = t[:markers[0][0]].strip()
        if prefix:
            senses.append({
                "sense_id": "1",
                "sense_group": "none",
                "raw_block": prefix,
                "confidence_parse": "0.6",
            })

    for i, (pos, sid, group) in enumerate(markers):
        start = pos
        end = markers[i + 1][0] if i + 1 < len(markers) else len(t)
        block = t[start:end].strip()
        senses.append({
            "sense_id": sid,
            "sense_group": group,
            "raw_block": block,
            "confidence_parse": "0.7",
        })

    return senses


def extract_gloss_from_block(raw_block: str) -> str:
    """
    Грубо извлекает gloss_ru:
    - убираем ведущий маркер "1)" / "1."
    - берём первую часть до первого явного примера "—" или до конца строки
    """
    b = raw_block.strip()

    # убрать ведущие "1)" или "1."
    b = re.sub(r"^\s*\d{1,2}[\)\.]\s*", "", b)

    # часто gloss сначала, потом примеры; если есть " — ", пытаемся отрезать до примера
    # но осторожно: в gloss тоже может быть тире. Оставим простой порог:
    # если нашли " ALT — RU " и ALT выглядит не русским (много некирилл?), тогда отрезаем.
    # В упрощённой версии: отрезаем по первому " ; " или по двум пробелам? Не надёжно.
    # Поэтому: берем строку целиком, но обрезаем длину по разумному максимуму.
    b = clean_spaces(b)
    return b


def extract_examples_from_block(entry_id: str, sense_id: str, raw_block: str) -> List[Dict[str, Any]]:
    """
    Примеры по шаблону "ALT — RU" + источник (АБ) в конце.
    Возвращает список rows для examples.csv.
    """
    out = []
    lines = [x.strip() for x in raw_block.split("\n") if x.strip()]
    ex_counter = 0

    for ln in lines:
        m = RE_EXAMPLE_DASH.match(ln)
        if not m:
            continue

        left = clean_spaces(m.group(1))
        right = clean_spaces(m.group(2))

        source_note = ""
        sm = RE_SOURCE_TAIL.search(right)
        if sm:
            source_note = sm.group(1)
            right = clean_spaces(right[:sm.start()])

        ex_counter += 1
        out.append({
            "entry_id": str(entry_id),
            "sense_id": str(sense_id),
            "ex_id": str(ex_counter),
            "example_alt_raw": left,
            "example_alt_norm": _normalize_for_match(left),
            "example_ru": right,
            "source_note": source_note,
            "raw_line": ln,
        })

    return out


def split_phrases(diamond_text: str) -> List[Tuple[str, str, str]]:
    """
    Режем ♦-блок на отдельные фразы.
    Возвращаем список (phrase_alt_raw, phrase_alt_norm, phrase_ru)
    """
    t = diamond_text.strip()
    if not t:
        return []

    # иногда фразы идут в строку через ';'
    chunks = []
    for ln in t.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        parts = [p.strip() for p in ln.split(";") if p.strip()]
        chunks.extend(parts)

    phrases = []
    for ch in chunks:
        # разделим по тире, если это "ALT — RU"
        m = RE_EXAMPLE_DASH.match(ch)
        if m:
            alt = clean_spaces(m.group(1))
            ru = clean_spaces(m.group(2))
        else:
            alt = clean_spaces(ch)
            ru = ""

        phrases.append((alt, _normalize_for_match(alt), ru))

    return phrases


def main() -> int:
    ensure_dirs()
    cfg = load_config()

    in_path = ROOT / "data" / "interim" / "entries_raw.jsonl"
    if not in_path.exists():
        print(f"[FAIL] Missing input: {in_path}")
        print("Run first: python scripts/30_segment_entries.py")
        return 1

    out_dir = ROOT / "data" / "processed" / "parsed"
    entries_csv = out_dir / "entries.csv"
    senses_csv = out_dir / "senses.csv"
    phrases_csv = out_dir / "phrases.csv"
    examples_csv = out_dir / "examples.csv"
    abbr_csv = out_dir / "abbr_labels.csv"

    # writers
    entries_fields = [
        "entry_id",
        "headword_raw",
        "headword_norm",
        "homonym_roman",
        "pos_primary",
        "labels",
        "has_phrases",
        "raw_text",
        "page_start",
        "page_end",
    ]

    senses_fields = [
        "entry_id",
        "sense_id",
        "sense_group",
        "pos",
        "gloss_ru",
        "gloss_ru_alt",
        "labels",
        "refs",
        "confidence_parse",
        "raw_block",
    ]

    phrases_fields = [
        "entry_id",
        "phrase_id",
        "phrase_alt_raw",
        "phrase_alt_norm",
        "phrase_ru",
        "labels",
        "raw_block",
    ]

    examples_fields = [
        "entry_id",
        "sense_id",
        "ex_id",
        "example_alt_raw",
        "example_alt_norm",
        "example_ru",
        "source_note",
        "raw_line",
    ]

    # Если у тебя уже есть отдельный парсер сокращений — можешь заменить.
    abbr_fields = ["abbr", "full", "type"]

    records = list(read_jsonl(in_path))
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Parse entries", unit="entry")

    total_examples = 0
    total_phrases = 0
    total_senses = 0

    with open(entries_csv, "w", encoding="utf-8", newline="") as f_entries, \
         open(senses_csv, "w", encoding="utf-8", newline="") as f_senses, \
         open(phrases_csv, "w", encoding="utf-8", newline="") as f_phrases, \
         open(examples_csv, "w", encoding="utf-8", newline="") as f_examples:

        w_entries = csv.DictWriter(f_entries, fieldnames=entries_fields)
        w_senses = csv.DictWriter(f_senses, fieldnames=senses_fields)
        w_phrases = csv.DictWriter(f_phrases, fieldnames=phrases_fields)
        w_examples = csv.DictWriter(f_examples, fieldnames=examples_fields)

        w_entries.writeheader()
        w_senses.writeheader()
        w_phrases.writeheader()
        w_examples.writeheader()

        for rec in iterator:
            entry_id = rec["entry_id"]
            head_line = rec.get("head_line") or rec.get("lines", [""])[0]
            raw_text = rec.get("raw_text") or "\n".join(rec.get("lines", []))
            page_start = rec.get("page_start", "")
            page_end = rec.get("page_end", page_start)

            headword_raw, homonym_roman, rest = split_head_line(head_line)
            pos_primary, labels, rest_wo = extract_pos_and_labels(rest)

            # отделяем ♦-блок
            diamond_match = RE_DIAMOND.search(raw_text)
            before_diamond = raw_text
            diamond_block = ""
            if diamond_match:
                before_diamond = raw_text[:diamond_match.start()]
                diamond_block = raw_text[diamond_match.end():]

            # убрать первую строку заголовка из before_diamond
            # (оставим только продолжение после первой строки)
            lines = raw_text.split("\n")
            after_head = "\n".join(lines[1:]).strip()
            if diamond_match:
                # если ♦ в первой строке, after_head уже содержит дальше — ок
                # но если ♦ был в before_diamond части, оставим before_diamond без первой строки
                bd_lines = before_diamond.split("\n")
                after_head = "\n".join(bd_lines[1:]).strip()

            # Если после head ничего нет — попробуем использовать "rest_wo" как базовый смысл
            # (иногда перевод в первой строке)
            sense_text = after_head if after_head else rest_wo

            sense_blocks = split_senses(sense_text)
            has_phrases = 1 if diamond_block.strip() else 0

            # entries row
            w_entries.writerow({
                "entry_id": str(entry_id),
                "headword_raw": headword_raw,
                "headword_norm": _normalize_for_match(headword_raw),
                "homonym_roman": homonym_roman,
                "pos_primary": pos_primary,
                "labels": labels,
                "has_phrases": str(has_phrases),
                "raw_text": raw_text,
                "page_start": str(page_start),
                "page_end": str(page_end),
            })

            # senses rows + examples
            for sb in sense_blocks:
                sid = sb["sense_id"]
                group = sb["sense_group"]
                raw_block = sb["raw_block"]
                conf = sb["confidence_parse"]

                gloss = extract_gloss_from_block(raw_block)

                w_senses.writerow({
                    "entry_id": str(entry_id),
                    "sense_id": str(sid),
                    "sense_group": group,
                    "pos": "",                 # пока пусто (можно расширить позже)
                    "gloss_ru": gloss,
                    "gloss_ru_alt": "",
                    "labels": "",              # пока пусто (можно расширить позже)
                    "refs": "",                # пока пусто (см./ср. можно добавить позже)
                    "confidence_parse": conf,
                    "raw_block": raw_block,
                })
                total_senses += 1

                ex_rows = extract_examples_from_block(str(entry_id), str(sid), raw_block)
                for ex in ex_rows:
                    w_examples.writerow(ex)
                total_examples += len(ex_rows)

            # phrases rows
            phrase_rows = split_phrases(diamond_block)
            for i, (alt_raw, alt_norm, ru) in enumerate(phrase_rows, start=1):
                w_phrases.writerow({
                    "entry_id": str(entry_id),
                    "phrase_id": str(i),
                    "phrase_alt_raw": alt_raw,
                    "phrase_alt_norm": alt_norm,
                    "phrase_ru": ru,
                    "labels": "",
                    "raw_block": diamond_block.strip(),
                })
            total_phrases += len(phrase_rows)

    # abbr_labels.csv: оставим пустой файл с заголовком (если не существует)
    if not abbr_csv.exists():
        with open(abbr_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=abbr_fields)
            w.writeheader()

    print(f"[OK] Saved parsed CSV to: {out_dir}")
    print(f"[INFO] senses={total_senses} examples={total_examples} phrases={total_phrases}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
