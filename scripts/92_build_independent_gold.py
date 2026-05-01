from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]

try:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from app.services.dopslovar import parse_dopslovar  # type: ignore
    from app.services.normalize import normalize_for_match  # type: ignore
except Exception:
    def parse_dopslovar(path: Path) -> List[Dict[str, str]]:
        return []

    def normalize_for_match(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_csv_map(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def iter_corpus_sentences(corpus_path: Path) -> Iterable[Tuple[str, str]]:
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)

    if corpus_path.suffix.lower() == ".jsonl":
        for i, obj in enumerate(read_jsonl(corpus_path), start=1):
            text = str(obj.get("text") or "")
            if not text:
                continue
            sent_id = str(obj.get("sent_id") or i)
            sent_num = _extract_trailing_number(sent_id)
            yield str(sent_num if sent_num is not None else i), text
    else:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                text = line.strip()
                if text:
                    yield str(i), text


def _extract_trailing_number(sent_id: str) -> int | None:
    m = re.search(r"(\d+)$", sent_id or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def _is_word_token(token: str) -> bool:
    return bool(re.search(r"\w", token or ""))


def _find_phrase_spans(token_norms: List[str], phrase_norm: str) -> List[Tuple[int, int]]:
    phrase_toks = [x for x in phrase_norm.split() if x]
    if not phrase_toks:
        return []

    spans: List[Tuple[int, int]] = []
    n = len(phrase_toks)
    for i in range(0, len(token_norms) - n + 1):
        if token_norms[i : i + n] == phrase_toks:
            spans.append((i, i + n - 1))
    return spans


def build_lexicon(clean_dir: Path, cfg: dict) -> tuple[Set[str], List[str]]:
    entries_file = clean_dir / cfg.get("ENTRIES_FILE", "entries_clean.csv")
    phrases_file = clean_dir / cfg.get("PHRASES_FILE", "phrases_clean.csv")

    if not entries_file.exists():
        raise FileNotFoundError(entries_file)

    entries = read_csv_map(entries_file)
    phrases = read_csv_map(phrases_file) if phrases_file.exists() else []

    lex_words: Set[str] = set()
    for row in entries:
        raw = (row.get("headword_raw") or "").strip()
        norm = normalize_for_match((row.get("headword_norm") or "").strip() or raw)
        if norm:
            lex_words.add(norm)

    phrase_norms: Set[str] = set()
    for row in phrases:
        raw = (row.get("phrase_alt_raw") or "").strip()
        norm = normalize_for_match((row.get("phrase_alt_norm") or "").strip() or raw)
        if norm:
            phrase_norms.add(norm)

    dop_path = Path(cfg.get("DOPSLOVAR_PATH", "dopslovar.txt"))
    if not dop_path.is_absolute():
        dop_path = ROOT / dop_path
    if dop_path.exists():
        for pair in parse_dopslovar(dop_path):
            hw = normalize_for_match(pair.get("headword") or "")
            if hw:
                lex_words.add(hw)

    phrase_list = sorted(list(phrase_norms), key=lambda s: len(s.split()), reverse=True)
    return lex_words, phrase_list


def load_manual_overrides(path: Path) -> Dict[str, Dict[int, str]]:
    if not path.exists():
        return {}

    out: Dict[str, Dict[int, str]] = {}
    for obj in read_jsonl(path):
        sid = str(obj.get("sent_id") or "").strip()
        if not sid:
            continue

        tok_map = out.setdefault(sid, {})
        for tok in obj.get("tokens") or []:
            if not isinstance(tok, dict):
                continue
            try:
                idx = int(tok.get("token_idx"))
            except Exception:
                continue
            lbl = str(tok.get("label") or "").strip() or "O"
            tok_map[idx] = lbl

    return out


def build_sentence_labels(
    sent_id: str,
    text: str,
    lex_words: Set[str],
    phrase_list: List[str],
    overrides: Dict[str, Dict[int, str]],
) -> dict:
    tokens = tokenize(text)
    token_norms = [normalize_for_match(t) for t in tokens]

    labels = ["O"] * len(tokens)

    for i, (token, norm) in enumerate(zip(tokens, token_norms)):
        if not _is_word_token(token):
            continue
        if norm and norm in lex_words:
            labels[i] = "LEX"

    sent_norm = normalize_for_match(text)
    for phrase in phrase_list:
        if not phrase or phrase not in sent_norm:
            continue
        for st, en in _find_phrase_spans(token_norms, phrase):
            for i in range(st, en + 1):
                if _is_word_token(tokens[i]):
                    labels[i] = "LEX"

    ov = overrides.get(sent_id, {})
    for idx, lbl in ov.items():
        if 0 <= idx < len(labels):
            labels[idx] = lbl

    return {
        "sent_id": sent_id,
        "text": text,
        "tokens": [
            {
                "token_idx": i,
                "token": tok,
                "label": labels[i],
            }
            for i, tok in enumerate(tokens)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build independent token-level gold set from corpus and dictionary source-of-truth (without weak_labels input)."
    )
    parser.add_argument(
        "--corpus",
        default="corpora/input/corpus.txt",
        help="Input corpus (.txt or .jsonl) relative to project root.",
    )
    parser.add_argument(
        "--manual-overrides",
        default="corpora/gold/manual_overrides.jsonl",
        help="Optional JSONL with manual overrides (sent_id/tokens[token_idx,label]).",
    )
    parser.add_argument(
        "--out",
        default="corpora/gold/gold_independent_65.jsonl",
        help="Output gold JSONL path relative to project root.",
    )
    parser.add_argument("--min-sentences", type=int, default=50)
    parser.add_argument("--max-sentences", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()

    corpus_path = ROOT / args.corpus
    overrides_path = ROOT / args.manual_overrides
    out_path = ROOT / args.out

    clean_dir = ROOT / "data" / "processed" / "clean"
    if not clean_dir.exists():
        print("[FAIL] Missing data/processed/clean (need clean dictionary CSVs).")
        return 1

    sentences = list(iter_corpus_sentences(corpus_path))
    n = len(sentences)

    if n < int(args.min_sentences):
        print(f"[FAIL] corpus has only {n} sentences; need at least {args.min_sentences}.")
        return 1

    if n > int(args.max_sentences):
        rnd = random.Random(int(args.seed))
        sentences = sorted(rnd.sample(sentences, int(args.max_sentences)), key=lambda x: int(x[0]))

    lex_words, phrase_list = build_lexicon(clean_dir, cfg)
    overrides = load_manual_overrides(overrides_path)

    rows = [
        build_sentence_labels(
            sent_id=sid,
            text=text,
            lex_words=lex_words,
            phrase_list=phrase_list,
            overrides=overrides,
        )
        for sid, text in sentences
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    token_count = sum(len(r["tokens"]) for r in rows)
    lex_count = sum(1 for r in rows for t in r["tokens"] if t.get("label") == "LEX")
    print("[OK] Independent gold built.")
    print(f"[INFO] corpus={corpus_path} sentences={len(rows)} tokens={token_count}")
    print(f"[INFO] lex_tokens={lex_count} lex_ratio={round((lex_count / token_count) * 100, 2) if token_count else 0.0}%")
    print(f"[INFO] manual_overrides={overrides_path} sentences_with_overrides={len(overrides)}")
    print(f"[OK] saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
