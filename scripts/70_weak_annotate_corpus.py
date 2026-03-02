from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

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


def _clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return float(x)


def _safe_float(v: str) -> float:
    s = str(v or "").strip().replace(",", ".")
    if not s:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs():
    (ROOT / "corpora" / "weak_labels").mkdir(parents=True, exist_ok=True)


def read_csv_map(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def iter_corpus_sentences(corpus_path: Path):
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)

    if corpus_path.suffix.lower() == ".jsonl":
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    yield str(obj.get("sent_id", i)), text
    else:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                text = line.strip()
                if text:
                    yield str(i), text


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def _build_pdf_score(entry_row: Dict[str, str], entry_senses: List[Dict[str, str]]) -> float:
    page_info = 1.0 if ((entry_row.get("page_start") or "").strip() or (entry_row.get("page_end") or "").strip()) else 0.0
    raw_text = (entry_row.get("raw_text") or "")
    raw_len_score = 1.0 if len(raw_text) >= 40 else 0.4 if raw_text else 0.0
    line_count = len([x for x in raw_text.splitlines() if x.strip()])
    line_score = 1.0 if line_count >= 2 else 0.4 if line_count == 1 else 0.0
    parse_vals = [_clamp01(_safe_float(x.get("confidence_parse") or "")) for x in entry_senses]
    parse_mean = (sum(parse_vals) / len(parse_vals)) if parse_vals else 0.0

    score = (
        0.35 * page_info
        + 0.20 * raw_len_score
        + 0.15 * line_score
        + 0.30 * parse_mean
    )
    return round(_clamp01(score), 6)


def build_lexicon(clean_dir: Path, cfg: dict) -> dict:
    entries_file = clean_dir / cfg.get("ENTRIES_FILE", "entries_clean.csv")
    senses_file = clean_dir / cfg.get("SENSES_FILE", "senses_clean.csv")
    phrases_file = clean_dir / cfg.get("PHRASES_FILE", "phrases_clean.csv")

    if not entries_file.exists():
        raise FileNotFoundError(entries_file)

    entries = read_csv_map(entries_file)
    senses = read_csv_map(senses_file) if senses_file.exists() else []
    phrases = read_csv_map(phrases_file) if phrases_file.exists() else []

    entry_by_norm: Dict[str, List[str]] = {}
    entry_meta: Dict[str, Dict[str, str]] = {}
    senses_by_entry: Dict[str, List[Dict[str, str]]] = {}
    phrase_list: List[Dict[str, str]] = []

    for e in entries:
        eid = (e.get("entry_id") or "").strip()
        if not eid:
            continue

        hw_raw = (e.get("headword_raw") or "").strip()
        hw_norm = normalize_for_match((e.get("headword_norm") or "").strip() or hw_raw)
        if not hw_norm:
            continue

        entry_by_norm.setdefault(hw_norm, []).append(eid)
        entry_meta[eid] = {
            "headword_raw": hw_raw,
            "headword_norm": hw_norm,
            "pos_primary": (e.get("pos_primary") or "").strip(),
            "labels": (e.get("labels") or "").strip(),
            "page_start": (e.get("page_start") or "").strip(),
            "page_end": (e.get("page_end") or "").strip(),
            "raw_text": (e.get("raw_text") or ""),
        }

    for s in senses:
        eid = (s.get("entry_id") or "").strip()
        if not eid:
            continue
        senses_by_entry.setdefault(eid, []).append(s)

    for p in phrases:
        eid = (p.get("entry_id") or "").strip()
        alt_raw = (p.get("phrase_alt_raw") or "").strip()
        alt_norm = normalize_for_match((p.get("phrase_alt_norm") or "").strip() or alt_raw)
        if not eid or not alt_norm:
            continue
        phrase_list.append(
            {
                "entry_id": eid,
                "phrase_alt_raw": alt_raw,
                "phrase_alt_norm": alt_norm,
                "phrase_ru": (p.get("phrase_ru") or "").strip(),
            }
        )

    dop_path = Path(cfg.get("DOPSLOVAR_PATH", "dopslovar.txt"))
    if not dop_path.is_absolute():
        dop_path = ROOT / dop_path
    if dop_path.exists():
        dop_pairs = parse_dopslovar(dop_path)
        for i, row in enumerate(dop_pairs, start=1):
            hw_raw = (row.get("headword") or "").strip()
            gloss_ru = (row.get("gloss_ru") or "").strip()
            if not hw_raw or not gloss_ru:
                continue

            hw_norm = normalize_for_match(hw_raw)
            if not hw_norm:
                continue

            existing_ids = entry_by_norm.get(hw_norm) or []
            if existing_ids:
                eid = existing_ids[0]
            else:
                eid = f"dop_{i}"
                entry_by_norm.setdefault(hw_norm, []).append(eid)
                entry_meta[eid] = {
                    "headword_raw": hw_raw,
                    "headword_norm": hw_norm,
                    "pos_primary": "",
                    "labels": "dopslovar",
                    "page_start": "",
                    "page_end": "",
                    "raw_text": f"{hw_raw}\n{gloss_ru}",
                }

            senses_by_entry.setdefault(eid, []).append(
                {
                    "entry_id": eid,
                    "sense_id": str(len(senses_by_entry.get(eid, [])) + 1),
                    "gloss_ru": gloss_ru,
                    "confidence_parse": "0.2",
                }
            )

    entry_pdf_score: Dict[str, float] = {}
    for eid, meta in entry_meta.items():
        entry_pdf_score[eid] = _build_pdf_score(meta, senses_by_entry.get(eid, []))

    phrase_list.sort(key=lambda x: len(x["phrase_alt_norm"]), reverse=True)

    return {
        "entry_by_norm": entry_by_norm,
        "entry_meta": entry_meta,
        "entry_pdf_score": entry_pdf_score,
        "senses_by_entry": senses_by_entry,
        "phrases": phrase_list,
    }


def choose_best_entry(entry_ids: List[str], senses_by_entry: Dict[str, List[Dict[str, str]]]) -> Tuple[str, float]:
    if len(entry_ids) == 1:
        return entry_ids[0], 0.85

    best = entry_ids[0]
    best_sc = -1
    for eid in entry_ids:
        sc = len(senses_by_entry.get(eid, []))
        if sc > best_sc:
            best_sc = sc
            best = eid

    conf = 0.55 if len(entry_ids) <= 3 else 0.45
    return best, conf


def _find_phrase_spans(token_norms: List[str], phrase_norm: str) -> List[Tuple[int, int]]:
    p_toks = [x for x in phrase_norm.split() if x]
    if not p_toks:
        return []
    spans = []
    n = len(p_toks)
    for i in range(0, len(token_norms) - n + 1):
        if token_norms[i : i + n] == p_toks:
            spans.append((i, i + n - 1))
    return spans


def _merge_confidence(text_conf: float, pdf_conf: float, mode: str, pdf_weight: float) -> float:
    text_conf = _clamp01(text_conf)
    pdf_conf = _clamp01(pdf_conf)
    if mode == "text-only":
        return text_conf
    w = _clamp01(pdf_weight)
    return _clamp01((1.0 - w) * text_conf + w * pdf_conf)


def weak_annotate_sentence(
    sent_id: str,
    text: str,
    lex: dict,
    mode: str,
    pdf_weight: float,
) -> Tuple[dict, dict]:
    tokens = tokenize(text)
    token_norms = [normalize_for_match(t) for t in tokens]
    sent_norm = normalize_for_match(text)

    token_labels: List[dict] = []
    for tok in tokens:
        token_labels.append(
            {
                "token": tok,
                "label": "O",
                "entry_id": "",
                "headword": "",
                "pos": "",
                "source": "none",
                "text_confidence": 0.0,
                "pdf_confidence": 0.0,
                "multimodal_confidence": 0.0,
                "confidence": 0.0,
            }
        )

    lex_hits = []
    covered = False

    # lemma hits
    for idx, (tok, tnorm) in enumerate(zip(tokens, token_norms)):
        if not tnorm or re.fullmatch(r"[^\w]+", tok or ""):
            continue

        entry_ids = lex["entry_by_norm"].get(tnorm)
        if not entry_ids:
            continue

        eid, text_conf = choose_best_entry(entry_ids, lex["senses_by_entry"])
        pdf_conf = lex["entry_pdf_score"].get(eid, 0.0)
        mm_conf = _merge_confidence(text_conf, pdf_conf, mode=mode, pdf_weight=pdf_weight)
        meta = lex["entry_meta"].get(eid, {})

        token_labels[idx] = {
            "token": tok,
            "label": "LEX",
            "entry_id": eid,
            "headword": meta.get("headword_raw", meta.get("headword_norm", "")),
            "pos": meta.get("pos_primary", ""),
            "source": "both" if mode != "text-only" else "text",
            "text_confidence": round(text_conf, 6),
            "pdf_confidence": round(pdf_conf, 6) if mode != "text-only" else 0.0,
            "multimodal_confidence": round(mm_conf, 6),
            "confidence": round(mm_conf, 6),  # legacy field for downstream compatibility
        }
        lex_hits.append(
            {
                "entry_id": eid,
                "headword": token_labels[idx]["headword"],
                "pos": token_labels[idx]["pos"],
                "match": "lemma",
                "span": [idx, idx],
                "text_confidence": round(text_conf, 6),
                "pdf_confidence": round(pdf_conf, 6) if mode != "text-only" else 0.0,
                "multimodal_confidence": round(mm_conf, 6),
            }
        )
        covered = True

    # phrase hits
    phrase_hits_count = 0
    for ph in lex["phrases"]:
        pn = ph["phrase_alt_norm"]
        if not pn or pn not in sent_norm:
            continue
        spans = _find_phrase_spans(token_norms, pn)
        if not spans:
            continue

        phrase_hits_count += 1
        eid = ph["entry_id"]
        pdf_conf = lex["entry_pdf_score"].get(eid, 0.0)
        text_conf = 0.95
        mm_conf = _merge_confidence(text_conf, pdf_conf, mode=mode, pdf_weight=pdf_weight)

        for st, en in spans:
            for i in range(st, en + 1):
                cur = token_labels[i]
                cur_conf = float(cur.get("multimodal_confidence") or 0.0)
                if cur_conf >= mm_conf and cur.get("label") == "LEX":
                    continue
                meta = lex["entry_meta"].get(eid, {})
                token_labels[i] = {
                    "token": tokens[i],
                    "label": "LEX",
                    "entry_id": eid,
                    "headword": meta.get("headword_raw", meta.get("headword_norm", "")),
                    "pos": meta.get("pos_primary", ""),
                    "source": "both" if mode != "text-only" else "text",
                    "text_confidence": round(text_conf, 6),
                    "pdf_confidence": round(pdf_conf, 6) if mode != "text-only" else 0.0,
                    "multimodal_confidence": round(mm_conf, 6),
                    "confidence": round(mm_conf, 6),
                }
            lex_hits.append(
                {
                    "entry_id": eid,
                    "headword": lex["entry_meta"].get(eid, {}).get("headword_raw", ""),
                    "pos": lex["entry_meta"].get(eid, {}).get("pos_primary", ""),
                    "match": "phrase",
                    "phrase": ph["phrase_alt_raw"],
                    "phrase_norm": pn,
                    "span": [st, en],
                    "text_confidence": round(text_conf, 6),
                    "pdf_confidence": round(pdf_conf, 6) if mode != "text-only" else 0.0,
                    "multimodal_confidence": round(mm_conf, 6),
                }
            )
            covered = True

    sent_row = {
        "sent_id": sent_id,
        "text": text,
        "text_norm": sent_norm,
        "covered": 1 if covered else 0,
        "hits_count": len(lex_hits),
        "phrase_hits_count": phrase_hits_count,
        "mode": mode,
        "hits": lex_hits,
    }

    token_row = {
        "sent_id": sent_id,
        "text": text,
        "mode": mode,
        "tokens": tokens,
        "token_labels": token_labels,
    }

    return sent_row, token_row


def _parse_args(cfg: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weakly annotate corpus with dictionary hits.")
    parser.add_argument(
        "--mode",
        choices=["text-only", "text+pdf"],
        default=cfg.get("WEAK_MODE", "text+pdf"),
        help="Scoring mode for confidence aggregation.",
    )
    parser.add_argument(
        "--pdf-weight",
        type=float,
        default=float(cfg.get("WEAK_PDF_WEIGHT", 0.30)),
        help="Weight of PDF modality in multimodal confidence.",
    )
    parser.add_argument(
        "--out-sent",
        default="corpora/weak_labels/sentence_level_lexhits.jsonl",
        help="Output JSONL path for sentence-level hits (relative to project root).",
    )
    parser.add_argument(
        "--out-token",
        default="corpora/weak_labels/token_level_labels.jsonl",
        help="Output JSONL path for token-level labels (relative to project root).",
    )
    return parser.parse_args()


def main() -> int:
    ensure_dirs()
    cfg = load_config()
    args = _parse_args(cfg)

    clean_dir = ROOT / "data" / "processed" / "clean"
    if not clean_dir.exists():
        print("[FAIL] Missing data/processed/clean (need clean dictionary CSV).")
        return 1

    corpus_cfg = cfg.get("corpus", {}) or {}
    corpus_path = corpus_cfg.get("path") or cfg.get("CORPUS_PATH") or "corpora/input/corpus.txt"
    corpus_path = ROOT / corpus_path

    out_sent = ROOT / args.out_sent
    out_tok = ROOT / args.out_token
    out_sent.parent.mkdir(parents=True, exist_ok=True)
    out_tok.parent.mkdir(parents=True, exist_ok=True)

    lex = build_lexicon(clean_dir, cfg)

    total = 0
    covered = 0
    total_hits = 0

    with open(out_sent, "w", encoding="utf-8") as fs, open(out_tok, "w", encoding="utf-8") as ft:
        for sent_id, text in iter_corpus_sentences(corpus_path):
            total += 1
            sent_row, token_row = weak_annotate_sentence(
                sent_id=sent_id,
                text=text,
                lex=lex,
                mode=args.mode,
                pdf_weight=float(args.pdf_weight),
            )

            if sent_row["covered"] == 1:
                covered += 1
            total_hits += sent_row["hits_count"]

            fs.write(json.dumps(sent_row, ensure_ascii=False) + "\n")
            ft.write(json.dumps(token_row, ensure_ascii=False) + "\n")

    coverage_pct = round((covered / total) * 100.0, 2) if total else 0.0
    avg_hits = round(total_hits / total, 3) if total else 0.0

    print("[OK] Weak annotation complete.")
    print(f"[INFO] mode={args.mode} pdf_weight={float(args.pdf_weight):.3f}")
    print(f"[INFO] corpus={corpus_path}")
    print(f"[INFO] sentences={total} covered={covered} ({coverage_pct}%) avg_hits={avg_hits}")
    print(f"[OK] saved: {out_sent}")
    print(f"[OK] saved: {out_tok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
