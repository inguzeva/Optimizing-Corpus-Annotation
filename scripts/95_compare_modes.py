from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_gold(path: Path) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    for obj in _read_jsonl(path):
        sent_id = str(obj.get("sent_id", "")).strip()
        for tok in obj.get("tokens") or []:
            if not isinstance(tok, dict):
                continue
            idx = int(tok.get("token_idx", 0))
            lbl = str(tok.get("label") or "O")
            out[(sent_id, idx)] = lbl
    return out


def load_pred(path: Path) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    for obj in _read_jsonl(path):
        sent_id = str(obj.get("sent_id", "")).strip()
        if not sent_id:
            continue

        pred = obj.get("pred")
        if isinstance(pred, list):
            for i, row in enumerate(pred):
                if isinstance(row, dict):
                    lbl = str(row.get("label") or "O")
                else:
                    lbl = "O"
                out[(sent_id, i)] = lbl
            continue

        token_labels = obj.get("token_labels")
        if isinstance(token_labels, list):
            for i, row in enumerate(token_labels):
                if isinstance(row, dict):
                    lbl = str(row.get("label") or "O")
                else:
                    lbl = "O"
                out[(sent_id, i)] = lbl
            continue

        tokens = obj.get("tokens")
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], dict):
            for i, row in enumerate(tokens):
                if not isinstance(row, dict):
                    continue
                idx = int(row.get("token_idx", i))
                lbl = str(row.get("label") or row.get("model_label") or "O")
                out[(sent_id, idx)] = lbl
            continue

        labels = obj.get("labels")
        if isinstance(labels, list):
            for i, lbl in enumerate(labels):
                out[(sent_id, i)] = str(lbl or "O")
            continue

    return out


def compute_binary_lex_metrics(
    gold: Dict[Tuple[str, int], str],
    pred: Dict[Tuple[str, int], str],
) -> dict:
    keys = set(gold.keys()) & set(pred.keys())
    if not keys:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
            "aligned_tokens": 0,
        }

    tp = 0
    fp = 0
    fn = 0
    support = 0
    for k in keys:
        g = gold[k] != "O"
        p = pred[k] != "O"
        if g:
            support += 1
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif (not p) and g:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "support": support,
        "aligned_tokens": len(keys),
    }


def compute_majority_o_baseline(gold: Dict[Tuple[str, int], str]) -> dict:
    support = sum(1 for lbl in gold.values() if lbl != "O")
    return {
        "tp": 0,
        "fp": 0,
        "fn": support,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": support,
        "aligned_tokens": len(gold),
    }


def _delta(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    if not a or not b:
        return None
    return {
        "precision": round(a.get("precision", 0.0) - b.get("precision", 0.0), 6),
        "recall": round(a.get("recall", 0.0) - b.get("recall", 0.0), 6),
        "f1": round(a.get("f1", 0.0) - b.get("f1", 0.0), 6),
    }


def _load_optional(path: Path) -> Optional[Dict[Tuple[str, int], str]]:
    if not path.exists():
        return None
    return load_pred(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare weak modes, ML model and external baseline on gold labels."
    )
    parser.add_argument("--gold", default="corpora/gold/gold_independent_65.jsonl")
    parser.add_argument(
        "--text-only",
        dest="text_only",
        default="corpora/weak_labels/token_level_labels_text_only.jsonl",
    )
    parser.add_argument(
        "--multimodal",
        default="corpora/weak_labels/token_level_labels_multimodal.jsonl",
    )
    parser.add_argument(
        "--model",
        default="corpora/weak_labels/model_predictions.jsonl",
        help="Predictions from local ML model.",
    )
    parser.add_argument(
        "--external-baseline",
        dest="external_baseline",
        default="corpora/weak_labels/external_baseline_predictions.jsonl",
        help="Predictions from external baseline model.",
    )
    parser.add_argument("--out", default="reports/mode_comparison.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    gold_path = ROOT / args.gold
    text_only_path = ROOT / args.text_only
    multimodal_path = ROOT / args.multimodal
    model_path = ROOT / args.model
    external_path = ROOT / args.external_baseline
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not gold_path.exists():
        print(f"[FAIL] Missing gold file: {gold_path}")
        return 1
    if not text_only_path.exists():
        print(f"[FAIL] Missing text-only file: {text_only_path}")
        return 1
    if not multimodal_path.exists():
        print(f"[FAIL] Missing multimodal file: {multimodal_path}")
        return 1

    gold = load_gold(gold_path)
    pred_text = load_pred(text_only_path)
    pred_mm = load_pred(multimodal_path)
    pred_model = _load_optional(model_path)
    pred_external = _load_optional(external_path)

    m_text = compute_binary_lex_metrics(gold, pred_text)
    m_mm = compute_binary_lex_metrics(gold, pred_mm)
    m_model = compute_binary_lex_metrics(gold, pred_model) if pred_model is not None else None
    m_external = compute_binary_lex_metrics(gold, pred_external) if pred_external is not None else None
    m_majority_o = compute_majority_o_baseline(gold)

    report = {
        "gold_path": str(gold_path),
        "text_only_path": str(text_only_path),
        "multimodal_path": str(multimodal_path),
        "model_path": str(model_path) if model_path.exists() else None,
        "external_baseline_path": str(external_path) if external_path.exists() else None,
        "text_only": m_text,
        "multimodal": m_mm,
        "model": m_model,
        "external_baseline": m_external,
        "majority_o_baseline": m_majority_o,
        "delta": {
            "multimodal_minus_text_only": _delta(m_mm, m_text),
            "model_minus_multimodal": _delta(m_model, m_mm),
            "model_minus_text_only": _delta(m_model, m_text),
            "model_minus_external": _delta(m_model, m_external),
            "external_minus_majority_o": _delta(m_external, m_majority_o),
            "model_minus_majority_o": _delta(m_model, m_majority_o),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] Mode comparison complete.")
    print(f"[OK] saved: {out_path}")
    print(
        f"[INFO] f1 text-only={m_text['f1']} multimodal={m_mm['f1']} "
        f"model={m_model['f1'] if m_model else 'n/a'} external={m_external['f1'] if m_external else 'n/a'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
