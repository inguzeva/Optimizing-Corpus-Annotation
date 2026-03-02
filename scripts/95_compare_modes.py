from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_gold(path: Path) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    for obj in _read_jsonl(path):
        sent_id = str(obj.get("sent_id", ""))
        for tok in obj.get("tokens") or []:
            idx = int(tok.get("token_idx", 0))
            lbl = str(tok.get("label") or "O")
            out[(sent_id, idx)] = lbl
    return out


def load_pred(path: Path) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    for obj in _read_jsonl(path):
        sent_id = str(obj.get("sent_id", ""))
        token_labels = obj.get("token_labels") or []
        for i, row in enumerate(token_labels):
            if isinstance(row, dict):
                lbl = str(row.get("label") or "O")
            else:
                lbl = "O"
            out[(sent_id, i)] = lbl
    return out


def compute_binary_lex_metrics(gold: Dict[Tuple[str, int], str], pred: Dict[Tuple[str, int], str]) -> dict:
    keys = set(gold.keys()) & set(pred.keys())
    if not keys:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare text-only and text+pdf weak modes on gold labels.")
    parser.add_argument("--gold", default="corpora/gold/gold_300.jsonl")
    parser.add_argument("--text-only", dest="text_only", default="corpora/weak_labels/token_level_labels_text_only.jsonl")
    parser.add_argument("--multimodal", default="corpora/weak_labels/token_level_labels_multimodal.jsonl")
    parser.add_argument("--out", default="reports/mode_comparison.json")
    args = parser.parse_args()

    gold_path = ROOT / args.gold
    text_only_path = ROOT / args.text_only
    multimodal_path = ROOT / args.multimodal
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

    m_text = compute_binary_lex_metrics(gold, pred_text)
    m_mm = compute_binary_lex_metrics(gold, pred_mm)

    report = {
        "gold_path": str(gold_path),
        "text_only_path": str(text_only_path),
        "multimodal_path": str(multimodal_path),
        "text_only": m_text,
        "multimodal": m_mm,
        "delta": {
            "precision": round(m_mm["precision"] - m_text["precision"], 6),
            "recall": round(m_mm["recall"] - m_text["recall"], 6),
            "f1": round(m_mm["f1"] - m_text["f1"], 6),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] Mode comparison complete.")
    print(f"[OK] saved: {out_path}")
    print(f"[INFO] text-only f1={m_text['f1']} multimodal f1={m_mm['f1']} delta={report['delta']['f1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
