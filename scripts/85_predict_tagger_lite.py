from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _id2label_lookup(id2label, idx: int) -> str:
    if isinstance(id2label, dict):
        if idx in id2label:
            return str(id2label[idx])
        if str(idx) in id2label:
            return str(id2label[str(idx)])
    if isinstance(id2label, list) and 0 <= idx < len(id2label):
        return str(id2label[idx])
    return "O"


def predict_rows(
    model_dir: Path,
    input_path: Path,
    max_len: int = 256,
) -> List[dict]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    device = _select_device(torch)
    model.to(device)

    out: List[dict] = []
    for obj in read_jsonl(input_path):
        sent_id = str(obj.get("sent_id", "")).strip()
        text = str(obj.get("text", "") or "")
        tokens = obj.get("tokens") or []
        if not sent_id or not isinstance(tokens, list) or not tokens:
            continue

        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits  # [1, seq, labels]
            probs = torch.softmax(logits, dim=-1)[0]  # [seq, labels]
            pred_ids = probs.argmax(dim=-1).tolist()
            pred_scores = probs.max(dim=-1).values.tolist()

        word_ids = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
        ).word_ids()

        # На слово берем самый уверенный сабтокен.
        word_best: Dict[int, Tuple[int, float]] = {}
        for tidx, wid in enumerate(word_ids):
            if wid is None:
                continue
            pid = int(pred_ids[tidx])
            sc = float(pred_scores[tidx])
            prev = word_best.get(wid)
            if prev is None or sc > prev[1]:
                word_best[wid] = (pid, sc)

        pred = []
        for i in range(len(tokens)):
            if i in word_best:
                pid, sc = word_best[i]
                pred.append(
                    {
                        "label": _id2label_lookup(model.config.id2label, pid),
                        "confidence": round(float(sc), 6),
                    }
                )
            else:
                pred.append({"label": "O", "confidence": 0.0})

        out.append(
            {
                "sent_id": sent_id,
                "text": text,
                "tokens": tokens,
                "pred": pred,
            }
        )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model_predictions.jsonl from trained token tagger.")
    parser.add_argument(
        "--model-dir",
        default="models/tagger_lite",
        help="Path to trained token-classification model directory (relative to project root).",
    )
    parser.add_argument(
        "--input",
        default="corpora/weak_labels/token_level_labels.jsonl",
        help="Input token-level corpus JSONL (sent_id/tokens).",
    )
    parser.add_argument(
        "--output",
        default="corpora/weak_labels/model_predictions.jsonl",
        help="Output predictions JSONL for DB import.",
    )
    parser.add_argument("--max-len", type=int, default=256, help="Max tokenized sequence length.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = ROOT / args.model_dir
    input_path = ROOT / args.input
    output_path = ROOT / args.output

    if not model_dir.exists():
        print(f"[FAIL] model dir not found: {model_dir}")
        return 1
    if not input_path.exists():
        print(f"[FAIL] input file not found: {input_path}")
        return 1

    rows = predict_rows(model_dir=model_dir, input_path=input_path, max_len=int(args.max_len))
    if not rows:
        print("[FAIL] no rows predicted (input may be empty).")
        return 1

    write_jsonl(output_path, rows)
    print("[OK] Predictions generated.")
    print(f"[OK] rows: {len(rows)}")
    print(f"[OK] model: {model_dir}")
    print(f"[OK] saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
