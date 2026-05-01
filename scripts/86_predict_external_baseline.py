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


def _extract_tokens(obj: dict) -> List[str]:
    tokens = obj.get("tokens")
    if isinstance(tokens, list) and tokens:
        first = tokens[0]
        if isinstance(first, str):
            return [str(x) for x in tokens]
        if isinstance(first, dict):
            out = []
            for row in tokens:
                if not isinstance(row, dict):
                    continue
                out.append(str(row.get("token") or ""))
            return [x for x in out if x]

    token_labels = obj.get("token_labels")
    if isinstance(token_labels, list) and token_labels and isinstance(token_labels[0], dict):
        out = [str(x.get("token") or "") for x in token_labels if isinstance(x, dict)]
        return [x for x in out if x]

    return []


def _map_external_to_lex(raw_label: str) -> str:
    s = (raw_label or "O").strip().upper()
    if s in {"", "O", "LABEL_0"}:
        return "O"
    return "LEX"


def predict_rows(
    model_name: str,
    input_path: Path,
    max_len: int = 256,
) -> List[dict]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()

    device = _select_device(torch)
    model.to(device)

    out: List[dict] = []
    for obj in read_jsonl(input_path):
        sent_id = str(obj.get("sent_id", "")).strip()
        text = str(obj.get("text", "") or "")
        tokens = _extract_tokens(obj)

        if not sent_id or not tokens:
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
                raw = _id2label_lookup(model.config.id2label, pid)
                pred.append(
                    {
                        "label": _map_external_to_lex(raw),
                        "raw_label": raw,
                        "confidence": round(float(sc), 6),
                    }
                )
            else:
                pred.append({"label": "O", "raw_label": "O", "confidence": 0.0})

        out.append(
            {
                "sent_id": sent_id,
                "text": text,
                "tokens": tokens,
                "pred": pred,
                "baseline_model": model_name,
            }
        )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate external baseline predictions using an external HF token-classification model."
    )
    parser.add_argument(
        "--model-name",
        default="Davlan/xlm-roberta-base-ner-hrl",
        help="HF model name or local path for token classification.",
    )
    parser.add_argument(
        "--input",
        default="corpora/gold/gold_independent_65.jsonl",
        help="Input JSONL (needs sent_id/text/tokens).",
    )
    parser.add_argument(
        "--output",
        default="corpora/weak_labels/external_baseline_predictions.jsonl",
        help="Output predictions JSONL path.",
    )
    parser.add_argument("--max-len", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output

    if not input_path.exists():
        print(f"[FAIL] input file not found: {input_path}")
        return 1

    rows = predict_rows(
        model_name=str(args.model_name),
        input_path=input_path,
        max_len=int(args.max_len),
    )

    if not rows:
        print("[FAIL] no rows predicted (input may be empty or malformed).")
        return 1

    write_jsonl(output_path, rows)
    print("[OK] External baseline predictions generated.")
    print(f"[INFO] model={args.model_name}")
    print(f"[INFO] rows={len(rows)}")
    print(f"[OK] saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
