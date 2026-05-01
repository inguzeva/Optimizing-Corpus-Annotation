from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs():
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "corpora" / "weak_labels").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    hf_cache = ROOT / ".cache" / "hf"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_train_eval(data: List[dict], eval_ratio: float = 0.1, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    if not data:
        return [], []
    n = len(data)
    n_eval = max(1, int(n * eval_ratio))
    rng = (seed * 1103515245 + 12345) & 0x7FFFFFFF

    order = list(range(n))
    for i in range(n - 1, 0, -1):
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        j = rng % (i + 1)
        order[i], order[j] = order[j], order[i]

    eval_idx = set(order[:n_eval])
    train = [data[i] for i in range(n) if i not in eval_idx]
    eval_ = [data[i] for i in range(n) if i in eval_idx]
    return train, eval_


def id2label_lookup(id2label, idx: int) -> str:
    if isinstance(id2label, dict):
        if idx in id2label:
            return str(id2label[idx])
        if str(idx) in id2label:
            return str(id2label[str(idx)])
    if isinstance(id2label, list) and 0 <= idx < len(id2label):
        return str(id2label[idx])
    return "O"


def build_base_dataset(token_labels_path: Path) -> List[dict]:
    data = []
    for obj in read_jsonl(token_labels_path):
        tokens = obj.get("tokens") or []
        tl = obj.get("token_labels") or []
        if not tokens or not isinstance(tl, list):
            continue

        labels = []
        for x in tl:
            if isinstance(x, dict):
                labels.append(x.get("label") or "O")
            else:
                labels.append("O")

        if len(tokens) != len(labels):
            continue

        data.append({"tokens": tokens, "labels": labels})
    return data


def predict_pseudo_labels(model_dir: Path, token_labels_path: Path, max_len: int = 256) -> List[dict]:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    id2label = model.config.id2label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out = []
    for obj in read_jsonl(token_labels_path):
        sent_id = obj.get("sent_id")
        tokens = obj.get("tokens") or []
        if not tokens:
            continue

        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits  # [1, seq, labels]
            probs = torch.softmax(logits, dim=-1)[0]  # [seq, labels]
            pred_ids = probs.argmax(dim=-1).tolist()
            pred_scores = probs.max(dim=-1).values.tolist()

        word_ids = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_len).word_ids()
        word_best = {}
        for tidx, wid in enumerate(word_ids):
            if wid is None:
                continue
            score = float(pred_scores[tidx])
            pid = int(pred_ids[tidx])
            if wid not in word_best or score > word_best[wid][1]:
                word_best[wid] = (pid, score)

        word_preds = []
        for i in range(len(tokens)):
            if i in word_best:
                pid, score = word_best[i]
                word_preds.append({"label": id2label_lookup(id2label, pid), "confidence": score})
            else:
                word_preds.append({"label": "O", "confidence": 0.0})

        out.append({
            "sent_id": sent_id,
            "tokens": tokens,
            "pred": word_preds,
        })

    return out


def select_high_confidence(pseudo: List[dict], threshold: float = 0.85, min_lex_ratio: float = 0.05) -> List[dict]:
    """
    Отбираем предложения для self-training:
    - средняя уверенность по НЕ-O токенам >= threshold
    - и доля НЕ-O токенов >= min_lex_ratio (чтобы не обучаться на пустоте)
    """
    selected = []
    for ex in pseudo:
        preds = ex["pred"]
        non_o = [p for p in preds if p["label"] != "O"]
        if not non_o:
            continue

        avg_conf = sum(p["confidence"] for p in non_o) / len(non_o)
        ratio = len(non_o) / max(1, len(preds))

        if avg_conf >= threshold and ratio >= min_lex_ratio:
            selected.append(ex)

    return selected


def merge_datasets(base: List[dict], selected: List[dict]) -> List[dict]:
    """
    base: [{"tokens":..., "labels":...}]
    selected: [{"tokens":..., "pred":[{"label":...}, ...]}]
    """
    merged = list(base)
    for ex in selected:
        merged.append({
            "tokens": ex["tokens"],
            "labels": [p["label"] for p in ex["pred"]],
        })
    return merged


def train_token_tagger(
    dataset: List[dict],
    model_name_or_dir: str,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    eval_ratio: float,
    max_len: int,
    seed: int,
) -> dict:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        TrainingArguments,
        Trainer,
    )

    try:
        import evaluate
        metric = evaluate.load("seqeval")
    except Exception:
        metric = None

    label2id = {"O": 0, "LEX": 1}
    id2label = {v: k for k, v in label2id.items()}

    def to_ids(labels: List[str]) -> List[int]:
        out = []
        for x in labels:
            out.append(label2id.get(x, 0))
        return out

    data_id = [{"tokens": d["tokens"], "labels": to_ids(d["labels"])} for d in dataset]

    train_data, eval_data = split_train_eval(data_id, eval_ratio=eval_ratio, seed=seed)
    ds_train = Dataset.from_list(train_data)
    ds_eval = Dataset.from_list(eval_data)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)

    def tokenize_and_align(batch):
        tok = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
        )
        aligned = []
        for i in range(len(batch["tokens"])):
            word_ids = tok.word_ids(batch_index=i)
            prev = None
            lbls = []
            for w in word_ids:
                if w is None:
                    lbls.append(-100)
                elif w != prev:
                    lbls.append(batch["labels"][i][w])
                else:
                    lbls.append(-100)
                prev = w
            aligned.append(lbls)
        tok["labels"] = aligned
        return tok

    ds_train = ds_train.map(tokenize_and_align, batched=True, remove_columns=["tokens", "labels"])
    ds_eval = ds_eval.map(tokenize_and_align, batched=True, remove_columns=["tokens", "labels"])

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_dir,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(p):
        if metric is None:
            return {}
        preds = p.predictions.argmax(-1)
        labels = p.label_ids

        true_preds = []
        true_labels = []
        for pred_seq, lab_seq in zip(preds, labels):
            seq_p = []
            seq_l = []
            for pr, lb in zip(pred_seq, lab_seq):
                if lb == -100:
                    continue
                seq_p.append(id2label[int(pr)])
                seq_l.append(id2label[int(lb)])
            true_preds.append(seq_p)
            true_labels.append(seq_l)

        res = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": float(res.get("overall_precision", 0.0)),
            "recall": float(res.get("overall_recall", 0.0)),
            "f1": float(res.get("overall_f1", 0.0)),
            "accuracy": float(res.get("overall_accuracy", 0.0)),
        }

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        optim="adafactor",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        auto_find_batch_size=True,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1" if metric is not None else None,
        greater_is_better=True if metric is not None else None,
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if metric is not None else None,
    )

    trainer.train()
    metrics = trainer.evaluate()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    return metrics


def main() -> int:
    ensure_dirs()
    cfg = load_config()

    try:
        import torch  # noqa: F401
    except Exception:
        print("[FAIL] torch is not installed. Install torch first (see ml.txt notes).")
        return 1

    weak_tok_path = ROOT / "corpora" / "weak_labels" / "token_level_labels.jsonl"
    weak_tok_path = ROOT / cfg.get("WEAK_TOKEN_LABELS_PATH", str(weak_tok_path))
    if not weak_tok_path.exists():
        print(f"[FAIL] Missing weak labels: {weak_tok_path}")
        print("Run first: python scripts/70_weak_annotate_corpus.py")
        return 1

    base_model_name = cfg.get("TAGGER_MODEL_NAME", "bert-base-multilingual-cased")
    base_model_dir = ROOT / cfg.get("SELFTRAIN_BASE_MODEL_DIR", "models/tagger_lite")

    iters = int(cfg.get("SELFTRAIN_ITERS", 2))
    threshold = float(cfg.get("SELFTRAIN_THRESHOLD", 0.85))
    min_lex_ratio = float(cfg.get("SELFTRAIN_MIN_LEX_RATIO", 0.05))

    epochs = int(cfg.get("TAGGER_EPOCHS", 2))
    batch_size = int(cfg.get("TAGGER_BATCH_SIZE", 8))
    lr = float(cfg.get("TAGGER_LR", 5e-5))
    eval_ratio = float(cfg.get("TAGGER_EVAL_RATIO", 0.1))
    max_len = int(cfg.get("TAGGER_MAX_LEN", 256))
    seed = int(cfg.get("SEED", 42))

    base_dataset = build_base_dataset(weak_tok_path)
    if not base_dataset:
        print("[FAIL] Base dataset is empty.")
        return 1

    history = []

    current_model = str(base_model_dir) if base_model_dir.exists() else base_model_name

    for it in range(1, iters + 1):
        model_out = ROOT / f"models/selftrain_iter_{it}"
        pseudo_path = ROOT / "corpora" / "weak_labels" / f"pseudo_iter_{it}.jsonl"

        pseudo = predict_pseudo_labels(Path(current_model), weak_tok_path, max_len=max_len)
        selected = select_high_confidence(pseudo, threshold=threshold, min_lex_ratio=min_lex_ratio)

        write_jsonl(pseudo_path, selected)

        merged = merge_datasets(base_dataset, selected)

        metrics = train_token_tagger(
            dataset=merged,
            model_name_or_dir=current_model,
            out_dir=model_out,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            eval_ratio=eval_ratio,
            max_len=max_len,
            seed=seed,
        )

        history.append({
            "iter": it,
            "selected_sentences": len(selected),
            "train_size": len(merged),
            "metrics": metrics,
            "model_dir": str(model_out),
            "pseudo_selected_path": str(pseudo_path),
        })

        print(f"[OK] self-train iter {it}: selected={len(selected)} train={len(merged)} metrics={metrics}")

        current_model = str(model_out)

    report_path = ROOT / "reports" / "selftrain_history.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("[OK] Self-training loop complete.")
    print(f"[OK] History saved: {report_path}")
    print(f"[OK] Final model: {current_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
