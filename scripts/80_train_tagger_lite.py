from __future__ import annotations

import json
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


def build_dataset(token_labels_path: Path, label2id: Dict[str, int]) -> List[dict]:
    data = []
    for obj in read_jsonl(token_labels_path):
        tokens = obj.get("tokens") or []
        tl = obj.get("token_labels") or []
        if tl and isinstance(tl, list) and isinstance(tl[0], dict):
            labels = [label2id.get((x.get("label") or "O"), label2id["O"]) for x in tl]
        else:
            labels = [label2id["O"]] * len(tokens)

        if len(tokens) != len(labels) or not tokens:
            continue

        data.append({"tokens": tokens, "labels": labels})
    return data


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


def main() -> int:
    ensure_dirs()
    cfg = load_config()

    try:
        import torch  # noqa: F401
    except Exception:
        print("[FAIL] torch is not installed. Install torch first (see ml.txt notes).")
        return 1

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
    except Exception:
        evaluate = None

    corpus_tok_path = ROOT / "corpora" / "weak_labels" / "token_level_labels.jsonl"
    corpus_tok_path = ROOT / cfg.get("WEAK_TOKEN_LABELS_PATH", str(corpus_tok_path))
    if not corpus_tok_path.exists():
        print(f"[FAIL] Missing weak labels: {corpus_tok_path}")
        print("Run first: python scripts/70_weak_annotate_corpus.py")
        return 1

    model_name = cfg.get("TAGGER_MODEL_NAME", "bert-base-multilingual-cased")
    out_dir = ROOT / cfg.get("TAGGER_OUT_DIR", "models/tagger_lite")
    out_dir = Path(out_dir)

    epochs = int(cfg.get("TAGGER_EPOCHS", 2))
    batch_size = int(cfg.get("TAGGER_BATCH_SIZE", 8))
    lr = float(cfg.get("TAGGER_LR", 5e-5))
    eval_ratio = float(cfg.get("TAGGER_EVAL_RATIO", 0.1))
    max_len = int(cfg.get("TAGGER_MAX_LEN", 256))
    seed = int(cfg.get("SEED", 42))

    label2id = {"O": 0, "LEX": 1}
    id2label = {v: k for k, v in label2id.items()}

    raw_data = build_dataset(corpus_tok_path, label2id)
    train_data, eval_data = split_train_eval(raw_data, eval_ratio=eval_ratio, seed=seed)
    if not train_data or not eval_data:
        print("[FAIL] Not enough data to train/evaluate.")
        return 1

    ds_train = Dataset.from_list(train_data)
    ds_eval = Dataset.from_list(eval_data)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(p):
        if evaluate is None:
            return {}
        metric = evaluate.load("seqeval")
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
        out = {
            "precision": float(res.get("overall_precision", 0.0)),
            "recall": float(res.get("overall_recall", 0.0)),
            "f1": float(res.get("overall_f1", 0.0)),
            "accuracy": float(res.get("overall_accuracy", 0.0)),
        }
        return out

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1" if evaluate is not None else None,
        greater_is_better=True if evaluate is not None else None,
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
        compute_metrics=compute_metrics if evaluate is not None else None,
    )

    trainer.train()
    metrics = trainer.evaluate()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    report_path = ROOT / "reports" / "tagger_lite_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[OK] Training complete.")
    print(f"[OK] Model saved: {out_dir}")
    print(f"[OK] Metrics saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
