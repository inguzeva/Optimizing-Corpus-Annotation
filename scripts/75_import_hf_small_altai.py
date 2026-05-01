from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
HF_DATASET = "Lil-Graver/small-altai-corpus"
API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET}"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PARQUET_URL = (
    "https://huggingface.co/datasets/"
    f"{HF_DATASET}/resolve/main/data/train-00000-of-00001.parquet"
)
PAGE_SIZE = 100

try:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.services.normalize import normalize_text  # type: ignore
except Exception:
    def normalize_text(text: str) -> str:
        return " ".join((text or "").split()).strip()


def fetch_json(url: str, retries: int = 6, sleep_s: float = 1.0) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < retries:
                time.sleep(sleep_s * attempt)
                continue
            raise
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_s * attempt)
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch JSON: {url}")


def fetch_total_rows() -> int:
    payload = fetch_json(API_URL)
    card = payload.get("cardData", {}) or {}
    info = card.get("dataset_info", {}) or {}
    splits = info.get("splits", []) or []
    for split in splits:
        if split.get("name") == "train":
            return int(split.get("num_examples") or 0)
    raise RuntimeError("Could not determine train split size from HF API.")


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(dst, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def try_load_rows_from_parquet(total_hint: int) -> List[Dict[str, str]] | None:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None

    tmp_path = ROOT / "data" / "raw" / "hf_small_altai_train.parquet"
    download_file(PARQUET_URL, tmp_path)
    table = pq.read_table(tmp_path)
    names = table.column_names
    if "Алтайский" not in names or "Русский" not in names:
        raise RuntimeError(f"Unexpected parquet columns: {names}")

    altai = table.column("Алтайский").to_pylist()
    russian = table.column("Русский").to_pylist()
    rows = [{"Алтайский": a or "", "Русский": r or ""} for a, r in zip(altai, russian)]
    if total_hint and len(rows) != total_hint:
        print(
            f"[WARN] parquet row count differs from API hint: parquet={len(rows)} api={total_hint}"
        )
    return rows


def iter_rows(total: int) -> Iterable[Dict[str, str]]:
    for offset in range(0, total, PAGE_SIZE):
        query = urllib.parse.urlencode(
            {
                "dataset": HF_DATASET,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        payload = fetch_json(f"{ROWS_URL}?{query}")
        for item in payload.get("rows", []):
            row = item.get("row", {}) or {}
            yield row


def load_existing(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dedupe_rows(rows: Iterable[dict]) -> List[dict]:
    seen = set()
    out: List[dict] = []
    for row in rows:
        text = normalize_text(row.get("text", ""))
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(row)
        normalized["text"] = text
        out.append(normalized)
    return out


def build_hf_rows(total: int) -> List[dict]:
    raw_rows = try_load_rows_from_parquet(total)
    if raw_rows is None:
        raw_rows = list(iter_rows(total))

    rows: List[dict] = []
    width = len(str(total))
    for idx, row in enumerate(raw_rows, start=1):
        altai = normalize_text(row.get("Алтайский", ""))
        russian = normalize_text(row.get("Русский", ""))
        if not altai:
            continue
        rows.append(
            {
                "sent_id": f"hf_small_altai_{idx:0{width}d}",
                "text": altai,
                "author": "",
                "title": "small-altai-corpus",
                "source_type": "huggingface_parallel_corpus",
                "source_dataset": HF_DATASET,
                "source_split": "train",
                "translation_ru": russian,
                "license": "CC-BY-4.0",
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_txt(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            text = (row.get("text") or "").strip()
            if text:
                f.write(text + "\n")


def main() -> int:
    total = fetch_total_rows()
    hf_rows = build_hf_rows(total)

    hf_only_path = ROOT / "corpora" / "input" / "small_altai_corpus.jsonl"
    merged_jsonl_path = ROOT / "corpora" / "input" / "corpus.jsonl"
    merged_txt_path = ROOT / "corpora" / "input" / "corpus.txt"

    existing_rows = load_existing(merged_jsonl_path)
    base_rows = [row for row in existing_rows if (row.get("source_dataset") or "") != HF_DATASET]
    merged_rows = dedupe_rows([*base_rows, *hf_rows])

    write_jsonl(hf_only_path, hf_rows)
    write_jsonl(merged_jsonl_path, merged_rows)
    write_txt(merged_txt_path, merged_rows)

    print(f"[OK] Imported HF dataset: {HF_DATASET}")
    print(f"[INFO] hf_rows={len(hf_rows)} total_from_api={total}")
    print(f"[INFO] existing_non_hf_rows={len(base_rows)} merged_rows={len(merged_rows)}")
    print(f"[OK] saved: {hf_only_path}")
    print(f"[OK] saved: {merged_jsonl_path}")
    print(f"[OK] saved: {merged_txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
