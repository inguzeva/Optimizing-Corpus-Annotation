

from __future__ import annotations

import json
from pathlib import Path

import pdfplumber
import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs():
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)


def iter_pages(pdf, start: int | None, end: int | None):
    """
    start/end — 0-based inclusive/exclusive:
    - start: индекс первой страницы
    - end: индекс ПОСЛЕДНЕЙ+1 (как в срезах)
    """
    total = len(pdf.pages)
    s = 0 if start is None else max(0, int(start))
    e = total if end is None else min(total, int(end))
    if e < s:
        e = s
    for i in range(s, e):
        yield i, pdf.pages[i]


def main() -> int:
    ensure_dirs()
    cfg = load_config()

    # PDF путь
    pdf_path = (
        (cfg.get("pdf", {}) or {}).get("path")
        or cfg.get("PDF_PATH")
        or "словарь для модели.pdf"
    )
    pdf_path = ROOT / pdf_path
    if not pdf_path.exists():
        print(f"[FAIL] PDF not found: {pdf_path}")
        return 1

    # диапазон страниц (если задан)
    page_start = (cfg.get("pdf", {}) or {}).get("page_start", None)
    page_end = (cfg.get("pdf", {}) or {}).get("page_end", None)

    # page_start/page_end в config.yaml можно задавать как 1-based (как в обычной нумерации)
    # но внутри скриптов мы работаем 0-based. Если хочешь строго 0-based — просто ставь 0/None.
    # Здесь сделаем мягко: если page_start >= 1, считаем, что это 1-based.
    if page_start is not None:
        page_start = int(page_start)
        if page_start >= 1:
            page_start = page_start - 1

    if page_end is not None:
        page_end = int(page_end)
        # page_end трактуем как 1-based "последняя включительно" если >=1
        if page_end >= 1:
            page_end = page_end  # станет exclusive ниже при -? (см. дальше)
            # Чтобы "включительно" превратить в exclusive в 0-based:
            page_end = page_end  # 1-based inclusive -> 0-based exclusive = page_end
            # Пример: page_end=10 (хочу до 10 страницы включительно) => exclusive index=10
            # Так как page indices 0.., это верно при 1-based интерпретации.

    out_path = ROOT / "data" / "raw" / "words_pages.jsonl"

    with pdfplumber.open(str(pdf_path)) as pdf, open(out_path, "w", encoding="utf-8") as out:
        total_pages = len(pdf.pages)

        # подготавливаем итератор и прогресс
        pages_iter = list(iter_pages(pdf, page_start, page_end))
        iterator = pages_iter
        if tqdm is not None:
            iterator = tqdm(pages_iter, desc="Extract words", unit="page")

        print(f"[INFO] PDF: {pdf_path.name}")
        print(f"[INFO] Total pages in PDF: {total_pages}")
        print(f"[INFO] Extract range: {pages_iter[0][0] if pages_iter else 'n/a'}..{pages_iter[-1][0] if pages_iter else 'n/a'}")
        print(f"[INFO] Output: {out_path}")

        for page_index, page in iterator:
            # extract_words даёт список слов с координатами
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=False,
            ) or []

            # нормализуем структуру (оставим только нужные поля)
            clean_words = []
            for w in words:
                text = (w.get("text") or "").strip()
                if not text:
                    continue
                clean_words.append({
                    "text": text,
                    "x0": float(w.get("x0", 0.0)),
                    "x1": float(w.get("x1", 0.0)),
                    "top": float(w.get("top", 0.0)),
                    "bottom": float(w.get("bottom", 0.0)),
                })

            record = {
                "page_index": int(page_index),
                "width": float(page.width),
                "height": float(page.height),
                "words_count": len(clean_words),
                "words": clean_words,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[OK] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
