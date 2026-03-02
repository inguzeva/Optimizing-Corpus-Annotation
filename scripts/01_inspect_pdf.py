

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import pdfplumber
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs():
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)


def guess_two_columns(words: list[dict]) -> dict:
    """
    Признак двух колонок: распределение x0 имеет 2 выраженных "облака"
    (для словарей часто так и есть).

    Мы делаем простую эвристику:
    - берём x0 каждого слова
    - считаем медиану и смотрим долю слов слева/справа
    - смотрим есть ли "пустой разрыв" в районе центра страницы
    """
    if not words:
        return {"two_columns": False, "reason": "no_words"}

    xs = sorted([w["x0"] for w in words if "x0" in w])
    if not xs:
        return {"two_columns": False, "reason": "no_x_coords"}

    med = median(xs)

    left = [x for x in xs if x < med]
    right = [x for x in xs if x >= med]
    left_ratio = len(left) / len(xs)
    right_ratio = len(right) / len(xs)

    # второй признак: есть ли заметная "дырка" в центре (межколоночный пробел)
    # оценим по максимальному разрыву между соседними x
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    max_gap = max(gaps) if gaps else 0

    # эвристика
    # - если есть приличная доля и слева и справа (не 95/5)
    # - и max_gap заметный (условно > 20)
    two_cols = (0.2 < left_ratio < 0.8) and (max_gap > 20)

    return {
        "two_columns": bool(two_cols),
        "left_ratio": round(left_ratio, 3),
        "right_ratio": round(right_ratio, 3),
        "median_x0": round(float(med), 2),
        "max_gap": round(float(max_gap), 2),
    }


def main() -> int:
    ensure_dirs()
    cfg = load_config()

    pdf_path = cfg.get("pdf", {}).get("path") or cfg.get("PDF_PATH") or "словарь для модели.pdf"
    pdf_path = ROOT / pdf_path

    if not pdf_path.exists():
        print(f"[FAIL] PDF not found: {pdf_path}")
        return 1

    sample_n = int(cfg.get("inspect", {}).get("sample_pages", 8) or 8)

    meta = {
        "pdf_path": str(pdf_path),
        "pages_total": None,
        "sample_pages": [],
        "text_layer_ok_pages": 0,
        "two_columns_pages": 0,
        "notes": [],
    }

    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        meta["pages_total"] = total

        # список страниц для сэмпла: первые + середина + ближе к концу
        candidates = set()
        for i in range(min(sample_n // 2, total)):
            candidates.add(i)
        if total > 10:
            candidates.add(total // 2)
            candidates.add(max(0, total // 2 - 1))
            candidates.add(min(total - 1, total // 2 + 1))
            candidates.add(total - 2)
            candidates.add(total - 1)

        # ограничим до sample_n
        page_ids = sorted(list(candidates))[:sample_n]

        for pno in page_ids:
            page = pdf.pages[pno]

            # words с координатами — важнее чем extract_text()
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=False,
            )

            extracted_text = page.extract_text() or ""
            text_len = len(extracted_text.strip())

            text_layer_ok = text_len > 30  # грубый признак
            if text_layer_ok:
                meta["text_layer_ok_pages"] += 1

            cols_info = guess_two_columns(words)
            if cols_info.get("two_columns"):
                meta["two_columns_pages"] += 1

            meta["sample_pages"].append({
                "page_index": pno,
                "text_len": text_len,
                "text_layer_ok": bool(text_layer_ok),
                "words_count": len(words),
                "columns_guess": cols_info,
                "text_preview": extracted_text.strip()[:300],
            })

    # вывод в файл
    out_path = ROOT / "data" / "raw" / "pages_meta.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] PDF inspected. Total pages: {meta['pages_total']}")
    print(f"[OK] Text layer OK on {meta['text_layer_ok_pages']} sampled pages.")
    print(f"[OK] Two columns guessed on {meta['two_columns_pages']} sampled pages.")
    print(f"[OK] Saved report: {out_path}")

    if meta["text_layer_ok_pages"] == 0:
        print("[WARN] Looks like there is NO text layer. OCR may be required.")
    if meta["two_columns_pages"] == 0:
        print("[WARN] Two columns not detected in sample. Reflow logic may need tuning.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
