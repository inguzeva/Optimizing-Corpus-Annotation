

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

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
    (ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def group_words_into_lines(words: list[dict], y_tol: float = 2.5) -> list[list[dict]]:
    """
    Группирует слова в строки по координате top.
    """
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = []
    current: list[dict] = []
    current_y = None

    for w in words_sorted:
        y = float(w["top"])
        if current_y is None:
            current_y = y
            current = [w]
            continue

        if abs(y - current_y) <= y_tol:
            current.append(w)
        else:
            lines.append(current)
            current_y = y
            current = [w]

    if current:
        lines.append(current)

    return lines


def join_line_words(line_words: list[dict]) -> str:
    """
    Склеивает слова строки в текст, сортируя по x0.
    """
    ws = sorted(line_words, key=lambda w: w["x0"])
    # Простая склейка пробелами; дефисы/переносы чистятся позже на этапе clean
    return " ".join(w["text"] for w in ws if w.get("text"))


def detect_split_x_auto(words: list[dict], page_width: float) -> float | None:
    """
    Автоопределение границы колонок по разрыву в x0.
    Идея: в двухколоночном PDF обычно есть большой "пустой" gap между колонками.
    """
    xs = sorted([w["x0"] for w in words if "x0" in w])
    if len(xs) < 50:
        return None

    # вычисляем разрывы между соседними x0
    gaps = [(xs[i + 1] - xs[i], xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
    gaps.sort(key=lambda t: t[0], reverse=True)

    # ищем самый большой gap около центра страницы
    center_min = page_width * 0.35
    center_max = page_width * 0.65

    for gap, a, b in gaps[:50]:
        mid = (a + b) / 2
        if center_min <= mid <= center_max and gap >= 15:
            return mid

    # fallback: медиана x0 (хуже, но лучше чем ничего)
    return float(median(xs))


def classify_column(word: dict, split_x: float) -> str:
    return "left" if float(word["x0"]) < split_x else "right"


def main() -> int:
    ensure_dirs()
    cfg = load_config()

    in_path = ROOT / "data" / "raw" / "words_pages.jsonl"
    if not in_path.exists():
        print(f"[FAIL] Missing input: {in_path}")
        print("Run first: python scripts/10_extract_words.py")
        return 1

    out_path = ROOT / "data" / "interim" / "lines_reflowed.jsonl"

    layout_cfg = cfg.get("layout", {}) or {}
    columns = int(layout_cfg.get("columns", 2) or 2)
    split_mode = layout_cfg.get("column_split_mode", "auto_gap")
    fixed_split_x = layout_cfg.get("fixed_split_x", None)
    y_tol = float(layout_cfg.get("line_y_tolerance", 2.5) or 2.5)

    records = list(read_jsonl(in_path))
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Reflow columns", unit="page")

    with open(out_path, "w", encoding="utf-8") as out:
        for rec in iterator:
            page_index = rec["page_index"]
            width = float(rec.get("width", 0.0))
            height = float(rec.get("height", 0.0))
            words = rec.get("words", []) or []

            columns_mode = "single"
            split_x = None

            if columns == 1:
                columns_mode = "single"
            else:
                if split_mode == "fixed_x" and fixed_split_x is not None:
                    split_x = float(fixed_split_x)
                    columns_mode = "fixed_x"
                else:
                    split_x = detect_split_x_auto(words, width)
                    columns_mode = "auto_gap"

            # если split_x не получился — считаем одноколоночным
            if split_x is None:
                columns_mode = "single"

            # группируем в строки
            if columns_mode == "single":
                lines_words = group_words_into_lines(words, y_tol=y_tol)
                lines = [join_line_words(lw) for lw in lines_words if lw]
                out_rec = {
                    "page_index": int(page_index),
                    "width": width,
                    "height": height,
                    "split_x": None,
                    "columns_mode": "single",
                    "lines_left": lines,
                    "lines_right": [],
                    "lines": lines,
                }
                out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                continue

            # двухколоночный случай:
            left_words = [w for w in words if classify_column(w, split_x) == "left"]
            right_words = [w for w in words if classify_column(w, split_x) == "right"]

            left_lines_words = group_words_into_lines(left_words, y_tol=y_tol)
            right_lines_words = group_words_into_lines(right_words, y_tol=y_tol)

            left_lines = [join_line_words(lw) for lw in left_lines_words if lw]
            right_lines = [join_line_words(lw) for lw in right_lines_words if lw]

            out_rec = {
                "page_index": int(page_index),
                "width": width,
                "height": height,
                "split_x": round(float(split_x), 2),
                "columns_mode": columns_mode,
                "lines_left": left_lines,
                "lines_right": right_lines,
                "lines": left_lines + right_lines,
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"[OK] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
