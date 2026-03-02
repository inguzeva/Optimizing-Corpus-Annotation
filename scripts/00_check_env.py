

from __future__ import annotations

import sys
from pathlib import Path
import traceback


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_DIRS = [
    "app",
    "scripts",
    "data",
    "requirements",
]

REQUIRED_ROOT_FILES = [
    "config.yaml",
    "словарь для модели.pdf",
]


def ok(msg: str) -> None:
    print(f"[OK]  {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        fail(f"Python {major}.{minor} is too old. Need Python 3.10+")
        return False
    ok(f"Python version: {major}.{minor} ({sys.executable})")
    return True


def check_structure() -> bool:
    good = True

    ok(f"Project root: {ROOT}")

    for d in REQUIRED_DIRS:
        p = ROOT / d
        if not p.exists() or not p.is_dir():
            fail(f"Missing directory: {d}")
            good = False
        else:
            ok(f"Directory exists: {d}")

    for f in REQUIRED_ROOT_FILES:
        p = ROOT / f
        if not p.exists() or not p.is_file():
            fail(f"Missing file in project root: {f}")
            good = False
        else:
            ok(f"File exists: {f}")

    return good


def load_yaml_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}

    try:
        import yaml  # type: ignore
    except Exception:
        warn("PyYAML is not installed. Skipping config.yaml parsing.")
        return {}

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    ok("config.yaml parsed successfully")
    return data


def check_clean_dict_files(cfg: dict) -> bool:
    clean_dir = ROOT / "data" / "processed" / "clean"
    if not clean_dir.exists():
        fail("Missing directory: data/processed/clean")
        return False

    entries_file = cfg.get("ENTRIES_FILE", "entries_clean.csv")
    senses_file = cfg.get("SENSES_FILE", "senses_clean.csv")
    phrases_file = cfg.get("PHRASES_FILE", "phrases_clean.csv")
    examples_file = cfg.get("EXAMPLES_FILE", "examples_clean.csv")
    abbr_file = cfg.get("ABBR_FILE", "abbr_labels_clean.csv")

    required = [entries_file, senses_file, phrases_file, examples_file, abbr_file]

    good = True
    for name in required:
        p = clean_dir / name
        if not p.exists() or not p.is_file():
            fail(f"Missing clean dictionary file: data/processed/clean/{name}")
            good = False
        else:
            ok(f"Clean dictionary file exists: {name}")

    return good


def check_imports() -> None:
    optional_modules = [
        ("flask", "Flask (web)"),
        ("yaml", "PyYAML (config)"),
        ("pdfplumber", "pdfplumber (parsing)"),
        ("fitz", "PyMuPDF (parsing)"),
        ("pandas", "pandas (optional)"),
        ("rapidfuzz", "rapidfuzz (matching)"),
    ]

    for mod, label in optional_modules:
        try:
            __import__(mod)
            ok(f"Import OK: {label} ({mod})")
        except Exception:
            warn(f"Import missing: {label} ({mod})")


def main() -> int:
    print("=== Environment check ===")

    all_good = True

    if not check_python():
        all_good = False

    if not check_structure():
        all_good = False

    cfg = load_yaml_config()

    if not check_clean_dict_files(cfg):
        all_good = False

    check_imports()

    print()
    if all_good:
        ok("Environment looks good. You can run the web app or scripts.")
        print("Next steps (example):")
        print("  - Web:   python run.py")
        print("  - Parse: python scripts/10_extract_words.py (if you re-parse PDF)")
        return 0

    fail("Environment check failed. Fix issues above and re-run.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        fail("Unexpected error during environment check:")
        traceback.print_exc()
        raise SystemExit(2)
