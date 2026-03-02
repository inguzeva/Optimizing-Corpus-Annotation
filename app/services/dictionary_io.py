import csv
from pathlib import Path
from flask import current_app

from app.services.normalize import normalize_headword


# ========= In-memory cache =========

_CACHE = {
    "entries": {},
    "senses": {},
    "phrases": {},
    "examples": {},
    "abbr": {},
    "loaded": False
}


# ========= Public API =========

def load_dictionary(force=False):
    """
    Загружает clean CSV в память.
    Вызывается один раз при первом обращении.
    """
    if _CACHE["loaded"] and not force:
        return

    clean_dir = Path(current_app.config["CLEAN_DIR"])

    entries_file = clean_dir / current_app.config.get("ENTRIES_FILE", "entries_clean.csv")
    senses_file = clean_dir / current_app.config.get("SENSES_FILE", "senses_clean.csv")
    phrases_file = clean_dir / current_app.config.get("PHRASES_FILE", "phrases_clean.csv")
    examples_file = clean_dir / current_app.config.get("EXAMPLES_FILE", "examples_clean.csv")
    abbr_file = clean_dir / current_app.config.get("ABBR_FILE", "abbr_labels_clean.csv")

    _CACHE["entries"] = _load_entries(entries_file)
    _CACHE["senses"] = _load_grouped(senses_file, "entry_id")
    _CACHE["phrases"] = _load_grouped(phrases_file, "entry_id")
    _CACHE["examples"] = _load_grouped(examples_file, "entry_id")
    _CACHE["abbr"] = _load_abbr(abbr_file)

    _CACHE["loaded"] = True


def get_entries():
    load_dictionary()
    return _CACHE["entries"]


def get_entry(entry_id):
    load_dictionary()
    return _CACHE["entries"].get(str(entry_id))


def get_senses(entry_id):
    load_dictionary()
    return _CACHE["senses"].get(str(entry_id), [])


def get_phrases(entry_id):
    load_dictionary()
    return _CACHE["phrases"].get(str(entry_id), [])


def get_examples(entry_id):
    load_dictionary()
    return _CACHE["examples"].get(str(entry_id), [])


def get_abbr_labels():
    load_dictionary()
    return _CACHE["abbr"]


# ========= Internal loaders =========

def _load_entries(path: Path):
    data = {}

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            entry_id = row["entry_id"]

            row["headword_norm"] = normalize_headword(row.get("headword_norm") or row["headword_raw"])

            data[entry_id] = row

    return data


def _load_grouped(path: Path, key_field: str):
    grouped = {}

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = row[key_field]
            grouped.setdefault(key, []).append(row)

    return grouped


def _load_abbr(path: Path):
    abbr = {}

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            abbr[row["abbr"]] = row

    return abbr


# ========= Utilities =========

def reload_dictionary():
    """
    Принудительно перечитать CSV (после правок в UI)
    """
    _CACHE["loaded"] = False
    load_dictionary(force=True)
