from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs():
    (ROOT / "data" / "db").mkdir(parents=True, exist_ok=True)


def read_csv_iter(path: Path) -> Tuple[List[str], Iterable[Dict[str, str]]]:
    f = open(path, "r", encoding="utf-8", newline="")
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    return fieldnames, reader


def table_name_from_file(path: Path) -> str:
    name = path.stem
    name = name.replace("_clean", "")
    return name


def sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def create_table(conn: sqlite3.Connection, table: str, cols: List[str]) -> None:
    cols_sql = ", ".join(f"{sql_ident(c)} TEXT" for c in cols)
    conn.execute(f"DROP TABLE IF EXISTS {sql_ident(table)}")
    conn.execute(f"CREATE TABLE {sql_ident(table)} ({cols_sql})")


def create_indexes(conn: sqlite3.Connection) -> None:
    # базовые индексы для скорости
    conn.execute('CREATE INDEX IF NOT EXISTS "idx_entries_entry_id" ON "entries"("entry_id")')
    conn.execute('CREATE INDEX IF NOT EXISTS "idx_entries_headword_norm" ON "entries"("headword_norm")')
    conn.execute('CREATE INDEX IF NOT EXISTS "idx_senses_entry_id" ON "senses"("entry_id")')
    conn.execute('CREATE INDEX IF NOT EXISTS "idx_phrases_entry_id" ON "phrases"("entry_id")')
    conn.execute('CREATE INDEX IF NOT EXISTS "idx_examples_entry_id" ON "examples"("entry_id")')
    conn.execute('CREATE INDEX IF NOT EXISTS "idx_examples_entry_sense" ON "examples"("entry_id","sense_id")')


def bulk_insert(conn: sqlite3.Connection, table: str, cols: List[str], rows: Iterable[Dict[str, str]], batch_size: int = 5000) -> int:
    placeholders = ", ".join(["?"] * len(cols))
    cols_sql = ", ".join(sql_ident(c) for c in cols)
    sql = f"INSERT INTO {sql_ident(table)} ({cols_sql}) VALUES ({placeholders})"

    buf = []
    total = 0

    for r in rows:
        buf.append([r.get(c, "") for c in cols])
        if len(buf) >= batch_size:
            conn.executemany(sql, buf)
            total += len(buf)
            buf.clear()

    if buf:
        conn.executemany(sql, buf)
        total += len(buf)

    return total


def main() -> int:
    ensure_dirs()
    cfg = load_config()

    clean_dir = ROOT / "data" / "processed" / "clean"
    if not clean_dir.exists():
        print("[FAIL] Missing data/processed/clean. Run 45_clean_parsed_csv.py or put clean csv there.")
        return 1

    entries_file = clean_dir / cfg.get("ENTRIES_FILE", "entries_clean.csv")
    senses_file = clean_dir / cfg.get("SENSES_FILE", "senses_clean.csv")
    phrases_file = clean_dir / cfg.get("PHRASES_FILE", "phrases_clean.csv")
    examples_file = clean_dir / cfg.get("EXAMPLES_FILE", "examples_clean.csv")
    abbr_file = clean_dir / cfg.get("ABBR_FILE", "abbr_labels_clean.csv")

    for p in [entries_file, senses_file, phrases_file, examples_file, abbr_file]:
        if not p.exists():
            print(f"[FAIL] Missing clean csv: {p}")
            return 1

    db_path = ROOT / "data" / "db" / "dict_clean.sqlite"

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    files = [entries_file, senses_file, phrases_file, examples_file, abbr_file]

    try:
        with conn:
            for csv_path in files:
                table = table_name_from_file(csv_path)

                cols, rows_iter = read_csv_iter(csv_path)
                if not cols:
                    print(f"[WARN] No columns in {csv_path.name}, skipping.")
                    continue

                create_table(conn, table, cols)

                inserted = bulk_insert(conn, table, cols, rows_iter)
                print(f"[OK] {table}: inserted {inserted} rows")

            create_indexes(conn)
            print("[OK] Indexes created")

        print(f"[OK] SQLite saved: {db_path}")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
