from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, render_template, request, send_file, flash, redirect, url_for

from app.models import db, Entry, Sense, Phrase, Example, AbbrLabel, EntryPdfFeature, GoldSentence, GoldToken, TokenAnnotation, CorpusSentence


export_bp = Blueprint("export", __name__)


def _exports_dir() -> Path:
    base_dir = Path(current_app.config.get("BASE_DIR", Path(__file__).resolve().parents[2]))
    out_dir = base_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


@export_bp.get("/")
def export_index():
    out_dir = _exports_dir()
    files = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    files_view = [{"name": p.name, "size": p.stat().st_size} for p in files[:50]]
    return render_template("export/index.html", files=files_view)


@export_bp.get("/download/<name>")
def download_export(name: str):
    out_dir = _exports_dir()
    path = out_dir / name
    if not path.exists():
        flash("Файл не найден", "error")
        return redirect(url_for("export.export_index"))
    return send_file(path, as_attachment=True)


@export_bp.post("/dictionary/csv")
def export_dictionary_csv():
    out_dir = _exports_dir()
    stamp = _timestamp()

    entries_path = out_dir / f"entries_export_{stamp}.csv"
    senses_path = out_dir / f"senses_export_{stamp}.csv"
    phrases_path = out_dir / f"phrases_export_{stamp}.csv"
    examples_path = out_dir / f"examples_export_{stamp}.csv"
    abbr_path = out_dir / f"abbr_labels_export_{stamp}.csv"
    pdf_features_path = out_dir / f"entry_pdf_features_export_{stamp}.csv"

    _write_entries(entries_path)
    _write_senses(senses_path)
    _write_phrases(phrases_path)
    _write_examples(examples_path)
    _write_abbr(abbr_path)
    _write_pdf_features(pdf_features_path)

    flash("Словарь экспортирован (CSV)", "success")
    return redirect(url_for("export.export_index"))


@export_bp.post("/gold/jsonl")
def export_gold_jsonl():
    out_dir = _exports_dir()
    stamp = _timestamp()
    path = out_dir / f"gold_{stamp}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        gold = GoldSentence.query.order_by(GoldSentence.id.asc()).all()
        for gs in gold:
            tokens = GoldToken.query.filter_by(gold_sentence_id=gs.id).order_by(GoldToken.token_idx.asc()).all()
            row = {
                "sent_id": gs.sent_id,
                "text": gs.text,
                "tokens": [{"token": t.token, "label": t.label, "token_idx": t.token_idx} for t in tokens],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    flash("Gold-набор экспортирован (JSONL)", "success")
    return redirect(url_for("export.export_index"))


@export_bp.post("/corpus/annotations/jsonl")
def export_corpus_annotations_jsonl():
    out_dir = _exports_dir()
    stamp = _timestamp()
    path = out_dir / f"corpus_annotations_{stamp}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        sents = CorpusSentence.query.order_by(CorpusSentence.sent_id.asc()).all()
        for s in sents:
            toks = TokenAnnotation.query.filter_by(sent_id=s.sent_id).order_by(TokenAnnotation.token_idx.asc()).all()
            row = {
                "sent_id": s.sent_id,
                "text": s.text,
                "tokens": [
                    {
                        "token_idx": t.token_idx,
                        "token": t.token,
                        "weak_label": t.weak_label,
                        "weak_confidence": t.weak_confidence,
                        "weak_source": t.weak_source,
                        "text_confidence": t.text_confidence,
                        "pdf_confidence": t.pdf_confidence,
                        "multimodal_confidence": t.multimodal_confidence,
                        "model_label": t.model_label,
                        "model_confidence": t.model_confidence,
                        "final_label": t.final_label,
                        "final_source": t.final_source,
                        "is_confirmed": t.is_confirmed,
                    }
                    for t in toks
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    flash("Аннотации корпуса экспортированы (JSONL)", "success")
    return redirect(url_for("export.export_index"))


def _write_entries(path: Path) -> None:
    rows = Entry.query.order_by(Entry.entry_id.asc()).all()
    fieldnames = [
        "entry_id",
        "headword_raw",
        "headword_norm",
        "homonym_roman",
        "pos_primary",
        "labels",
        "has_phrases",
        "raw_text",
        "page_start",
        "page_end",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())


def _write_senses(path: Path) -> None:
    rows = Sense.query.order_by(Sense.entry_id.asc(), Sense.sense_id.asc()).all()
    fieldnames = [
        "entry_id",
        "sense_id",
        "sense_group",
        "pos",
        "gloss_ru",
        "gloss_ru_alt",
        "labels",
        "refs",
        "confidence_parse",
        "raw_block",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())


def _write_phrases(path: Path) -> None:
    rows = Phrase.query.order_by(Phrase.entry_id.asc(), Phrase.phrase_id.asc()).all()
    fieldnames = [
        "entry_id",
        "phrase_id",
        "phrase_alt_raw",
        "phrase_alt_norm",
        "phrase_ru",
        "labels",
        "raw_block",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())


def _write_examples(path: Path) -> None:
    rows = Example.query.order_by(Example.entry_id.asc(), Example.sense_id.asc(), Example.ex_id.asc()).all()
    fieldnames = [
        "entry_id",
        "sense_id",
        "ex_id",
        "example_alt_raw",
        "example_alt_norm",
        "example_ru",
        "source_note",
        "raw_line",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())


def _write_abbr(path: Path) -> None:
    rows = AbbrLabel.query.order_by(AbbrLabel.abbr.asc()).all()
    fieldnames = ["abbr", "full", "type"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())


def _write_pdf_features(path: Path) -> None:
    rows = EntryPdfFeature.query.order_by(EntryPdfFeature.entry_id.asc()).all()
    fieldnames = [
        "entry_id",
        "page_start_int",
        "page_end_int",
        "page_span",
        "has_page_info",
        "raw_text_chars",
        "raw_text_lines",
        "has_phrase_marker",
        "parse_confidence_mean",
        "parse_confidence_max",
        "pdf_quality_score",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())
