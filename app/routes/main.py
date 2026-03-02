from __future__ import annotations

from pathlib import Path

from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app

from app.models import db, Entry, Sense, Phrase, Example, AbbrLabel, EntryPdfFeature, CorpusSentence, TokenAnnotation, ReviewItem
from app.services.validation import compute_dictionary_stats, validate_dictionary_links, validate_entry_fields
from app.db import import_clean_dictionary, import_weak_labels, import_model_predictions, build_review_queue


main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def index():
    stats = compute_dictionary_stats()

    # DB counts (если БД уже заполнена)
    db_counts = {}
    try:
        db_counts = {
            "entries": Entry.query.count(),
            "senses": Sense.query.count(),
            "phrases": Phrase.query.count(),
            "examples": Example.query.count(),
            "abbr_labels": AbbrLabel.query.count(),
            "entry_pdf_features": EntryPdfFeature.query.count(),
            "corpus_sentences": CorpusSentence.query.count(),
            "token_annotations": TokenAnnotation.query.count(),
            "review_items_open": ReviewItem.query.filter(ReviewItem.status == "open").count(),
        }
    except Exception:
        db_counts = {"error": "db_not_ready"}

    return render_template("index.html", stats=stats, db_counts=db_counts)


@main_bp.get("/health")
def health():
    return {
        "ok": True,
        "debug": bool(current_app.config.get("DEBUG")),
    }


@main_bp.get("/diagnostics")
def diagnostics():
    links = validate_dictionary_links(sample_limit=200)
    missing = validate_entry_fields(sample_limit=200)
    return render_template("diagnostics.html", links=links, missing=missing)


@main_bp.post("/admin/import_dictionary")
def admin_import_dictionary():
    try:
        res = import_clean_dictionary(replace=True, include_dopslovar=True)
        flash(f"Словарь импортирован: {res}", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Ошибка импорта словаря: {e}", "error")
    return redirect(url_for("main.index"))


@main_bp.post("/admin/import_weak_labels")
def admin_import_weak_labels():
    try:
        res = import_weak_labels(replace=True)
        flash(f"Слабая разметка импортирована: {res}", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Ошибка импорта слабой разметки: {e}", "error")
    return redirect(url_for("main.index"))


@main_bp.post("/admin/import_model_predictions")
def admin_import_model_predictions():
    pred_path_raw = (request.form.get("pred_path") or "").strip()
    pred_path = Path(pred_path_raw) if pred_path_raw else None

    try:
        res = import_model_predictions(predictions_path=pred_path, replace=True)
        flash(f"Предсказания модели импортированы: {res}", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Ошибка импорта предсказаний модели: {e}", "error")
    return redirect(url_for("main.index"))


@main_bp.post("/admin/build_review_queue")
def admin_build_review_queue():
    conf_low = float(request.form.get("conf_low", current_app.config.get("CONFIDENCE_LOW", 0.4)))
    conf_mid = float(request.form.get("conf_mid", current_app.config.get("CONFIDENCE_MID", 0.65)))
    conf_pdf = float(request.form.get("conf_pdf", current_app.config.get("CONFIDENCE_PDF_LOW", 0.45)))
    limit_per_type = int(request.form.get("limit", 5000))
    include_conflict = request.form.get("include_conflict", "0") == "1"
    include_ambiguity = request.form.get("include_ambiguity", "0") == "1"
    include_pdf_risky = request.form.get("include_pdf_risky", "0") == "1"

    try:
        res = build_review_queue(
            conf_low=conf_low,
            conf_mid=conf_mid,
            conf_pdf=conf_pdf,
            limit_per_type=limit_per_type,
            replace_open=True,
            include_conflict=include_conflict,
            include_ambiguity=include_ambiguity,
            include_pdf_risky=include_pdf_risky,
        )
        flash(f"Очередь ревью сформирована: {res}", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Ошибка формирования очереди ревью: {e}", "error")

    return redirect(url_for("main.index"))


@main_bp.post("/admin/reset_review_queue")
def admin_reset_review_queue():
    try:
        ReviewItem.query.delete()
        db.session.commit()
        flash("Очередь ревью очищена", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Ошибка очистки очереди ревью: {e}", "error")
    return redirect(url_for("main.index"))
