from __future__ import annotations

from flask import Blueprint, render_template, request, redirect, url_for, flash

from app.models import db, Entry, Sense, Phrase, Example, EntryPdfFeature
from app.services.search import search_entries
from app.services.normalize import normalize_headword


entries_bp = Blueprint("entries", __name__)


@entries_bp.get("/")
def list_entries():
    q = request.args.get("q", "").strip()
    gq = request.args.get("gq", "").strip()

    phrases = request.args.get("phrases", "").strip()
    examples = request.args.get("examples", "").strip()
    min_senses = request.args.get("min_senses", "").strip()

    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    has_phrases = True if phrases == "1" else (False if phrases == "0" else None)
    has_examples = True if examples == "1" else (False if examples == "0" else None)
    min_senses_int = int(min_senses) if min_senses.isdigit() else 0

    data = search_entries(
        query=q,
        gloss_query=gq,
        has_phrases=has_phrases,
        has_examples=has_examples,
        min_senses=min_senses_int,
        page=page,
        per_page=per_page,
    )

    return render_template(
        "entries/list.html",
        data=data,
        q=q,
        gq=gq,
        phrases=phrases,
        examples=examples,
        min_senses=min_senses,
    )


@entries_bp.get("/<entry_id>")
def view_entry(entry_id: str):
    entry = Entry.query.get(entry_id)
    if not entry:
        flash("Словарная статья не найдена", "error")
        return redirect(url_for("entries.list_entries"))

    senses = Sense.query.filter_by(entry_id=entry_id).order_by(Sense.sense_id.asc()).all()
    phrases = Phrase.query.filter_by(entry_id=entry_id).order_by(Phrase.phrase_id.asc()).all()
    examples = Example.query.filter_by(entry_id=entry_id).order_by(Example.sense_id.asc(), Example.ex_id.asc()).all()

    examples_by_sense = {}
    for ex in examples:
        examples_by_sense.setdefault(ex.sense_id, []).append(ex)

    pdf_feature = EntryPdfFeature.query.filter_by(entry_id=entry_id).first()

    return render_template(
        "entries/view.html",
        entry=entry,
        senses=senses,
        phrases=phrases,
        examples_by_sense=examples_by_sense,
        pdf_feature=pdf_feature,
    )


@entries_bp.route("/<entry_id>/edit", methods=["GET", "POST"])
def edit_entry(entry_id: str):
    entry = Entry.query.get(entry_id)
    if not entry:
        flash("Словарная статья не найдена", "error")
        return redirect(url_for("entries.list_entries"))

    if request.method == "POST":
        entry.headword_raw = request.form.get("headword_raw", entry.headword_raw).strip()
        entry.headword_norm = normalize_headword(entry.headword_raw)

        entry.homonym_roman = request.form.get("homonym_roman", entry.homonym_roman).strip()
        entry.pos_primary = request.form.get("pos_primary", entry.pos_primary).strip()
        entry.labels = request.form.get("labels", entry.labels).strip()
        entry.raw_text = request.form.get("raw_text", entry.raw_text)

        try:
            db.session.commit()
            flash("Изменения сохранены", "success")
        except Exception:
            db.session.rollback()
            flash("Ошибка сохранения", "error")

        return redirect(url_for("entries.view_entry", entry_id=entry.entry_id))

    return render_template("entries/edit.html", entry=entry)
