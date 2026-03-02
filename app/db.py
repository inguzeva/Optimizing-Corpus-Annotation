from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import current_app
from sqlalchemy import func, text

from app.models import (
    db,
    Entry,
    Sense,
    Phrase,
    Example,
    AbbrLabel,
    EntryPdfFeature,
    CorpusSentence,
    TokenAnnotation,
    ReviewItem,
    GoldSentence,
)
from app.services.dopslovar import parse_dopslovar
from app.services.normalize import normalize_for_match


def _clean_path(p: str) -> Path:
    return Path(p).resolve()


def init_sqlalchemy(app) -> None:
    """
    Подключение SQLAlchemy к Flask-приложению.
    Использует app.config["SQLALCHEMY_DATABASE_URI"] если задано,
    иначе создаёт sqlite в data/db/app.sqlite.
    """
    base_dir = Path(app.config.get("BASE_DIR", Path(__file__).resolve().parents[1]))

    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        db_path = base_dir / "data" / "db" / "app.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"

    app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)

    db.init_app(app)

    with app.app_context():
        db.create_all()
        _ensure_sqlite_columns()


def reset_database() -> None:
    db.drop_all()
    db.create_all()


def _ensure_sqlite_columns() -> None:
    """
    Лёгкая миграция для SQLite: добавляем новые колонки, если БД уже была создана раньше.
    """
    conn = db.engine.connect()
    try:
        info = conn.execute(text("PRAGMA table_info(token_annotations)")).fetchall()
        existing = {row[1] for row in info}
        additions = [
            ("weak_source", "TEXT NOT NULL DEFAULT 'none'"),
            ("text_confidence", "REAL NOT NULL DEFAULT 0.0"),
            ("pdf_confidence", "REAL NOT NULL DEFAULT 0.0"),
            ("multimodal_confidence", "REAL NOT NULL DEFAULT 0.0"),
        ]
        for col_name, col_type in additions:
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE token_annotations ADD COLUMN {col_name} {col_type}"))
        conn.commit()
    finally:
        conn.close()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def import_clean_dictionary(
    clean_dir: Optional[Path] = None,
    replace: bool = True,
    include_dopslovar: bool = True,
) -> dict:
    """
    Импортирует clean CSV словаря в SQLite.

    replace=True:
      - очищает таблицы entries/senses/phrases/examples/abbr_labels
      - потом импортирует заново
    """
    if clean_dir is None:
        clean_dir = Path(current_app.config["CLEAN_DIR"])

    entries_file = clean_dir / current_app.config.get("ENTRIES_FILE", "entries_clean.csv")
    senses_file = clean_dir / current_app.config.get("SENSES_FILE", "senses_clean.csv")
    phrases_file = clean_dir / current_app.config.get("PHRASES_FILE", "phrases_clean.csv")
    examples_file = clean_dir / current_app.config.get("EXAMPLES_FILE", "examples_clean.csv")
    abbr_file = clean_dir / current_app.config.get("ABBR_FILE", "abbr_labels_clean.csv")

    for p in [entries_file, senses_file, phrases_file, examples_file, abbr_file]:
        if not p.exists():
            raise FileNotFoundError(f"Missing clean CSV: {p}")

    if replace:
        db.session.query(EntryPdfFeature).delete()
        db.session.query(Example).delete()
        db.session.query(Phrase).delete()
        db.session.query(Sense).delete()
        db.session.query(Entry).delete()
        db.session.query(AbbrLabel).delete()
        db.session.commit()

    entries_rows = _read_csv_rows(entries_file)
    senses_rows = _read_csv_rows(senses_file)
    phrases_rows = _read_csv_rows(phrases_file)
    examples_rows = _read_csv_rows(examples_file)
    abbr_rows = _read_csv_rows(abbr_file)

    entries_count = 0
    for r in entries_rows:
        entry = Entry(
            entry_id=(r.get("entry_id") or "").strip(),
            headword_raw=r.get("headword_raw") or "",
            headword_norm=(r.get("headword_norm") or normalize_for_match(r.get("headword_raw") or "")),
            homonym_roman=r.get("homonym_roman") or "",
            pos_primary=r.get("pos_primary") or "",
            labels=r.get("labels") or "",
            has_phrases=int(r.get("has_phrases") or 0),
            raw_text=r.get("raw_text") or "",
            page_start=r.get("page_start") or "",
            page_end=r.get("page_end") or "",
        )
        if not entry.entry_id:
            continue
        db.session.add(entry)
        entries_count += 1

    db.session.commit()

    senses_count = 0
    for r in senses_rows:
        eid = (r.get("entry_id") or "").strip()
        if not eid:
            continue
        sense = Sense(
            entry_id=eid,
            sense_id=r.get("sense_id") or "1",
            sense_group=r.get("sense_group") or "none",
            pos=r.get("pos") or "",
            gloss_ru=r.get("gloss_ru") or "",
            gloss_ru_alt=r.get("gloss_ru_alt") or "",
            labels=r.get("labels") or "",
            refs=r.get("refs") or "",
            confidence_parse=r.get("confidence_parse") or "",
            raw_block=r.get("raw_block") or "",
        )
        db.session.add(sense)
        senses_count += 1

    phrases_count = 0
    for r in phrases_rows:
        eid = (r.get("entry_id") or "").strip()
        if not eid:
            continue
        phrase = Phrase(
            entry_id=eid,
            phrase_id=r.get("phrase_id") or "1",
            phrase_alt_raw=r.get("phrase_alt_raw") or "",
            phrase_alt_norm=(r.get("phrase_alt_norm") or normalize_for_match(r.get("phrase_alt_raw") or "")),
            phrase_ru=r.get("phrase_ru") or "",
            labels=r.get("labels") or "",
            raw_block=r.get("raw_block") or "",
        )
        db.session.add(phrase)
        phrases_count += 1

    db.session.flush()
    phrase_entry_ids = {
        eid for (eid,) in db.session.query(Phrase.entry_id).distinct().all()
    }
    if phrase_entry_ids:
        (
            db.session.query(Entry)
            .filter(Entry.entry_id.in_(phrase_entry_ids))
            .update({Entry.has_phrases: 1}, synchronize_session=False)
        )

    examples_count = 0
    for r in examples_rows:
        eid = (r.get("entry_id") or "").strip()
        if not eid:
            continue
        ex = Example(
            entry_id=eid,
            sense_id=r.get("sense_id") or "1",
            ex_id=r.get("ex_id") or "1",
            example_alt_raw=r.get("example_alt_raw") or "",
            example_alt_norm=(r.get("example_alt_norm") or normalize_for_match(r.get("example_alt_raw") or "")),
            example_ru=r.get("example_ru") or "",
            source_note=r.get("source_note") or "",
            raw_line=r.get("raw_line") or "",
        )
        db.session.add(ex)
        examples_count += 1

    abbr_count = 0
    for r in abbr_rows:
        abbr = (r.get("abbr") or "").strip()
        if not abbr:
            continue
        obj = AbbrLabel(
            abbr=abbr,
            full=r.get("full") or "",
            type=r.get("type") or "label",
        )
        db.session.merge(obj)
        abbr_count += 1

    db.session.commit()

    result = {
        "entries": entries_count,
        "senses": senses_count,
        "phrases": phrases_count,
        "examples": examples_count,
        "abbr_labels": abbr_count,
    }

    if include_dopslovar:
        base_dir = Path(current_app.config.get("BASE_DIR", Path(__file__).resolve().parents[1]))
        dop_path_cfg = current_app.config.get("DOPSLOVAR_PATH", "dopslovar.txt")
        dop_path = Path(dop_path_cfg)
        if not dop_path.is_absolute():
            dop_path = base_dir / dop_path

        dop_stats = _import_dopslovar_entries(dop_path)
        result.update(dop_stats)

    pdf_stats = _rebuild_entry_pdf_features()
    result.update(pdf_stats)

    return result


def _import_dopslovar_entries(path: Path) -> dict:
    pairs = parse_dopslovar(path)
    if not pairs:
        return {
            "dopslovar_path": str(path),
            "dop_pairs": 0,
            "dop_entries_added": 0,
            "dop_senses_added": 0,
            "dop_merged_into_existing": 0,
        }

    existing = db.session.query(Entry.entry_id, Entry.headword_norm).all()
    norm_to_entry_id: Dict[str, str] = {}
    for eid, norm in existing:
        if norm and norm not in norm_to_entry_id:
            norm_to_entry_id[norm] = eid

    sense_counts: Dict[str, int] = {
        eid: int(cnt)
        for eid, cnt in (
            db.session.query(Sense.entry_id, func.count(Sense.id))
            .group_by(Sense.entry_id)
            .all()
        )
    }

    seen_entry_ids = {eid for eid, _ in existing}
    serial = 1

    entries_added = 0
    senses_added = 0
    merged_into_existing = 0

    for row in pairs:
        hw = (row.get("headword") or "").strip()
        gloss = (row.get("gloss_ru") or "").strip()
        if not hw or not gloss:
            continue

        hw_norm = normalize_for_match(hw)
        entry_id = norm_to_entry_id.get(hw_norm)

        if not entry_id:
            while True:
                candidate = f"dop_{serial}"
                serial += 1
                if candidate not in seen_entry_ids:
                    entry_id = candidate
                    break

            seen_entry_ids.add(entry_id)
            norm_to_entry_id[hw_norm] = entry_id

            db.session.add(
                Entry(
                    entry_id=entry_id,
                    headword_raw=hw,
                    headword_norm=hw_norm,
                    homonym_roman="",
                    pos_primary="",
                    labels="dopslovar",
                    has_phrases=0,
                    raw_text=f"{hw}\n{gloss}",
                    page_start="",
                    page_end="",
                )
            )
            sense_counts.setdefault(entry_id, 0)
            entries_added += 1
        else:
            merged_into_existing += 1

        sense_counts[entry_id] = int(sense_counts.get(entry_id, 0)) + 1
        db.session.add(
            Sense(
                entry_id=entry_id,
                sense_id=str(sense_counts[entry_id]),
                sense_group="none",
                pos="",
                gloss_ru=gloss,
                gloss_ru_alt="",
                labels="dopslovar",
                refs="",
                confidence_parse="1.0",
                raw_block=f"{hw} — {gloss}",
            )
        )
        senses_added += 1

    db.session.commit()

    return {
        "dopslovar_path": str(path),
        "dop_pairs": len(pairs),
        "dop_entries_added": entries_added,
        "dop_senses_added": senses_added,
        "dop_merged_into_existing": merged_into_existing,
    }


def _safe_int(value: str) -> Optional[int]:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _safe_float(value: str) -> Optional[float]:
    s = str(value or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return float(x)


def _rebuild_entry_pdf_features() -> dict:
    db.session.query(EntryPdfFeature).delete()
    db.session.flush()

    senses = Sense.query.with_entities(Sense.entry_id, Sense.confidence_parse).all()
    conf_by_entry: Dict[str, List[float]] = {}
    for entry_id, conf_raw in senses:
        val = _safe_float(conf_raw)
        if val is None:
            continue
        conf_by_entry.setdefault(entry_id, []).append(_clamp01(val))

    rows = Entry.query.all()
    created = 0
    for e in rows:
        p1 = _safe_int(e.page_start)
        p2 = _safe_int(e.page_end)
        has_page_info = 1 if (p1 is not None or p2 is not None) else 0
        page_start_int = int(p1 if p1 is not None else (p2 if p2 is not None else 0))
        page_end_int = int(p2 if p2 is not None else (p1 if p1 is not None else 0))
        page_span = abs(page_end_int - page_start_int) + 1 if has_page_info else 0

        raw_text = e.raw_text or ""
        raw_text_chars = len(raw_text)
        raw_text_lines = len([ln for ln in raw_text.splitlines() if ln.strip()])
        has_phrase_marker = 1 if "♦" in raw_text else 0

        confs = conf_by_entry.get(e.entry_id, [])
        conf_mean = (sum(confs) / len(confs)) if confs else 0.0
        conf_max = max(confs) if confs else 0.0

        # Простая мультимодальная "качественная" оценка на основе PDF/layout-сигналов.
        quality = 0.0
        quality += 0.30 if has_page_info else 0.0
        quality += 0.20 if raw_text_chars >= 40 else 0.05 if raw_text_chars > 0 else 0.0
        quality += 0.15 if raw_text_lines >= 2 else 0.05 if raw_text_lines == 1 else 0.0
        quality += 0.10 if has_phrase_marker else 0.0
        quality += 0.25 * _clamp01(conf_mean)
        quality = _clamp01(quality)

        db.session.add(
            EntryPdfFeature(
                entry_id=e.entry_id,
                page_start_int=page_start_int,
                page_end_int=page_end_int,
                page_span=page_span,
                has_page_info=has_page_info,
                raw_text_chars=raw_text_chars,
                raw_text_lines=raw_text_lines,
                has_phrase_marker=has_phrase_marker,
                parse_confidence_mean=round(conf_mean, 6),
                parse_confidence_max=round(conf_max, 6),
                pdf_quality_score=round(quality, 6),
            )
        )
        created += 1

    db.session.commit()
    return {"entry_pdf_features": int(created)}


def import_weak_labels(
    token_labels_path: Optional[Path] = None,
    sentence_hits_path: Optional[Path] = None,
    replace: bool = True,
) -> dict:
    """
    Импортирует результаты 70_weak_annotate_corpus.py в таблицы:
    - corpus_sentences
    - token_annotations

    token_labels_path: corpora/weak_labels/token_level_labels.jsonl
    sentence_hits_path: corpora/weak_labels/sentence_level_lexhits.jsonl
    """
    base_dir = Path(current_app.config.get("BASE_DIR", Path(__file__).resolve().parents[1]))

    if token_labels_path is None:
        token_labels_path = base_dir / "corpora" / "weak_labels" / "token_level_labels.jsonl"
    if sentence_hits_path is None:
        sentence_hits_path = base_dir / "corpora" / "weak_labels" / "sentence_level_lexhits.jsonl"

    if not token_labels_path.exists():
        raise FileNotFoundError(token_labels_path)
    if not sentence_hits_path.exists():
        raise FileNotFoundError(sentence_hits_path)

    if replace:
        db.session.query(TokenAnnotation).delete()
        db.session.query(ReviewItem).delete()
        db.session.query(CorpusSentence).delete()
        db.session.commit()

    sent_map: Dict[str, dict] = {}
    with open(sentence_hits_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sid = str(obj.get("sent_id"))
            sent_map[sid] = obj

    sent_count = 0
    tok_count = 0

    with open(token_labels_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sid = str(obj.get("sent_id"))
            tokens = obj.get("tokens") or []
            token_labels = obj.get("token_labels") or []
            text = obj.get("text") or ""

            sh = sent_map.get(sid, {})
            sent = CorpusSentence(
                sent_id=sid,
                text=text,
                text_norm=normalize_for_match(text),
                covered=int(sh.get("covered") or 0),
                hits_count=int(sh.get("hits_count") or 0),
                phrase_hits_count=int(sh.get("phrase_hits_count") or 0),
            )
            db.session.add(sent)
            sent_count += 1

            for i, tok in enumerate(tokens):
                row = token_labels[i] if i < len(token_labels) and isinstance(token_labels[i], dict) else {}
                mm_conf = float(
                    row.get("multimodal_confidence")
                    or row.get("confidence")
                    or 0.0
                )
                text_conf = float(row.get("text_confidence") or 0.0)
                pdf_conf = float(row.get("pdf_confidence") or 0.0)
                ta = TokenAnnotation(
                    sent_id=sid,
                    token_idx=i,
                    token=str(tok),
                    weak_label=row.get("label") or "O",
                    weak_entry_id=row.get("entry_id") or "",
                    weak_pos=row.get("pos") or "",
                    weak_confidence=mm_conf,
                    weak_source=(row.get("source") or "none"),
                    text_confidence=text_conf,
                    pdf_confidence=pdf_conf,
                    multimodal_confidence=mm_conf,
                    model_label="O",
                    model_confidence=0.0,
                    final_label="",
                    final_source="",
                    is_confirmed=0,
                )
                db.session.add(ta)
                tok_count += 1

    db.session.commit()
    return {"sentences": sent_count, "tokens": tok_count}


def import_model_predictions(predictions_path: Optional[Path] = None, replace: bool = True) -> dict:
    base_dir = Path(current_app.config.get("BASE_DIR", Path(__file__).resolve().parents[1]))

    if predictions_path is None:
        pred_cfg = current_app.config.get("MODEL_PREDICTIONS_PATH", "corpora/weak_labels/model_predictions.jsonl")
        predictions_path = Path(pred_cfg)

    if not predictions_path.is_absolute():
        predictions_path = base_dir / predictions_path

    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    if replace:
        (
            db.session.query(TokenAnnotation)
            .update(
                {
                    TokenAnnotation.model_label: "O",
                    TokenAnnotation.model_confidence: 0.0,
                },
                synchronize_session=False,
            )
        )
        db.session.commit()

    updated = 0
    missing = 0

    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sent_id = str(obj.get("sent_id", "")).strip()
            if not sent_id:
                continue

            for token_idx, label, conf in _iter_pred_rows(obj):
                ta = TokenAnnotation.query.filter_by(sent_id=sent_id, token_idx=int(token_idx)).first()
                if not ta:
                    missing += 1
                    continue
                ta.model_label = str(label or "O")
                ta.model_confidence = float(conf or 0.0)
                updated += 1

    db.session.commit()
    return {
        "predictions_path": str(predictions_path),
        "updated_tokens": updated,
        "missing_tokens": missing,
    }


def _iter_pred_rows(obj: dict) -> List[Tuple[int, str, float]]:
    out: List[Tuple[int, str, float]] = []

    pred = obj.get("pred")
    if isinstance(pred, list):
        for i, row in enumerate(pred):
            if isinstance(row, dict):
                out.append((i, row.get("label") or "O", float(row.get("confidence") or 0.0)))
        return out

    token_labels = obj.get("token_labels")
    if isinstance(token_labels, list):
        for i, row in enumerate(token_labels):
            if isinstance(row, dict):
                out.append((i, row.get("label") or "O", float(row.get("confidence") or 0.0)))
            else:
                out.append((i, "O", 0.0))
        return out

    tokens = obj.get("tokens")
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], dict):
        for row in tokens:
            idx = int(row.get("token_idx", 0))
            out.append((idx, row.get("label") or row.get("model_label") or "O", float(row.get("confidence") or row.get("model_confidence") or 0.0)))
        return out

    labels = obj.get("labels")
    confidences = obj.get("confidences") or []
    if isinstance(labels, list):
        for i, lbl in enumerate(labels):
            conf = confidences[i] if i < len(confidences) else 0.0
            out.append((i, str(lbl or "O"), float(conf or 0.0)))
        return out

    return out


def build_review_queue(
    conf_low: float = 0.4,
    conf_mid: float = 0.65,
    conf_pdf: float = 0.45,
    limit_per_type: int = 5000,
    replace_open: bool = True,
    include_conflict: bool = True,
    include_ambiguity: bool = True,
    include_pdf_risky: bool = True,
) -> dict:
    """
    Создаёт очередь ReviewItem из token_annotations:
    - low_confidence: weak_confidence < conf_low и weak_label != 'O'
    - conflict: weak_label != model_label, если оба != O
    - ambiguity: средняя зона уверенности [conf_low, conf_mid)
    - pdf_risky: слабая pdf-опора в multimodal-режиме
    """
    if replace_open:
        db.session.query(ReviewItem).filter(ReviewItem.status == "open").delete()
        db.session.commit()

    gold_sent_ids = db.session.query(GoldSentence.sent_id).distinct()

    q = (
        db.session.query(TokenAnnotation)
        .filter(~TokenAnnotation.sent_id.in_(gold_sent_ids))
        .filter(TokenAnnotation.weak_label != "O")
        .filter(
            func.coalesce(TokenAnnotation.multimodal_confidence, TokenAnnotation.weak_confidence, 0.0) < float(conf_low)
        )
        .limit(int(limit_per_type))
        .all()
    )

    created_low = 0
    created_conflict = 0
    created_ambiguity = 0
    created_pdf_risky = 0

    for t in q:
        conf_eff = float(t.multimodal_confidence or t.weak_confidence or 0.0)
        item = ReviewItem(
            sent_id=t.sent_id,
            item_type="low_confidence",
            token_idx=t.token_idx,
            reason=f"confidence<{conf_low} ({conf_eff:.3f})",
            score=conf_eff,
            status="open",
        )
        db.session.add(item)
        created_low += 1

    if include_conflict:
        q_conf = (
            db.session.query(TokenAnnotation)
            .filter(~TokenAnnotation.sent_id.in_(gold_sent_ids))
            .filter(TokenAnnotation.weak_label != "O")
            .filter(TokenAnnotation.model_label != "O")
            .filter(TokenAnnotation.weak_label != TokenAnnotation.model_label)
            .limit(int(limit_per_type))
            .all()
        )
        for t in q_conf:
            item = ReviewItem(
                sent_id=t.sent_id,
                item_type="conflict",
                token_idx=t.token_idx,
                reason=f"weak={t.weak_label}, model={t.model_label}",
                score=float(min(t.multimodal_confidence or t.weak_confidence or 0.0, t.model_confidence or 0.0)),
                status="open",
            )
            db.session.add(item)
            created_conflict += 1

    if include_ambiguity:
        q_amb = (
            db.session.query(TokenAnnotation)
            .filter(~TokenAnnotation.sent_id.in_(gold_sent_ids))
            .filter(TokenAnnotation.weak_label != "O")
            .filter(func.coalesce(TokenAnnotation.multimodal_confidence, TokenAnnotation.weak_confidence, 0.0) >= float(conf_low))
            .filter(func.coalesce(TokenAnnotation.multimodal_confidence, TokenAnnotation.weak_confidence, 0.0) < float(conf_mid))
            .limit(int(limit_per_type))
            .all()
        )
        for t in q_amb:
            conf_eff = float(t.multimodal_confidence or t.weak_confidence or 0.0)
            item = ReviewItem(
                sent_id=t.sent_id,
                item_type="ambiguity",
                token_idx=t.token_idx,
                reason=f"{conf_low}<=confidence<{conf_mid} ({conf_eff:.3f})",
                score=conf_eff,
                status="open",
            )
            db.session.add(item)
            created_ambiguity += 1

    if include_pdf_risky:
        q_pdf = (
            db.session.query(TokenAnnotation)
            .filter(~TokenAnnotation.sent_id.in_(gold_sent_ids))
            .filter(TokenAnnotation.weak_label != "O")
            .filter(TokenAnnotation.weak_source.in_(["both", "pdf"]))
            .filter(TokenAnnotation.pdf_confidence < float(conf_pdf))
            .limit(int(limit_per_type))
            .all()
        )
        for t in q_pdf:
            item = ReviewItem(
                sent_id=t.sent_id,
                item_type="pdf_risky",
                token_idx=t.token_idx,
                reason=f"pdf_confidence<{conf_pdf} ({float(t.pdf_confidence or 0.0):.3f})",
                score=float(t.pdf_confidence or 0.0),
                status="open",
            )
            db.session.add(item)
            created_pdf_risky += 1

    db.session.commit()
    return {
        "review_items_created": int(created_low + created_conflict + created_ambiguity + created_pdf_risky),
        "low_confidence": int(created_low),
        "conflict": int(created_conflict),
        "ambiguity": int(created_ambiguity),
        "pdf_risky": int(created_pdf_risky),
    }
