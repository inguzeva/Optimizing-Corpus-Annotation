from __future__ import annotations

from collections import Counter

from sqlalchemy import and_, func

from app.models import db, Entry, Sense, Phrase, Example


def compute_dictionary_stats() -> dict:
    """
    Статистика по словарю из БД (если словарь импортирован).
    """
    total_entries = Entry.query.count()
    total_senses = Sense.query.count()
    total_phrases = Phrase.query.count()
    total_examples = Example.query.count()

    with_pos = Entry.query.filter(Entry.pos_primary != "").count()
    with_labels = Entry.query.filter(Entry.labels != "").count()
    with_phrases = (
        db.session.query(Phrase.entry_id).distinct().count()
    )
    with_examples = (
        db.session.query(Example.entry_id).distinct().count()
    )
    multi_sense = (
        db.session.query(Sense.entry_id)
        .group_by(Sense.entry_id)
        .having(func.count(Sense.id) >= 2)
        .count()
    )

    senses_count_dist = Counter()
    for _, cnt in (
        db.session.query(Sense.entry_id, func.count(Sense.id))
        .group_by(Sense.entry_id)
        .all()
    ):
        senses_count_dist[int(cnt)] += 1

    # entries без значений
    entries_without_senses = total_entries - sum(senses_count_dist.values())
    if entries_without_senses > 0:
        senses_count_dist[0] += entries_without_senses

    return {
        "counts": {
            "entries": int(total_entries),
            "senses": int(total_senses),
            "phrases": int(total_phrases),
            "examples": int(total_examples),
        },
        "coverage": {
            "with_pos": int(with_pos),
            "with_labels": int(with_labels),
            "with_phrases": int(with_phrases),
            "with_examples": int(with_examples),
            "multi_sense": int(multi_sense),
        },
        "coverage_pct": {
            "with_pos": _pct(with_pos, total_entries),
            "with_labels": _pct(with_labels, total_entries),
            "with_phrases": _pct(with_phrases, total_entries),
            "with_examples": _pct(with_examples, total_entries),
            "multi_sense": _pct(multi_sense, total_entries),
        },
        "senses_count_distribution": dict(senses_count_dist),
    }


def validate_dictionary_links(sample_limit: int = 200) -> dict:
    """
    Проверка целостности:
      - senses/examples/phrases не должны ссылаться на несуществующий entry_id
      - базовые проверки пустых полей
    """
    entry_ids_subq = db.session.query(Entry.entry_id)

    senses_orphans = (
        Sense.query.filter(~Sense.entry_id.in_(entry_ids_subq))
        .limit(sample_limit)
        .all()
    )
    phrases_orphans = (
        Phrase.query.filter(~Phrase.entry_id.in_(entry_ids_subq))
        .limit(sample_limit)
        .all()
    )
    examples_orphans = (
        Example.query.filter(~Example.entry_id.in_(entry_ids_subq))
        .limit(sample_limit)
        .all()
    )

    empty_senses = (
        Sense.query.filter(Sense.gloss_ru == "")
        .limit(sample_limit)
        .all()
    )
    empty_examples = (
        Example.query
        .filter(and_(Example.example_alt_raw == "", Example.example_ru == ""))
        .limit(sample_limit)
        .all()
    )

    empty_fields = []
    for s in empty_senses:
        empty_fields.append(
            {
                "type": "sense",
                "entry_id": s.entry_id,
                "sense_id": s.sense_id,
                "field": "gloss_ru",
            }
        )
    for ex in empty_examples:
        if len(empty_fields) >= sample_limit:
            break
        empty_fields.append(
            {
                "type": "example",
                "entry_id": ex.entry_id,
                "sense_id": ex.sense_id,
                "field": "example_alt_raw/example_ru",
            }
        )

    samples = {
        "senses_orphans": [
            {"entry_id": s.entry_id, "sense_id": s.sense_id, "gloss_ru": (s.gloss_ru or "")[:120]}
            for s in senses_orphans
        ],
        "phrases_orphans": [
            {"entry_id": p.entry_id, "phrase_id": p.phrase_id, "phrase_alt_raw": (p.phrase_alt_raw or "")[:120]}
            for p in phrases_orphans
        ],
        "examples_orphans": [
            {
                "entry_id": ex.entry_id,
                "sense_id": ex.sense_id,
                "ex_id": ex.ex_id,
                "example_alt_raw": (ex.example_alt_raw or "")[:120],
            }
            for ex in examples_orphans
        ],
        "empty_fields": empty_fields,
    }

    summary = {k: len(v) for k, v in samples.items()}
    return {"summary": summary, "samples": samples}


def validate_entry_fields(sample_limit: int = 200) -> dict:
    """
    Проверяет заполненность ключевых полей entries.
    """
    rows = (
        Entry.query
        .filter(
            (Entry.headword_raw == "")
            | (Entry.headword_norm == "")
            | (Entry.raw_text == "")
        )
        .limit(sample_limit)
        .all()
    )

    samples = []
    for e in rows:
        hw_raw = (e.headword_raw or "").strip()
        hw_norm = (e.headword_norm or "").strip()
        raw_text = (e.raw_text or "").strip()
        samples.append(
            {
                "entry_id": e.entry_id,
                "headword_raw": hw_raw[:60],
                "headword_norm": hw_norm[:60],
                "missing_headword_raw": 0 if hw_raw else 1,
                "missing_headword_norm": 0 if hw_norm else 1,
                "missing_raw_text": 0 if raw_text else 1,
            }
        )

    return {"count": len(samples), "samples": samples}


def _pct(x: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((x / total) * 100.0, 2)
