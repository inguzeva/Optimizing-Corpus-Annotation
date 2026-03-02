from __future__ import annotations

from math import ceil
from typing import Optional

from sqlalchemy import func, or_

from app.models import db, Entry, Sense, Phrase, Example
from app.services.normalize import normalize_for_match


def _paginate_meta(total: int, page: int, per_page: int) -> dict:
    page = max(1, int(page or 1))
    per_page = max(1, min(int(per_page or 50), 200))

    pages = max(1, ceil(total / per_page)) if total else 1
    page = min(page, pages)

    return {
        "page": page,
        "per_page": per_page,
        "total": int(total),
        "pages": pages,
        "has_prev": page > 1,
        "has_next": page < pages,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < pages else None,
    }


def _counts_map(model, entry_ids: list[str], count_col) -> dict[str, int]:
    if not entry_ids:
        return {}

    rows = (
        db.session.query(model.entry_id, func.count(count_col))
        .filter(model.entry_id.in_(entry_ids))
        .group_by(model.entry_id)
        .all()
    )
    return {eid: int(cnt) for eid, cnt in rows}


def search_entries(
    query: str = "",
    gloss_query: str = "",
    has_phrases: Optional[bool] = None,
    has_examples: Optional[bool] = None,
    min_senses: int = 0,
    page: int = 1,
    per_page: int = 50,
) -> dict:
    """
    Поиск и фильтры по словарю из БД.
    """
    q = normalize_for_match(query) if query else ""
    gq = (gloss_query or "").strip()

    q_entries = db.session.query(Entry)

    if q:
        q_entries = q_entries.filter(
            or_(
                Entry.headword_norm.contains(q),
                Entry.headword_raw.ilike(f"%{query}%"),
            )
        )

    if gq:
        gloss_ids = (
            db.session.query(Sense.entry_id)
            .filter(
                or_(
                    Sense.gloss_ru.ilike(f"%{gq}%"),
                    Sense.gloss_ru_alt.ilike(f"%{gq}%"),
                )
            )
            .distinct()
        )
        q_entries = q_entries.filter(Entry.entry_id.in_(gloss_ids))

    if has_phrases is True:
        with_phrase_ids = db.session.query(Phrase.entry_id).distinct()
        q_entries = q_entries.filter(Entry.entry_id.in_(with_phrase_ids))
    elif has_phrases is False:
        with_phrase_ids = db.session.query(Phrase.entry_id).distinct()
        q_entries = q_entries.filter(~Entry.entry_id.in_(with_phrase_ids))

    if has_examples is True:
        with_example_ids = db.session.query(Example.entry_id).distinct()
        q_entries = q_entries.filter(Entry.entry_id.in_(with_example_ids))
    elif has_examples is False:
        with_example_ids = db.session.query(Example.entry_id).distinct()
        q_entries = q_entries.filter(~Entry.entry_id.in_(with_example_ids))

    if min_senses and int(min_senses) > 0:
        senses_ids = (
            db.session.query(Sense.entry_id)
            .group_by(Sense.entry_id)
            .having(func.count(Sense.id) >= int(min_senses))
        )
        q_entries = q_entries.filter(Entry.entry_id.in_(senses_ids))

    total = q_entries.count()
    meta = _paginate_meta(total=total, page=page, per_page=per_page)

    rows = (
        q_entries.order_by(Entry.headword_norm.asc(), Entry.headword_raw.asc(), Entry.entry_id.asc())
        .offset((meta["page"] - 1) * meta["per_page"])
        .limit(meta["per_page"])
        .all()
    )

    entry_ids = [r.entry_id for r in rows]
    senses_count = _counts_map(Sense, entry_ids, Sense.id)
    phrases_count = _counts_map(Phrase, entry_ids, Phrase.id)
    examples_count = _counts_map(Example, entry_ids, Example.id)

    items = []
    for e in rows:
        row = e.to_dict()
        row["senses_count"] = int(senses_count.get(e.entry_id, 0))
        row["phrases_count"] = int(phrases_count.get(e.entry_id, 0))
        row["examples_count"] = int(examples_count.get(e.entry_id, 0))
        items.append(row)

    meta["items"] = items
    return meta


def get_entry_preview(entry_id: str, max_senses: int = 3) -> dict:
    entry = Entry.query.get(str(entry_id))
    if not entry:
        return {}

    senses = (
        Sense.query.filter_by(entry_id=entry.entry_id)
        .order_by(Sense.sense_id.asc())
        .limit(max_senses)
        .all()
    )
    return {
        "entry": entry.to_dict(),
        "senses": [s.to_dict() for s in senses],
        "senses_count": Sense.query.filter_by(entry_id=entry.entry_id).count(),
        "phrases_count": Phrase.query.filter_by(entry_id=entry.entry_id).count(),
        "examples_count": Example.query.filter_by(entry_id=entry.entry_id).count(),
    }
