from __future__ import annotations

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Entry(db.Model):
    __tablename__ = "entries"

    entry_id = db.Column(db.String, primary_key=True)
    headword_raw = db.Column(db.Text, nullable=False, default="")
    headword_norm = db.Column(db.Text, nullable=False, index=True, default="")
    homonym_roman = db.Column(db.String(10), nullable=False, default="")
    pos_primary = db.Column(db.String(64), nullable=False, index=True, default="")
    labels = db.Column(db.Text, nullable=False, default="")
    has_phrases = db.Column(db.Integer, nullable=False, default=0)
    raw_text = db.Column(db.Text, nullable=False, default="")
    page_start = db.Column(db.String(32), nullable=False, default="")
    page_end = db.Column(db.String(32), nullable=False, default="")

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    senses = db.relationship("Sense", backref="entry", lazy=True, cascade="all, delete-orphan")
    phrases = db.relationship("Phrase", backref="entry", lazy=True, cascade="all, delete-orphan")
    examples = db.relationship("Example", backref="entry", lazy=True, cascade="all, delete-orphan")
    pdf_feature = db.relationship("EntryPdfFeature", backref="entry", uselist=False, cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "headword_raw": self.headword_raw,
            "headword_norm": self.headword_norm,
            "homonym_roman": self.homonym_roman,
            "pos_primary": self.pos_primary,
            "labels": self.labels,
            "has_phrases": self.has_phrases,
            "raw_text": self.raw_text,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


class Sense(db.Model):
    __tablename__ = "senses"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    entry_id = db.Column(db.String, db.ForeignKey("entries.entry_id"), nullable=False, index=True)
    sense_id = db.Column(db.String(16), nullable=False, default="1")
    sense_group = db.Column(db.String(16), nullable=False, default="none")

    pos = db.Column(db.String(64), nullable=False, default="")
    gloss_ru = db.Column(db.Text, nullable=False, default="")
    gloss_ru_alt = db.Column(db.Text, nullable=False, default="")
    labels = db.Column(db.Text, nullable=False, default="")
    refs = db.Column(db.Text, nullable=False, default="")
    confidence_parse = db.Column(db.String(16), nullable=False, default="")

    raw_block = db.Column(db.Text, nullable=False, default="")

    __table_args__ = (
        db.Index("idx_senses_entry_sense", "entry_id", "sense_id"),
    )

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "sense_id": self.sense_id,
            "sense_group": self.sense_group,
            "pos": self.pos,
            "gloss_ru": self.gloss_ru,
            "gloss_ru_alt": self.gloss_ru_alt,
            "labels": self.labels,
            "refs": self.refs,
            "confidence_parse": self.confidence_parse,
            "raw_block": self.raw_block,
        }


class Phrase(db.Model):
    __tablename__ = "phrases"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    entry_id = db.Column(db.String, db.ForeignKey("entries.entry_id"), nullable=False, index=True)
    phrase_id = db.Column(db.String(32), nullable=False, default="1")

    phrase_alt_raw = db.Column(db.Text, nullable=False, default="")
    phrase_alt_norm = db.Column(db.Text, nullable=False, index=True, default="")
    phrase_ru = db.Column(db.Text, nullable=False, default="")
    labels = db.Column(db.Text, nullable=False, default="")
    raw_block = db.Column(db.Text, nullable=False, default="")

    __table_args__ = (
        db.Index("idx_phrases_entry_phrase", "entry_id", "phrase_id"),
    )

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "phrase_id": self.phrase_id,
            "phrase_alt_raw": self.phrase_alt_raw,
            "phrase_alt_norm": self.phrase_alt_norm,
            "phrase_ru": self.phrase_ru,
            "labels": self.labels,
            "raw_block": self.raw_block,
        }


class Example(db.Model):
    __tablename__ = "examples"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    entry_id = db.Column(db.String, db.ForeignKey("entries.entry_id"), nullable=False, index=True)
    sense_id = db.Column(db.String(16), nullable=False, default="1")
    ex_id = db.Column(db.String(32), nullable=False, default="1")

    example_alt_raw = db.Column(db.Text, nullable=False, default="")
    example_alt_norm = db.Column(db.Text, nullable=False, index=True, default="")
    example_ru = db.Column(db.Text, nullable=False, default="")
    source_note = db.Column(db.String(32), nullable=False, default="")
    raw_line = db.Column(db.Text, nullable=False, default="")

    __table_args__ = (
        db.Index("idx_examples_entry_sense_ex", "entry_id", "sense_id", "ex_id"),
    )

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "sense_id": self.sense_id,
            "ex_id": self.ex_id,
            "example_alt_raw": self.example_alt_raw,
            "example_alt_norm": self.example_alt_norm,
            "example_ru": self.example_ru,
            "source_note": self.source_note,
            "raw_line": self.raw_line,
        }


class AbbrLabel(db.Model):
    __tablename__ = "abbr_labels"

    abbr = db.Column(db.String(64), primary_key=True)
    full = db.Column(db.Text, nullable=False, default="")
    type = db.Column(db.String(32), nullable=False, default="label")

    def to_dict(self) -> dict:
        return {"abbr": self.abbr, "full": self.full, "type": self.type}


class CorpusSentence(db.Model):
    __tablename__ = "corpus_sentences"

    sent_id = db.Column(db.String, primary_key=True)
    text = db.Column(db.Text, nullable=False, default="")
    text_norm = db.Column(db.Text, nullable=False, default="", index=True)

    covered = db.Column(db.Integer, nullable=False, default=0)
    hits_count = db.Column(db.Integer, nullable=False, default=0)
    phrase_hits_count = db.Column(db.Integer, nullable=False, default=0)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    tokens = db.relationship("TokenAnnotation", backref="sentence", lazy=True, cascade="all, delete-orphan")
    review_items = db.relationship("ReviewItem", backref="sentence", lazy=True, cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "sent_id": self.sent_id,
            "text": self.text,
            "text_norm": self.text_norm,
            "covered": self.covered,
            "hits_count": self.hits_count,
            "phrase_hits_count": self.phrase_hits_count,
        }


class TokenAnnotation(db.Model):
    __tablename__ = "token_annotations"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    sent_id = db.Column(db.String, db.ForeignKey("corpus_sentences.sent_id"), nullable=False, index=True)
    token_idx = db.Column(db.Integer, nullable=False)

    token = db.Column(db.Text, nullable=False, default="")

    weak_label = db.Column(db.String(32), nullable=False, default="O")
    weak_entry_id = db.Column(db.String, nullable=False, default="")
    weak_pos = db.Column(db.String(64), nullable=False, default="")
    weak_confidence = db.Column(db.Float, nullable=False, default=0.0)
    weak_source = db.Column(db.String(32), nullable=False, default="none")
    text_confidence = db.Column(db.Float, nullable=False, default=0.0)
    pdf_confidence = db.Column(db.Float, nullable=False, default=0.0)
    multimodal_confidence = db.Column(db.Float, nullable=False, default=0.0)

    model_label = db.Column(db.String(32), nullable=False, default="O")
    model_confidence = db.Column(db.Float, nullable=False, default=0.0)

    final_label = db.Column(db.String(32), nullable=False, default="")
    final_source = db.Column(db.String(32), nullable=False, default="")  # user/model/dict
    is_confirmed = db.Column(db.Integer, nullable=False, default=0)

    __table_args__ = (
        db.UniqueConstraint("sent_id", "token_idx", name="uq_token_sent_idx"),
        db.Index("idx_token_sent_idx", "sent_id", "token_idx"),
    )

    def to_dict(self) -> dict:
        return {
            "sent_id": self.sent_id,
            "token_idx": self.token_idx,
            "token": self.token,
            "weak_label": self.weak_label,
            "weak_entry_id": self.weak_entry_id,
            "weak_pos": self.weak_pos,
            "weak_confidence": self.weak_confidence,
            "weak_source": self.weak_source,
            "text_confidence": self.text_confidence,
            "pdf_confidence": self.pdf_confidence,
            "multimodal_confidence": self.multimodal_confidence,
            "model_label": self.model_label,
            "model_confidence": self.model_confidence,
            "final_label": self.final_label,
            "final_source": self.final_source,
            "is_confirmed": self.is_confirmed,
        }


class EntryPdfFeature(db.Model):
    __tablename__ = "entry_pdf_features"

    entry_id = db.Column(db.String, db.ForeignKey("entries.entry_id"), primary_key=True)

    page_start_int = db.Column(db.Integer, nullable=False, default=0)
    page_end_int = db.Column(db.Integer, nullable=False, default=0)
    page_span = db.Column(db.Integer, nullable=False, default=0)
    has_page_info = db.Column(db.Integer, nullable=False, default=0)

    raw_text_chars = db.Column(db.Integer, nullable=False, default=0)
    raw_text_lines = db.Column(db.Integer, nullable=False, default=0)
    has_phrase_marker = db.Column(db.Integer, nullable=False, default=0)

    parse_confidence_mean = db.Column(db.Float, nullable=False, default=0.0)
    parse_confidence_max = db.Column(db.Float, nullable=False, default=0.0)

    pdf_quality_score = db.Column(db.Float, nullable=False, default=0.0, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "page_start_int": self.page_start_int,
            "page_end_int": self.page_end_int,
            "page_span": self.page_span,
            "has_page_info": self.has_page_info,
            "raw_text_chars": self.raw_text_chars,
            "raw_text_lines": self.raw_text_lines,
            "has_phrase_marker": self.has_phrase_marker,
            "parse_confidence_mean": self.parse_confidence_mean,
            "parse_confidence_max": self.parse_confidence_max,
            "pdf_quality_score": self.pdf_quality_score,
        }


class ReviewItem(db.Model):
    __tablename__ = "review_items"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    sent_id = db.Column(db.String, db.ForeignKey("corpus_sentences.sent_id"), nullable=False, index=True)
    item_type = db.Column(db.String(32), nullable=False, index=True, default="low_confidence")  # low_confidence/conflict/ambiguity/pdf_risky

    token_idx = db.Column(db.Integer, nullable=True)
    reason = db.Column(db.Text, nullable=False, default="")
    score = db.Column(db.Float, nullable=False, default=0.0)

    status = db.Column(db.String(16), nullable=False, default="open")  # open/done/skipped
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sent_id": self.sent_id,
            "item_type": self.item_type,
            "token_idx": self.token_idx,
            "reason": self.reason,
            "score": self.score,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class GoldSentence(db.Model):
    __tablename__ = "gold_sentences"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    sent_id = db.Column(db.String, nullable=False, index=True)
    text = db.Column(db.Text, nullable=False, default="")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    tokens = db.relationship("GoldToken", backref="gold_sentence", lazy=True, cascade="all, delete-orphan")


class GoldToken(db.Model):
    __tablename__ = "gold_tokens"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    gold_sentence_id = db.Column(db.Integer, db.ForeignKey("gold_sentences.id"), nullable=False, index=True)

    token_idx = db.Column(db.Integer, nullable=False)
    token = db.Column(db.Text, nullable=False, default="")
    label = db.Column(db.String(32), nullable=False, default="O")

    __table_args__ = (
        db.UniqueConstraint("gold_sentence_id", "token_idx", name="uq_gold_token_idx"),
    )


def init_db(app) -> None:
    db.init_app(app)
    with app.app_context():
        db.create_all()
