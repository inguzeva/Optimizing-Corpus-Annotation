from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, render_template, request, flash, redirect, url_for

from app.models import db, CorpusSentence, TokenAnnotation, ReviewItem, GoldSentence, GoldToken


review_bp = Blueprint("review", __name__)


def _is_sentence_fully_reviewed(sent_id: str) -> bool:
    toks = TokenAnnotation.query.filter_by(sent_id=sent_id).all()
    if not toks:
        return False
    return all(int(t.is_confirmed or 0) == 1 for t in toks)


@review_bp.get("/")
def review_queue():
    qtype = request.args.get("type", "low_confidence").strip()

    items = (
        ReviewItem.query
        .filter(ReviewItem.status == "open")
        .filter(ReviewItem.item_type == qtype)
        .order_by(ReviewItem.score.asc(), ReviewItem.id.asc())
        .limit(200)
        .all()
    )

    # подгружаем тексты предложений пачкой
    sent_ids = list({i.sent_id for i in items})
    sent_map = {}
    if sent_ids:
        for s in CorpusSentence.query.filter(CorpusSentence.sent_id.in_(sent_ids)).all():
            sent_map[s.sent_id] = s

    return render_template(
        "review/queue.html",
        items=items,
        qtype=qtype,
        sent_map=sent_map,
    )


@review_bp.get("/sent/<sent_id>")
def review_sentence(sent_id: str):
    sent = CorpusSentence.query.get(sent_id)
    if not sent:
        flash("Предложение не найдено", "error")
        return redirect(url_for("review.review_queue"))

    tokens = (
        TokenAnnotation.query
        .filter_by(sent_id=sent_id)
        .order_by(TokenAnnotation.token_idx.asc())
        .all()
    )

    return render_template(
        "review/sentence.html",
        sent=sent,
        tokens=tokens,
    )


@review_bp.post("/update_token")
def update_token():
    data = request.get_json(silent=True) or {}
    sent_id = str(data.get("sent_id", "")).strip()
    token_idx = data.get("token_idx", None)
    label = (data.get("label") or "").strip()

    if not sent_id or token_idx is None:
        return jsonify({"ok": False, "error": "Не переданы sent_id/token_idx"}), 400

    try:
        token_idx = int(token_idx)
    except Exception:
        return jsonify({"ok": False, "error": "token_idx должен быть целым числом"}), 400

    ta = TokenAnnotation.query.filter_by(sent_id=sent_id, token_idx=token_idx).first()
    if not ta:
        return jsonify({"ok": False, "error": "Токен не найден"}), 404

    # если пользователь удалил метку — считаем как O
    if label == "":
        label = "O"

    ta.final_label = label
    ta.final_source = "user"
    ta.is_confirmed = 1

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        return jsonify({"ok": False, "error": "Ошибка сохранения в БД"}), 500

    return jsonify({"ok": True, "sent_id": sent_id, "token_idx": token_idx, "label": label})


@review_bp.post("/queue_action")
def queue_action():
    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "").strip()
    item_id = data.get("item_id", None)

    if not action or item_id is None:
        return jsonify({"ok": False, "error": "Не переданы action/item_id"}), 400

    try:
        item_id = int(item_id)
    except Exception:
        return jsonify({"ok": False, "error": "item_id должен быть целым числом"}), 400

    item = ReviewItem.query.get(item_id)
    if not item:
        return jsonify({"ok": False, "error": "Элемент очереди не найден"}), 404

    if action == "skip":
        item.status = "skipped"
        item.resolved_at = datetime.utcnow()

    elif action == "done":
        item.status = "done"
        item.resolved_at = datetime.utcnow()

    elif action == "add_to_gold":
        # сохраняем предложение и текущие final_label (или weak_label если final пусто)
        sent = CorpusSentence.query.get(item.sent_id)
        if not sent:
            return jsonify({"ok": False, "error": "Предложение не найдено"}), 404

        if not _is_sentence_fully_reviewed(sent.sent_id):
            return jsonify(
                {
                    "ok": False,
                    "error": "Предложение проверено не полностью. Подтвердите все токены на странице предложения.",
                }
            ), 400

        existing = GoldSentence.query.filter_by(sent_id=sent.sent_id).first()
        if existing:
            item.status = "done"
            item.resolved_at = datetime.utcnow()
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
                return jsonify({"ok": False, "error": "Ошибка сохранения в БД"}), 500
            return jsonify({"ok": True, "item_id": item.id, "status": item.status, "message": "Предложение уже в gold"})

        gs = GoldSentence(sent_id=sent.sent_id, text=sent.text)
        db.session.add(gs)
        db.session.flush()  # получить gs.id

        toks = (
            TokenAnnotation.query
            .filter_by(sent_id=sent.sent_id)
            .order_by(TokenAnnotation.token_idx.asc())
            .all()
        )

        for t in toks:
            lbl = t.final_label if (t.final_label or "").strip() else "O"
            gt = GoldToken(
                gold_sentence_id=gs.id,
                token_idx=t.token_idx,
                token=t.token,
                label=lbl,
            )
            db.session.add(gt)

        # Закрываем все открытые элементы очереди для этого предложения.
        open_items = ReviewItem.query.filter_by(sent_id=sent.sent_id, status="open").all()
        for it in open_items:
            it.status = "done"
            it.resolved_at = datetime.utcnow()

    else:
        return jsonify({"ok": False, "error": f"Неизвестное действие: {action}"}), 400

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        return jsonify({"ok": False, "error": "Ошибка сохранения в БД"}), 500

    return jsonify({"ok": True, "item_id": item.id, "status": item.status})


@review_bp.post("/sentence/<sent_id>/confirm_all")
def confirm_sentence_all(sent_id: str):
    sent = CorpusSentence.query.get(sent_id)
    if not sent:
        flash("Предложение не найдено", "error")
        return redirect(url_for("review.review_queue"))

    toks = (
        TokenAnnotation.query
        .filter_by(sent_id=sent_id)
        .order_by(TokenAnnotation.token_idx.asc())
        .all()
    )

    for t in toks:
        lbl = (t.final_label or "").strip()
        if not lbl:
            lbl = (t.weak_label or "").strip() or "O"
        t.final_label = lbl
        t.final_source = "user"
        t.is_confirmed = 1

    try:
        db.session.commit()
        flash("Все токены подтверждены", "success")
    except Exception:
        db.session.rollback()
        flash("Ошибка сохранения в БД", "error")

    return redirect(url_for("review.review_sentence", sent_id=sent_id))


@review_bp.post("/sentence/<sent_id>/reset_final")
def reset_sentence_final(sent_id: str):
    sent = CorpusSentence.query.get(sent_id)
    if not sent:
        flash("Предложение не найдено", "error")
        return redirect(url_for("review.review_queue"))

    toks = TokenAnnotation.query.filter_by(sent_id=sent_id).all()
    for t in toks:
        t.final_label = ""
        t.final_source = ""
        t.is_confirmed = 0

    try:
        db.session.commit()
        flash("Финальные метки очищены", "success")
    except Exception:
        db.session.rollback()
        flash("Ошибка сохранения в БД", "error")

    return redirect(url_for("review.review_sentence", sent_id=sent_id))
