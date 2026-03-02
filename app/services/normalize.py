import re
import unicodedata


# символы, которые часто ломаются в PDF
CHAR_NORMALIZATION_MAP = {
    "ö": "ӧ",
    "Ö": "Ӧ",
    "ÿ": "ӱ",
    "Ÿ": "Ӱ",
    "ä": "ӓ",
    "Ä": "Ӓ",
    "ü": "ӱ",   # иногда ü вместо ӱ
    "Ü": "Ӱ",
}


WHITESPACE_RE = re.compile(r"\s+")
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def normalize_chars(text: str) -> str:
    """
    Приводит спецсимволы к единому виду (PDF → нормальные буквы)
    """
    for bad, good in CHAR_NORMALIZATION_MAP.items():
        text = text.replace(bad, good)
    return text


def remove_pdf_artifacts(text: str) -> str:
    """
    Убирает типичные мусорные символы после OCR/PDF
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")   # soft hyphen
    return text


def join_hyphenated_lines(text: str) -> str:
    """
    Склеивает переносы слов:
    при-\nмер → пример
    """
    return HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """
    Полная нормализация для хранения и поиска
    """
    if not text:
        return ""

    text = remove_pdf_artifacts(text)
    text = join_hyphenated_lines(text)
    text = normalize_chars(text)
    text = normalize_whitespace(text)

    return text


def normalize_for_match(text: str) -> str:
    """
    Агрессивная нормализация для сопоставления
    (для weak annotation и поиска)
    """
    text = normalize_text(text)
    text = text.lower()

    # убираем пунктуацию
    text = re.sub(r"[^\w\s]", "", text)

    return text


def normalize_headword(word: str) -> str:
    """
    Нормализация словарной леммы
    """
    return normalize_for_match(word)


def normalize_sentence(sentence: str) -> str:
    """
    Нормализация предложения корпуса
    """
    return normalize_for_match(sentence)
