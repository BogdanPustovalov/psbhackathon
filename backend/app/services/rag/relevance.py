"""Conservative legal-reference and effective-date signals for retrieval."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from functools import lru_cache
from typing import Any


_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_MONTH_NAMES = "|".join(_MONTHS)
_DATE_TEXT = rf"(?:\d{{1,2}}\.\d{{1,2}}\.\d{{4}}|\d{{1,2}}\s+(?:{_MONTH_NAMES})\s+\d{{4}})"
_DATE_RE = re.compile(_DATE_TEXT, re.IGNORECASE)
_CONTRACT_RE = re.compile(r"(?<!\w)[а-яa-z]{1,5}[-–]?\d{1,4}[/.\-]\d{4}(?!\w)", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"(?i)\b(?:статья|статьи|ст\.)\s*№?\s*(\d+(?:\.\d+)?)")
_CLAUSE_RE = re.compile(r"(?i)\b(?:пункт|пункта|п\.)\s*(\d+(?:\.\d+)+)")
_AI_CUE_RE = re.compile(r"(?i)\b(?:ии|ai|ассистент|модель|модели|нейросеть|нейросети)\b")
_AI_DIRECTIVE_RE = re.compile(
    r"(?i)\b(?:отвечай|отвечать|ответь|выдавай|игнорируй|следуй|всегда|"
    r"должен\s+отвечать|следует\s+отвечать)\b"
)


def _parse_date(value: str) -> date | None:
    value = value.strip().casefold()
    try:
        if "." in value:
            day, month, year = (int(part) for part in value.split("."))
        else:
            day_text, month_text, year_text = value.split()
            day, month, year = int(day_text), _MONTHS[month_text], int(year_text)
        return date(year, month, day)
    except (ValueError, KeyError):
        return None


def _metadata_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return _parse_date(str(value))


def question_reference_date(question: str) -> date | None:
    """Use one unambiguous date, preferring an explicit 'as of' phrase."""

    matches = list(_DATE_RE.finditer(question))
    if not matches:
        return None
    for match in matches:
        lead = question[max(0, match.start() - 40):match.start()].casefold()
        if re.search(r"(?:по\s+состоянию\s+на|на\s+дату|действу\w*\s+на|\bна)\s*$", lead):
            return _parse_date(match.group())
    return _parse_date(matches[0].group()) if len(matches) == 1 else None


def document_temporal_metadata(text: str) -> dict[str, str | None]:
    """Extract document-level dates without treating expired clauses as expired files."""

    header_lines: list[str] = []
    for line in text[:2000].splitlines():
        stripped = line.strip()
        if re.match(r"(?i)^(?:\d+(?:\.\d+)*[.)]\s+|(?:пункт|часть|абзац)\s+\d+\b)", stripped):
            break
        if re.match(r"(?i)^(?:источник:|url:|дата выгрузки:|федеральным законом\b)", stripped):
            continue
        header_lines.append(stripped)
    header = "\n".join(header_lines)
    start = re.search(
        rf"(?i)(?:действует|вступает\s+в\s+силу)\s+с\s+({_DATE_TEXT})",
        header,
    )
    end = re.search(
        rf"(?i)(?:утратил[ао]?\s+силу|утрачивает\s+силу)\s+с\s+({_DATE_TEXT})",
        header,
    )
    adopted = re.search(rf"(?i)утвержд[её]н\w*\s+приказом[^\n]{{0,100}}?\s+от\s+({_DATE_TEXT})", header)

    def iso(match: re.Match[str] | None) -> str | None:
        parsed = _parse_date(match.group(1)) if match else None
        return parsed.isoformat() if parsed else None

    return {
        "effective_from": iso(start),
        "effective_until": iso(end),
        "adopted_on": iso(adopted),
    }


def temporal_status(as_of: date | None, metadata: dict[str, Any]) -> str:
    if as_of is None:
        return "not_requested"
    start = _metadata_date(metadata.get("effective_from"))
    end = _metadata_date(metadata.get("effective_until"))
    adopted = _metadata_date(metadata.get("adopted_on"))
    if start and as_of < start:
        return "future"
    if not start and adopted and as_of < adopted:
        return "future"
    if end and as_of >= end:
        return "expired"
    return "active" if start or end else "unknown"


def _compact(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFC", value).casefold()
        if char.isalnum()
    ).replace("ё", "е")


def exact_legal_reference_match(question: str, text: str, metadata: dict[str, Any]) -> bool:
    """Match an identifier in the question to the source title and passage."""

    source = str(metadata.get("filename") or metadata.get("source") or "")
    source_key = _compact(source)
    passage_key = _compact(text + " " + str(metadata.get("section") or "") + " " + str(metadata.get("clause") or ""))
    contracts = {_compact(match.group()) for match in _CONTRACT_RE.finditer(question)}
    if contracts:
        matching = {identifier for identifier in contracts if identifier in source_key}
        if not matching:
            return False
        clauses = {_compact(match.group(1)) for match in _CLAUSE_RE.finditer(question)}
        return not clauses or any(clause in passage_key for clause in clauses)
    articles = {"ст" + _compact(match.group(1)) for match in _ARTICLE_RE.finditer(question)}
    return bool(articles and any(article in source_key for article in articles))


def suspected_prompt_injection(text: str) -> bool:
    """Flag passages that instruct an AI to answer in a prescribed way."""

    return any(
        _AI_DIRECTIVE_RE.search(text[max(0, cue.start() - 160):cue.end() + 160])
        for cue in _AI_CUE_RE.finditer(text)
    )


@lru_cache(maxsize=8)
def _encoding_for_model(model: str):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_model_tokens(text: str, model: str) -> int:
    """Count tokens with a real tokenizer; use o200k for unknown model names."""

    return len(_encoding_for_model(model).encode(text, disallowed_special=()))
