import re
from collections.abc import Iterable
from datetime import date


YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
ABSTRACT_PATTERN = re.compile(r"\babstract\b\s*:?", re.IGNORECASE)
KEYWORDS_PATTERN = re.compile(r"\b(?:keywords?|key words?|index terms?)\s*(?:[:\-—]\s*)?", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
AFFILIATION_PATTERN = re.compile(
    r"\b(?:university|institute|department|school|college|laboratory|laboratories|"
    r"faculty|engineering|science|geography|environmental|carleton|inc\.|ltd\.)\b",
    re.IGNORECASE,
)
AUTHOR_MARKER_PATTERN = re.compile(
    r"\b(?:corresponding author|author for correspondence|orcid)\b",
    re.IGNORECASE,
)
HEADING_PATTERN = re.compile(r"(?m)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Za-z /&-]{2,60}\s*$")


def _item(value: object, page_number: int, confidence: float, evidence: str) -> dict[str, object] | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return {
        "value": value,
        "page_number": page_number,
        "confidence": round(min(confidence, 0.99), 2),
        "evidence": evidence[:400],
    }


def _first_page(pages: list[dict[str, object]]) -> tuple[int, str]:
    if not pages:
        return 1, ""
    return int(pages[0].get("page_number", 1)), str(pages[0].get("text", "")).strip()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\r\n:;-")


def _abstract_from_sections(sections: Iterable[dict[str, object]]) -> dict[str, object] | None:
    for section in sections:
        if section.get("section_id") == "abstract":
            page = int(section.get("page_start", 1))
            text = re.split(r"\b(?:keywords?|key words?|index terms?)\s*[:\-—]", str(section.get("text", "")), maxsplit=1, flags=re.IGNORECASE)[0]
            text = _clean(text).lstrip("—-: ")
            return _item(text, page, 0.94, text)
        nested = _abstract_from_sections(section.get("subsections", []))
        if nested:
            return nested
    return None


def _extract_abstract(first_text: str, page_number: int) -> dict[str, object] | None:
    match = ABSTRACT_PATTERN.search(first_text)
    if not match:
        return None
    remainder = first_text[match.end():]
    stop = re.search(
        r"\bkeywords?\s*[:\-—]|\bkey words?\s*[:\-—]|\bindex terms?\s*[:\-—]|"
        r"\b(?:(?:\d+|[IVX]+)[.)]?\s+)(?:introduction|background|methods?|methodology)\b",
        remainder,
        re.IGNORECASE | re.MULTILINE,
    )
    abstract = remainder[: stop.start()] if stop else remainder
    abstract = _clean(abstract)
    return _item(abstract, page_number, 0.84, abstract)


def _extract_keywords(first_text: str, page_number: int) -> dict[str, object] | None:
    match = KEYWORDS_PATTERN.search(first_text)
    if not match:
        return None
    remainder = first_text[match.end():]
    stop = re.search(
        r"\b(?:(?:\d+|[IVX]+)[.)]?\s+)(?:introduction|background|methods?|methodology)\b",
        remainder,
        re.IGNORECASE | re.MULTILINE,
    )
    keyword_text = _clean(remainder[: stop.start()] if stop else remainder)
    keyword_text = re.split(r"\b(?:abstract)\b", keyword_text, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ;,")
    values = [_clean(value) for value in re.split(r",|;|•|\|", keyword_text) if _clean(value)]
    if not values or len(values) > 20:
        return None
    return _item(values, page_number, 0.84, keyword_text)


def _title_candidates(first_text: str) -> list[str]:
    abstract_position = ABSTRACT_PATTERN.search(first_text)
    prefix = first_text[:abstract_position.start()] if abstract_position else first_text[:1000]
    author_start = re.search(
        r"\b[A-Z][a-z.'-]+\s+[A-Z][a-z.'-]+(?=\d|[,*;])",
        prefix,
    )
    if author_start:
        prefix = prefix[:author_start.start()]
    prefix = re.sub(
        r"^.*(?:\b\d+\s+)(?=[A-Z][A-Za-z])",
        "",
        prefix,
        count=1,
    )
    lines = [_clean(line) for line in re.split(r"\n+", prefix) if _clean(line)]
    candidates = []
    for line in lines:
        if EMAIL_PATTERN.search(line) or URL_PATTERN.search(line):
            continue
        if AFFILIATION_PATTERN.search(line) or YEAR_PATTERN.search(line):
            continue
        if AUTHOR_MARKER_PATTERN.search(line):
            continue
        if len(line.split()) >= 3:
            candidates.append(line)
    if candidates:
        return candidates
    compact = _clean(prefix)
    compact = re.split(
        r"\b(?:abstract|[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:\s*,|\s+and\b))",
        compact,
        maxsplit=1,
    )[0]
    return [compact] if len(compact.split()) >= 3 else []


def _extract_title(first_text: str, page_number: int) -> dict[str, object] | None:
    candidates = _title_candidates(first_text)
    if not candidates:
        return None
    title = max(candidates, key=lambda value: (min(len(value.split()), 18), -len(value)))
    return _item(title, page_number, 0.78, title)


def _extract_authors(first_text: str, title: dict[str, object] | None, page_number: int) -> dict[str, object] | None:
    abstract_match = ABSTRACT_PATTERN.search(first_text)
    prefix = first_text[:abstract_match.start()] if abstract_match else first_text[:1500]
    title_text = str(title["value"]) if title else ""
    if title_text and title_text in prefix:
        prefix = prefix[prefix.find(title_text) + len(title_text):]
    author_start = re.search(
        r"\b[A-Z][a-z.'-]+\s+[A-Z][a-z.'-]+(?=\d|[,*;])",
        prefix,
    )
    if author_start:
        prefix = prefix[author_start.start():]
    prefix = re.split(r"\babstract\b|\bindex terms?\b", prefix, maxsplit=1, flags=re.IGNORECASE)[0]
    prefix = EMAIL_PATTERN.sub("", URL_PATTERN.sub("", prefix))
    prefix = AUTHOR_MARKER_PATTERN.sub("", prefix)
    prefix = re.sub(r"\b(?:ORCID|doi)\s*:\s*\S+", "", prefix, flags=re.IGNORECASE)
    names = []
    for match in re.finditer(r"\b[A-Z][A-Za-z.'-]+(?:[ \t]+[A-Z](?:\.)?)?(?:[ \t]+[A-Z][A-Za-z.'-]+){1,2}\b", prefix):
        name = _clean(match.group(0))
        if (
            not AFFILIATION_PATTERN.search(name)
            and not YEAR_PATTERN.search(name)
            and len(name.split()) <= 4
            and not re.search(r"\b(?:fellow|member|ieee)\b", name, re.IGNORECASE)
            and name.lower() not in {"abstract", "index terms"}
        ):
            names.append(name)
    unique = list(dict.fromkeys(names))
    return _item(unique, page_number, 0.74, ", ".join(unique)) if unique else None


def _extract_year(
    first_text: str,
    page_number: int,
    document_metadata: dict[str, object] | None = None,
) -> dict[str, object] | None:
    first_page = first_text[:2500]
    abstract_position = ABSTRACT_PATTERN.search(first_page)
    front_matter = first_page[:abstract_position.start()] if abstract_position else first_page
    context = re.search(
        r"(?i)(?:published|publication|accepted|received|conference|proceedings|©|copyright)"
        r".{0,80}",
        front_matter,
    )
    search_text = context.group(0) if context else front_matter
    years = [int(match.group(1)) for match in YEAR_PATTERN.finditer(search_text)]
    valid = [year for year in years if 1900 <= year <= date.today().year]
    if not valid:
        valid = [
            int(match.group(1))
            for match in YEAR_PATTERN.finditer(front_matter)
            if 1900 <= int(match.group(1)) <= date.today().year
        ]
        search_text = front_matter
    if not valid and document_metadata:
        metadata_text = " ".join(
            str(document_metadata.get(key, ""))
            for key in ("creationDate", "modDate", "subject", "keywords")
        )
        valid = [
            int(match.group(1))
            for match in YEAR_PATTERN.finditer(metadata_text)
            if 1900 <= int(match.group(1)) <= date.today().year
        ]
        search_text = metadata_text
    if not valid:
        return None
    year = valid[-1]
    confidence = 0.9 if context else 0.78 if front_matter else 0.62
    return _item(year, page_number, confidence, search_text)


def extract_metadata(
    pages: Iterable[dict[str, object]],
    sections: Iterable[dict[str, object]] | None = None,
    document_metadata: dict[str, object] | None = None,
) -> dict[str, dict[str, object] | None]:
    """Extract basic first-page metadata without inferring missing values."""
    page_list = list(pages)
    page_number, first_text = _first_page(page_list)
    title = _extract_title(first_text, page_number)
    abstract = _abstract_from_sections(sections or []) or _extract_abstract(first_text, page_number)
    return {
        "title": title,
        "authors": _extract_authors(first_text, title, page_number),
        "publication_year": _extract_year(first_text, page_number, document_metadata),
        "abstract": abstract,
        "keywords": _extract_keywords(first_text, page_number),
    }
