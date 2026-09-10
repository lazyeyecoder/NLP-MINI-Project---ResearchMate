import math
import re
from collections import Counter
from collections.abc import Iterable


SUMMARY_SECTIONS = {
    "abstract": {"abstract"},
    "methodology": {"methodology", "proposed_method", "materials", "experiments"},
    "results": {"results", "discussion", "experiments"},
    "conclusion": {"conclusion"},
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "we",
    "with", "were", "our", "their", "they", "these", "using",
}
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "Fig.", "Eq.", "No.", "Dr.", "vs.")
BOILERPLATE_PATTERNS = (
    r"\bin recent years\b",
    r"\bprevious studies have\b",
    r"\bprior work\b",
    r"\bthe literature\b",
    r"\bthis paper is organized\b",
)
RESULT_PATTERNS = (
    r"\b(?:achieve|achieved|improv(?:e|ed|es)|outperform(?:s|ed)?|obtain|obtained)\b",
    r"\b(?:accuracy|f1|precision|recall|m(?:a|i)pp?|rmse|mae)\b",
    r"\b\d+(?:\.\d+)?\s*%",
    r"\b(?:results?|findings?)\s+(?:show|demonstrate|indicate)\b",
    r"\b(?:baseline|state[- ]of[- ]the[- ]art)\b",
)
METHOD_PATTERNS = (
    r"\b(?:we|our)\s+(?:propose|present|introduce|develop|train|pretrain|fine[- ]?tune|use|employ)\b",
    r"\b(?:architecture|encoder|decoder|objective|loss function|preprocess|training procedure)\b",
    r"\b(?:model|framework|method)\s+(?:consists|uses|contains|combines)\b",
)


def _flatten_sections(sections: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    flattened = []
    for section in sections:
        flattened.append(section)
        flattened.extend(_flatten_sections(section.get("subsections", [])))
    return flattened


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS
    ]


def _split_sentences(text: str) -> list[str]:
    protected = text
    placeholders = {}
    for index, abbreviation in enumerate(ABBREVIATIONS):
        marker = f"__ABBR{index}__"
        protected = protected.replace(abbreviation, marker)
        placeholders[marker] = abbreviation
    protected = re.sub(r"(?<=\d)\.(?=\d)", "__DECIMAL__", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)
    restored = []
    for part in parts:
        for marker, value in placeholders.items():
            part = part.replace(marker, value)
        part = part.replace("__DECIMAL__", ".").replace("__CITATION__", ".")
        part = re.sub(r"\s+", " ", part).strip()
        if 35 <= len(part) <= 700:
            restored.append(part)
    return restored


def _sentences(section: dict[str, object]) -> list[dict[str, object]]:
    records = section.get("page_records")
    if not isinstance(records, list) or not records:
        records = [{"page_number": section.get("page_start", 1), "text": section.get("text", "")}]
    sentences = []
    position = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        for text in _split_sentences(str(record.get("text", ""))):
            sentences.append({
                "text": text,
                "page_number": int(record.get("page_number", section.get("page_start", 1))),
                "section": section,
                "position": position,
            })
            position += 1
    return sentences


def _insight_terms(insights: dict[str, object]) -> set[str]:
    terms = set()
    for value in insights.values():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict):
                extracted = item.get("value")
                if isinstance(extracted, dict):
                    extracted = " ".join(str(part) for part in extracted.values() if part)
                if extracted:
                    terms.update(_tokens(str(extracted)))
    return terms


def _tfidf_scores(sentences: list[dict[str, object]]) -> dict[int, float]:
    """Return normalized mean TF-IDF scores for sentence-level ranking."""
    documents = [_tokens(sentence["text"]) for sentence in sentences]
    document_frequency = Counter(
        token for document in documents for token in set(document)
    )
    count = max(1, len(documents))
    scores = {}
    for index, document in enumerate(documents):
        if not document:
            scores[index] = 0.0
            continue
        term_frequency = Counter(document)
        weights = [
            (term_frequency[token] / len(document))
            * (math.log((1 + count) / (1 + document_frequency[token])) + 1)
            for token in term_frequency
        ]
        scores[index] = min(1.0, sum(weights) / max(1, len(weights)))
    return scores


def _section_weight(sentence: dict[str, object], summary_type: str) -> float:
    section = sentence["section"]
    section_id = str(section.get("section_id", ""))
    heading = str(section.get("original_heading", "")).lower()
    if summary_type == "methodology":
        weights = {"proposed_method": 1.0, "methodology": 0.98, "materials": 0.88, "experiments": 0.62}
        weight = weights.get(section_id, 0.35)
        if "background" in heading or "related work" in heading:
            weight *= 0.35
        return weight
    if summary_type == "results":
        weights = {"results": 1.0, "discussion": 0.9, "experiments": 0.52}
        weight = weights.get(section_id, 0.2)
        if "dataset" in heading or "setup" in heading:
            weight *= 0.45
        return weight
    return 1.0 if section_id in SUMMARY_SECTIONS.get(summary_type, set()) else 0.25


def _pattern_score(text: str, patterns: tuple[str, ...]) -> float:
    return min(1.0, sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in patterns) / 2)


def _boilerplate_penalty(text: str) -> float:
    citation_count = len(re.findall(r"\[\d+(?:,\s*\d+)*\]", text))
    return min(0.35, 0.18 * sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in BOILERPLATE_PATTERNS) + 0.03 * citation_count)


def _is_fragment(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
    has_verb = bool(re.search(r"\b(?:is|are|was|were|use|uses|show|shows|achieve|achieves|outperform|outperforms|train|trained|consists|propose|proposes|evaluate|evaluated|improve|improves|reaches|reports|supports)\b", text, re.IGNORECASE))
    return len(words) < 10 and not has_verb


def _base_score(sentence: dict[str, object], index: int, tfidf: dict[int, float], insight_terms: set[str], summary_type: str, total: int) -> float:
    tokens = set(_tokens(sentence["text"]))
    insight_score = len(tokens & insight_terms) / max(1, len(tokens))
    position_score = max(0.0, 1.0 - sentence["position"] * 0.06)
    result_score = _pattern_score(sentence["text"], RESULT_PATTERNS) if summary_type == "results" else 0.0
    method_score = _pattern_score(sentence["text"], METHOD_PATTERNS) if summary_type == "methodology" else 0.0
    quality_bonus = max(result_score, method_score)
    section_relevance = _section_weight(sentence, summary_type)
    if summary_type == "results":
        section_relevance *= 0.45 + 0.55 * result_score
    if summary_type == "methodology" and method_score == 0:
        section_relevance *= 0.45
    if summary_type == "methodology" and re.search(r"\b(?:dataset|benchmark|evaluation|all evaluations)\b", sentence["text"], re.IGNORECASE):
        section_relevance *= 0.55
    if summary_type == "methodology" and re.search(
        r"\b(?:previous|prior|most work|other frameworks|inspired by|background|related)\b|"
        r"\bthe [A-Z][A-Za-z0-9-]+ framework\b|for all evaluations",
        sentence["text"],
        re.IGNORECASE,
    ):
        section_relevance *= 0.45
    score = (
        0.35 * tfidf[index]
        + 0.25 * section_relevance
        + 0.15 * insight_score
        + 0.10 * position_score
        + 0.15 * quality_bonus
        - _boilerplate_penalty(sentence["text"])
    )
    if _is_fragment(sentence["text"]):
        score *= 0.2
    return max(0.0, round(score, 4))


def _similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _select(sentences: list[dict[str, object]], insights: dict[str, object], summary_type: str, limit: int) -> list[dict[str, object]]:
    if not sentences:
        return []
    unique = []
    seen_text = set()
    for sentence in sentences:
        normalized = re.sub(r"\W+", " ", sentence["text"].lower()).strip()
        if normalized not in seen_text and not _is_fragment(sentence["text"]):
            unique.append(sentence)
            seen_text.add(normalized)
    tfidf = _tfidf_scores(unique)
    terms = _insight_terms(insights)
    candidates = [
        {
            "sentence": sentence,
            "tokens": set(_tokens(sentence["text"])),
            "base": _base_score(sentence, index, tfidf, terms, summary_type, len(unique)),
        }
        for index, sentence in enumerate(unique)
    ]
    selected = []
    while candidates and len(selected) < limit:
        best = max(
            candidates,
            key=lambda item: (
                item["base"] - 0.32 * max((_similarity(item["tokens"], chosen["tokens"]) for chosen in selected), default=0.0),
                -item["sentence"]["position"],
                item["sentence"]["text"],
            ),
        )
        redundancy = max((_similarity(best["tokens"], chosen["tokens"]) for chosen in selected), default=0.0)
        final_score = max(0.0, round(best["base"] - 0.32 * redundancy, 4))
        selected.append({
            "tokens": best["tokens"],
            "base": best["base"],
            "text": best["sentence"]["text"],
            "page_number": best["sentence"]["page_number"],
            "score": final_score,
        })
        candidates.remove(best)
    return [
        {
            "text": item["text"],
            "page_number": item["page_number"],
            "score": item["score"],
        }
        for item in selected
    ]


def _role(sentence: dict[str, object]) -> str:
    section_id = str(sentence["section"].get("section_id", ""))
    text = sentence["text"]
    if section_id in {"results", "discussion"} or _pattern_score(text, RESULT_PATTERNS) >= 0.5:
        return "result"
    if section_id in {"methodology", "proposed_method", "materials"} or _pattern_score(text, METHOD_PATTERNS) >= 0.5:
        return "method"
    if re.search(r"\b(?:propose|present|show|aim|goal|objective|introduce|develop)\b", text, re.IGNORECASE):
        return "approach"
    if section_id in {"conclusion"}:
        return "conclusion"
    if section_id in {"abstract", "introduction", "background", "related_work"}:
        return "problem"
    return "other"


def _executive_summary(sentences: list[dict[str, object]], insights: dict[str, object]) -> list[dict[str, object]]:
    grouped = {}
    for sentence in sentences:
        grouped.setdefault(_role(sentence), []).append(sentence)
    selected = []
    used = set()
    for role in ("problem", "approach", "method", "result", "conclusion"):
        options = [sentence for sentence in grouped.get(role, []) if sentence["text"] not in used]
        if role == "problem":
            focused = [
                sentence for sentence in options
                if str(sentence["section"].get("section_id", "")) in {"introduction", "background"}
                and re.search(r"\b(?:challenge|limited|sparse|lack|however|problem|need)\w*\b", sentence["text"], re.IGNORECASE)
            ]
            options = focused or options
        elif role == "approach":
            focused = [
                sentence for sentence in options
                if str(sentence["section"].get("section_id", "")) in {"abstract", "proposed_method", "methodology"}
            ]
            options = focused or options
        elif role == "result":
            focused = [sentence for sentence in options if _pattern_score(sentence["text"], RESULT_PATTERNS) >= 0.5]
            options = focused or options
        if role == "conclusion":
            preferred = [
                sentence for sentence in options
                if not re.search(r"\b(?:limitation|future|will explore)\b", sentence["text"], re.IGNORECASE)
            ]
            options = preferred or options
        chosen = _select(options, insights, "executive", 1)
        if chosen:
            selected.extend(chosen)
            used.add(chosen[0]["text"])
    if len(selected) < 5:
        remaining = [sentence for sentence in sentences if sentence["text"] not in used]
        selected.extend(_select(remaining, insights, "executive", 5 - len(selected)))
    return selected[:5]


def generate_summaries(sections: Iterable[dict[str, object]], insights: dict[str, object] | None = None) -> dict[str, dict[str, object]]:
    """Create deterministic, extractive summaries using sentence-level TF-IDF."""
    all_sections = _flatten_sections(sections)
    insights = insights or {}
    summaries = {}
    for summary_type, section_ids in SUMMARY_SECTIONS.items():
        relevant = [section for section in all_sections if section["section_id"] in section_ids]
        sentences = [sentence for section in relevant for sentence in _sentences(section)]
        chosen = _select(sentences, insights, summary_type, 3)
        summaries[summary_type] = (
            {"summary_type": summary_type, "available": True, "sentences": chosen}
            if chosen
            else {"summary_type": summary_type, "available": False, "sentences": [], "message": "Unavailable from the detected paper sections."}
        )

    executive_sections = [
        section for section in all_sections
        if section["section_id"] in {"abstract", "introduction", "methodology", "proposed_method", "materials", "experiments", "results", "discussion", "conclusion"}
    ]
    executive_sentences = [sentence for section in executive_sections for sentence in _sentences(section)]
    executive = _executive_summary(executive_sentences, insights)
    summaries["executive"] = (
        {"summary_type": "executive", "available": True, "sentences": executive}
        if executive
        else {"summary_type": "executive", "available": False, "sentences": [], "message": "Unavailable from the detected paper sections."}
    )
    return summaries
