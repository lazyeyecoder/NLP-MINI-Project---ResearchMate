import re
from collections.abc import Iterable


FIELD_SECTIONS = {
    "problem": {"abstract", "introduction", "background", "related_work"},
    "objective": {"abstract", "introduction", "proposed_method", "conclusion"},
    "methodology": {"methodology", "proposed_method", "materials", "experiments"},
    "experimental_setup": {"materials", "experiments", "methodology"},
    "dataset": {"dataset", "experiments", "results"},
    "limitations": {"limitations", "discussion", "conclusion"},
    "future_work": {"future_work", "conclusion", "discussion"},
}
OBJECTIVE_PRIORITY = {
    "abstract": 1.0,
    "introduction": 0.95,
    "objective": 1.0,
    "research_aim": 1.0,
    "proposed_method": 0.9,
    "methodology": 0.85,
    "conclusion": 0.55,
}

_CUES = {
    "problem": (r"\bhowever\b", r"\bchallenge(?:s)?\b", r"\bproblem\b", r"\black(?:s|ing)?\b", r"\bneed(?:s)?\b"),
    "objective": (r"\bwe (?:propose|present|introduce|develop|aim|seek)\b", r"\bthis (?:paper|study|work) (?:aims?|presents?|proposes?)\b", r"\b(?:objective|goal|purpose)\b"),
    "methodology": (r"\bwe (?:use|employ|train|implement|conduct)\b", r"\bour method\b", r"\bmethodology\b"),
    "experimental_setup": (r"\bwe (?:pretrain|finetune|fine-tune|evaluate|compare)\b", r"\bexperimental setup\b", r"\bexperiments?\b"),
    "limitations": (r"\blimitations?\b", r"\bhowever\b", r"\bmay not\b", r"\bdoes not\b"),
    "future_work": (r"\bfuture (?:work|research|directions?)\b", r"\bwe leave\b", r"\bshould be\b", r"\bwill explore\b"),
}

MODEL_TERMS = (
    "transformer", "bert", "cnn", "lstm", "vit", "autoencoder", "masked autoencoder",
    "contrastive learning", "self-supervised learning", "text rank", "random forest",
    "support vector machine", "svm", "neural network", "k-means", "mae", "dino",
)
DATASET_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,3}\s+dataset)\b"
    r"|\b([A-Za-z][A-Za-z0-9-]*-[A-Z][A-Za-z0-9-]*\s+dataset)\b"
    r"|\b([A-Z][A-Za-z0-9-]{2,}(?:Net|SAT|World|2020|EO))\b"
)
SOURCE_CONTEXT = re.compile(
    r"\b(sensor|satellite|platform|instrument|radar|optical|imagery|bands?|channels?|"
    r"remote sensing|data source|acquired by)\b",
    re.IGNORECASE,
)
SOURCE_NAME_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z0-9]+)+\b|\b[A-Z][A-Za-z]+-\d+\b")
METRIC_PATTERN = re.compile(
    r"\b(accuracy|precision|recall|f1(?:[- ]score)?|mAP|mIoU|RMSE|MAE|BLEU|ROUGE|"
    r"top[- ]1 accuracy|mean absolute error|mean squared error)\b(?:\s*(?:of|:|=)\s*)?([0-9]+(?:\.[0-9]+)?%?)?",
    re.IGNORECASE,
)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _flatten_sections(sections: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    flattened = []
    for section in sections:
        flattened.append(section)
        flattened.extend(_flatten_sections(section.get("subsections", [])))
    return flattened


def _sentences(section: dict[str, object]) -> list[dict[str, object]]:
    text = re.sub(r"\s+", " ", str(section.get("text", "")).strip())
    return [
        {"text": sentence.strip(), "section": section}
        for sentence in SENTENCE_PATTERN.split(text)
        if len(sentence.strip()) >= 25
    ]


def _evidence(sentence: dict[str, object], confidence: float) -> dict[str, object]:
    section = sentence["section"]
    value = str(sentence["text"]).strip()
    return {
        "value": value,
        "source_section": section["section_id"],
        "page_numbers": list(range(int(section["page_start"]), int(section["page_end"]) + 1)),
        "evidence": value[:400],
        "confidence": round(min(confidence, 0.99), 2),
    }


def _cue_score(field: str, sentence: str) -> float:
    return sum(bool(re.search(pattern, sentence, re.IGNORECASE)) for pattern in _CUES.get(field, ()))


def _best_sentence(field: str, sections: list[dict[str, object]]) -> dict[str, object] | None:
    candidates = []
    for section in sections:
        if section["section_id"] not in FIELD_SECTIONS[field]:
            continue
        for position, sentence in enumerate(_sentences(section)):
            cue_score = _cue_score(field, sentence["text"])
            if cue_score:
                section_score = 0.18 if section["section_id"] in FIELD_SECTIONS[field] else 0
                position_score = 0.08 if position < 3 else 0
                candidates.append((cue_score + section_score + position_score, sentence))
    if not candidates:
        return None
    score, sentence = max(
        candidates,
        key=lambda item: item[0] * OBJECTIVE_PRIORITY.get(item[1]["section"]["section_id"], 0.75)
        if field == "objective"
        else item[0],
    )
    section_weight = OBJECTIVE_PRIORITY.get(sentence["section"]["section_id"], 0.75) if field == "objective" else 1
    base = 0.55 + min(score * 0.08, 0.35)
    return _evidence(sentence, base * section_weight)


def _term_list(sections: list[dict[str, object]], terms: Iterable[str], field: str) -> list[dict[str, object]]:
    results = []
    seen = set()
    ordered_terms = sorted(terms, key=len, reverse=True)
    for section in sections:
        allowed_sections = FIELD_SECTIONS.get(field, set())
        if field == "models":
            allowed_sections = {"methodology", "proposed_method", "experiments", "results"}
        if section["section_id"] not in allowed_sections:
            continue
        text = str(section.get("text", ""))
        for term in ordered_terms:
            if any(term.lower() in longer.lower() for longer in ordered_terms if len(longer) > len(term) and re.search(rf"(?<!\w){re.escape(longer)}(?!\w)", text, re.IGNORECASE)):
                continue
            match = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE)
            if match and term.lower() not in seen:
                sentence = next((item for item in _sentences(section) if match.group(0).lower() in item["text"].lower()), None)
                if term.lower() == "mae" and sentence and not _supports_model_mae(sentence["text"]):
                    continue
                seen.add(term.lower())
                if sentence:
                    item = _evidence(sentence, 0.7)
                    item["value"] = match.group(0)
                    results.append(item)
    return results


def _datasets(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    results = []
    seen = set()
    for section in sections:
        if section["section_id"] not in FIELD_SECTIONS["dataset"]:
            continue
        for match in DATASET_PATTERN.finditer(str(section.get("text", ""))):
            value = next(group for group in match.groups() if group)
            value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
            sentence = next((item for item in _sentences(section) if value.lower() in item["text"].lower()), None)
            if not sentence:
                continue
            if SOURCE_CONTEXT.search(sentence["text"]) and not re.search(r"\bdataset\b|\bbenchmark\b", sentence["text"], re.IGNORECASE):
                continue
            normalized_value = re.sub(r"\s+dataset$", "", value, flags=re.IGNORECASE).lower()
            if normalized_value in {"this", "the", "data"} or normalized_value in seen:
                continue
            seen.add(normalized_value)
            item = _evidence(sentence, 0.76)
            item["value"] = value
            results.append(item)
    return results


def _data_sources(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    results = []
    seen = set()
    for section in sections:
        if section["section_id"] not in FIELD_SECTIONS["dataset"]:
            continue
        for sentence in _sentences(section):
            if not SOURCE_CONTEXT.search(sentence["text"]):
                continue
            for match in SOURCE_NAME_PATTERN.finditer(sentence["text"]):
                value = match.group(0)
                if value.lower() in seen:
                    continue
                context_start = max(0, match.start() - 70)
                context_end = min(len(sentence["text"]), match.end() + 70)
                if not SOURCE_CONTEXT.search(sentence["text"][context_start:context_end]):
                    continue
                if value.lower() in {"i-jepa", "vit-b", "satvit-v2"}:
                    continue
                seen.add(value.lower())
                item = _evidence(sentence, 0.68)
                item["value"] = value
                results.append(item)
    return results


def _supports_model_mae(text: str) -> bool:
    return bool(re.search(r"\bmasked autoencoder\b|\bMAE (?:model|framework|pretraining|objective)\b", text, re.IGNORECASE))


def _supports_metric_mae(text: str) -> bool:
    return bool(re.search(r"\bmean absolute error\b|\bMAE\b\s*(?:of|=|:)\s*[0-9]", text, re.IGNORECASE))


def _metrics(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    results = []
    seen = set()
    for section in sections:
        if section["section_id"] not in {"results", "experiments", "discussion"}:
            continue
        for sentence in _sentences(section):
            for match in METRIC_PATTERN.finditer(sentence["text"]):
                metric = match.group(1)
                value = match.group(2)
                if metric.lower() == "mean absolute error":
                    metric = "MAE"
                if metric.lower() == "mae" and not _supports_metric_mae(sentence["text"]):
                    continue
                key = f"{metric.lower()}:{value or ''}"
                if key in seen:
                    continue
                seen.add(key)
                item = _evidence(sentence, 0.78 if value else 0.65)
                item["value"] = {"metric": metric, "value": value}
                results.append(item)
    return results


def extract_research_insights(sections: Iterable[dict[str, object]]) -> dict[str, object]:
    """Extract supported research insights from the existing detected sections."""
    flat_sections = _flatten_sections(sections)
    return {
        "research_problem": _best_sentence("problem", flat_sections),
        "research_objective": _best_sentence("objective", flat_sections),
        "methodology": _best_sentence("methodology", flat_sections),
        "experimental_setup": _best_sentence("experimental_setup", flat_sections),
        "models_algorithms": _term_list(flat_sections, MODEL_TERMS, "models"),
        "datasets": _datasets(flat_sections),
        "data_sources": _data_sources(flat_sections),
        "results_metrics": _metrics(flat_sections),
        "limitations": _best_sentence("limitations", flat_sections),
        "future_work": _best_sentence("future_work", flat_sections),
    }
