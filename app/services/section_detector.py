import re
from collections.abc import Iterable


SECTION_ALIASES = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "introductory remarks"),
    "background": ("background",),
    "related_work": ("related work", "literature review", "prior work"),
    "methodology": ("methodology", "methods", "method"),
    "proposed_method": ("proposed method", "proposed approach", "our approach"),
    "materials": ("materials", "experimental setup", "experimental design"),
    "dataset": ("dataset", "data", "data collection"),
    "experiments": ("experiments", "experimental study"),
    "results": ("results", "experimental results", "evaluation", "findings"),
    "discussion": ("discussion",),
    "limitations": ("limitations", "limitations and threats to validity"),
    "conclusion": ("conclusion", "conclusions"),
    "future_work": ("future work", "future research", "future directions"),
    "acknowledgements": ("acknowledgements", "acknowledgments"),
    "references": ("references", "bibliography"),
}

DISPLAY_NAMES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related_work": "Related Work",
    "methodology": "Methodology",
    "proposed_method": "Proposed Method",
    "materials": "Materials / Experimental Setup",
    "dataset": "Dataset / Data",
    "experiments": "Experiments",
    "results": "Results / Evaluation",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "future_work": "Future Work",
    "acknowledgements": "Acknowledgements",
    "references": "References",
}

_NUMBERING = r"(?:(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+)?"
_ALIASES = sorted(
    (alias, section_id)
    for section_id, aliases in SECTION_ALIASES.items()
    for alias in aliases
)
_ALIAS_PATTERN = "|".join(re.escape(alias) for alias, _ in sorted(_ALIASES, reverse=True))
_HEADING_PATTERN = re.compile(
    rf"(?<![\w])(?P<heading>{_NUMBERING}(?:{_ALIAS_PATTERN}))(?=$|[\s:.-])",
    re.IGNORECASE,
)
_NUMBERED_GENERIC_PATTERN = re.compile(
    r"(?m)^(?P<heading>(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+[A-Z][A-Za-z0-9/& -]{1,70})$"
)


def _classify(heading: str) -> tuple[str, float]:
    without_number = re.sub(r"^\s*(?:(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+)", "", heading)
    normalized = re.sub(r"\s+", " ", without_number).strip().lower().rstrip(".:")
    for alias, section_id in _ALIASES:
        if normalized == alias:
            numbered = bool(re.match(r"^\s*(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+", heading))
            uppercase = without_number.isupper()
            return section_id, min(0.99, 0.84 + (0.08 if numbered else 0) + (0.04 if uppercase else 0))
    return "other", 0.45


def _is_candidate(match: re.Match[str], text: str) -> bool:
    heading = match.group("heading")
    numbered = bool(re.match(r"^\s*(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+", heading))
    words = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+", "", heading)
    if words.strip().lower() in {"method", "data"}:
        return numbered
    left_context_raw = text[: match.start()]
    left_context = left_context_raw.rstrip()
    line_boundary = "\n" in left_context_raw[len(left_context):]
    return numbered or words.isupper() or words[:1].isupper() and (
        not left_context or left_context[-1] in "\n.:[(" or line_boundary
    )


def _is_numbered(heading: str) -> bool:
    return bool(re.match(r"^\s*(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+", heading))


def _heading_level(heading: str) -> int:
    match = re.match(r"^\s*(\d+(?:\.\d+)*|[IVX]+)[.)]?\s+", heading)
    if not match:
        return 0
    marker = match.group(1)
    return marker.count(".") + 1 if marker[0].isdigit() else 1


def _parent_number(heading: str) -> str | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)+)[.)]?\s+", heading)
    return match.group(1).rsplit(".", 1)[0] if match else None


def _section_number(heading: str) -> str | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*|[IVX]+)[.)]?\s+", heading)
    return match.group(1) if match else None


def _heading_matches(text: str) -> list[re.Match[str]]:
    matches = [match for match in _HEADING_PATTERN.finditer(text) if _is_candidate(match, text)]
    matches += [match for match in _NUMBERED_GENERIC_PATTERN.finditer(text) if _is_candidate(match, text)]
    matches.sort(key=lambda match: (match.start(), -len(match.group("heading"))))
    selected = []
    for match in matches:
        if not selected or match.start() >= selected[-1].end():
            selected.append(match)
    return selected


def detect_sections(pages: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Detect academic sections from preprocessed, page-structured text."""
    page_list = list(pages)
    segments = []
    for page in page_list:
        page_number = int(page["page_number"])
        text = str(page.get("text", "")).strip()
        matches = _heading_matches(text)
        if not matches:
            segments.append({"page_number": page_number, "text": text})
            continue

        for index, match in enumerate(matches):
            previous_end = matches[index - 1].end() if index else 0
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            prefix = text[previous_end:match.start()].strip()
            if prefix:
                segments.append({"page_number": page_number, "text": prefix})
            segments.append(
                {
                    "page_number": page_number,
                    "heading": match.group("heading").strip(),
                    "heading_end": match.end(),
                    "text": text[match.end():next_start].strip(),
                }
            )

    detected = []
    for index, segment in enumerate(segments):
        if "heading" not in segment:
            continue
        section_id, confidence = _classify(str(segment["heading"]))
        next_heading = next(
            (
                candidate_index
                for candidate_index in range(index + 1, len(segments))
                if segments[candidate_index].get("heading")
            ),
            len(segments),
        )
        content_segments = [
            candidate
            for candidate in segments[index:next_heading]
            if str(candidate.get("text", "")).strip()
        ]
        page_records = [
            {
                "page_number": int(candidate["page_number"]),
                "text": str(candidate["text"]).strip(),
            }
            for candidate in content_segments
        ]
        section_text = "\n".join(record["text"] for record in page_records)
        detected.append(
            {
                "section_id": section_id,
                "title": DISPLAY_NAMES.get(section_id, "Other"),
                "original_heading": segment["heading"],
                "page_start": segment["page_number"],
                "page_end": int(page_records[-1]["page_number"]) if page_records else int(segment["page_number"]),
                "text": section_text,
                "page_records": page_records,
                "confidence": round(confidence, 2),
                "level": str(segment["heading"]).split()[0] if _is_numbered(str(segment["heading"])) else "top-level",
                "hierarchy_level": _heading_level(str(segment["heading"])),
                "parent_number": _parent_number(str(segment["heading"])),
                "section_number": _section_number(str(segment["heading"])),
            }
        )

    deduplicated = []
    for section in detected:
        if (
            deduplicated
            and section["section_id"] == deduplicated[-1]["section_id"]
            and section["page_start"] == deduplicated[-1]["page_start"]
            and not _is_numbered(str(section["original_heading"]))
        ):
            continue
        deduplicated.append(section)
    detected = deduplicated

    if page_list:
        first_page_text = str(page_list[0].get("text", ""))
        first_heading_position = min(
            (
                first_page_text.find(str(item["original_heading"]))
                for item in detected
                if item["page_start"] == int(page_list[0]["page_number"])
                and first_page_text.find(str(item["original_heading"])) >= 0
            ),
            default=len(first_page_text),
        )
        abstract_match = re.search(r"(?i)\babstract\b\s*:?", first_page_text[:first_heading_position])
        if abstract_match and not any(item["section_id"] == "abstract" for item in detected):
            abstract_text = first_page_text[abstract_match.end():first_heading_position].strip()
            abstract_text = re.split(r"(?i)\bkeywords?\s*:", abstract_text, maxsplit=1)[0].strip()
            if abstract_text:
                detected.insert(
                    0,
                    {
                        "section_id": "abstract",
                        "title": "Abstract",
                        "original_heading": abstract_match.group(0).strip(),
                        "page_start": int(page_list[0]["page_number"]),
                        "page_end": int(page_list[0]["page_number"]),
                        "text": abstract_text,
                        "page_records": [{"page_number": int(page_list[0]["page_number"]), "text": abstract_text}],
                        "confidence": 0.9,
                        "level": "front-matter",
                        "hierarchy_level": 0,
                        "parent_number": None,
                        "section_number": None,
                    },
                )
        first_heading = next((item for item in detected if item["page_start"] == int(page_list[0]["page_number"])), None)
        first_text = str(page_list[0].get("text", ""))
        if first_heading:
            prefix = first_text[: first_text.find(str(first_heading["original_heading"]))].strip()
            if prefix:
                detected.insert(
                    0,
                    {
                        "section_id": "title_front_matter",
                        "title": "Title / Front Matter",
                        "original_heading": prefix.splitlines()[0],
                        "page_start": int(page_list[0]["page_number"]),
                        "page_end": first_heading["page_start"],
                        "text": prefix,
                        "page_records": [{"page_number": int(page_list[0]["page_number"]), "text": prefix}],
                        "confidence": 0.72,
                        "level": "front-matter",
                        "hierarchy_level": 0,
                        "parent_number": None,
                        "section_number": None,
                    },
                )

    major_sections = []
    current_major = None
    for section_index, section in enumerate(detected):
        parent = next(
            (
                candidate
                for candidate in reversed(major_sections)
                if section["parent_number"]
                and candidate.get("section_number") == section["parent_number"]
                and not any(
                    item.get("hierarchy_level", 0) <= 1
                    for item in detected[detected.index(candidate) + 1 : section_index]
                )
            ),
            None,
        )
        if section["hierarchy_level"] <= 1 or parent is None:
            section["subsections"] = []
            section["is_major"] = True
            major_sections.append(section)
            current_major = section
        else:
            section["is_major"] = False
            section["parent_section_id"] = parent["section_id"]
            section["subsections"] = []
            parent["subsections"].append(section)

    for index, section in enumerate(major_sections):
        next_major = major_sections[index + 1] if index + 1 < len(major_sections) else None
        section["page_end"] = (
            max(section["page_start"], next_major["page_start"] - 1)
            if next_major
            else int(page_list[-1]["page_number"])
        )
        for child_index, child in enumerate(section["subsections"]):
            next_child = section["subsections"][child_index + 1] if child_index + 1 < len(section["subsections"]) else None
            child["page_end"] = (
                max(child["page_start"], next_child["page_start"] - 1)
                if next_child
                else section["page_end"]
            )
    return major_sections
