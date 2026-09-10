import re
from collections import Counter


def normalize_whitespace(text: str) -> str:
    """Normalize spacing without collapsing paragraph or sentence boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def repair_line_breaks(text: str) -> str:
    """Join wrapped lines while retaining blank lines between paragraphs."""
    text = normalize_whitespace(text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def _repeated_margin_lines(raw_pages: list[dict[str, object]]) -> set[str]:
    candidates = []
    for page in raw_pages:
        lines = [line.strip() for line in str(page["text"]).splitlines() if line.strip()]
        if len(lines) >= 2:
            candidates.extend((lines[0], lines[-1]))

    counts = Counter(line for line in candidates if line)
    threshold = max(2, len(raw_pages) // 2)
    return {line for line, count in counts.items() if count >= threshold}


def preprocess_pages(raw_pages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Clean pages conservatively and retain page numbers for later evidence links."""
    repeated_lines = _repeated_margin_lines(raw_pages)
    processed = []

    for page in raw_pages:
        lines = str(page["text"]).replace("\r\n", "\n").replace("\r", "\n").splitlines()
        while lines and lines[0].strip() in repeated_lines:
            lines.pop(0)
        while lines and lines[-1].strip() in repeated_lines:
            lines.pop()

        page_text = repair_line_breaks("\n".join(lines))
        processed.append({"page_number": page["page_number"], "text": page_text})

    return processed
