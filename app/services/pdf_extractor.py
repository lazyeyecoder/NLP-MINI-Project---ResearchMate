from pathlib import Path

import pymupdf as fitz

from app.services.text_preprocessor import preprocess_pages


def extract_pdf_text(pdf_path: Path) -> dict[str, object]:
    """Extract and preprocess text while retaining the source page for each result."""
    try:
        pdf_data = pdf_path.read_bytes()
        document = fitz.open(stream=pdf_data, filetype="pdf")
    except (OSError, RuntimeError, fitz.FileDataError) as error:
        raise ValueError("The uploaded file could not be opened as a PDF.") from error

    with document:
        if document.page_count == 0:
            raise ValueError("The uploaded PDF does not contain any pages.")
        document_metadata = document.metadata or {}

        raw_pages = [
            {"page_number": page_number, "text": page.get_text("text")}
            for page_number, page in enumerate(document, start=1)
        ]

    extracted_characters = sum(len(page["text"].strip()) for page in raw_pages)
    if extracted_characters == 0:
        return {
            "page_count": len(raw_pages),
            "pages": [],
            "text": "",
            "text_length": 0,
            "processed": False,
            "extraction_status": "scanned",
            "document_metadata": document_metadata,
        }

    processed_pages = preprocess_pages(raw_pages)
    processed_text = "\n\n".join(
        f"[Page {page['page_number']}]\n{page['text']}"
        for page in processed_pages
        if page["text"]
    )
    return {
        "page_count": len(raw_pages),
        "pages": processed_pages,
        "text": processed_text,
        "text_length": len(processed_text),
        "processed": True,
        "extraction_status": "text",
        "document_metadata": document_metadata,
    }
