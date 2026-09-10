import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from app.services.pdf_extractor import extract_pdf_text
from app.services.text_preprocessor import (
    normalize_whitespace,
    preprocess_pages,
    repair_line_breaks,
)


class TextPreprocessorTests(unittest.TestCase):
    def test_whitespace_normalization(self):
        self.assertEqual(normalize_whitespace("  Alpha \t beta \n\n Gamma  "), "Alpha beta\n\nGamma")

    def test_hyphenated_line_repair(self):
        self.assertEqual(repair_line_breaks("state-\nof-the-art\nmethod"), "stateof-the-art method")

    def test_repeated_header_and_footer_removal(self):
        pages = [
            {"page_number": 1, "text": "ResearchMate Journal\nBody one\nPage footer"},
            {"page_number": 2, "text": "ResearchMate Journal\nBody two\nPage footer"},
        ]
        result = preprocess_pages(pages)
        self.assertEqual(result[0]["text"], "Body one")
        self.assertEqual(result[1]["text"], "Body two")


class PdfExtractionTests(unittest.TestCase):
    def create_pdf(self, path: Path, page_texts: list[str]):
        document = pymupdf.open()
        for page_text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), page_text)
        document.save(path)
        document.close()

    def test_multi_page_extraction_preserves_pages(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            self.create_pdf(path, ["First page", "Second page"])
            result = extract_pdf_text(path)
        self.assertEqual(result["page_count"], 2)
        self.assertEqual([page["page_number"] for page in result["pages"]], [1, 2])
        self.assertIn("[Page 2]\nSecond page", result["text"])

    def test_empty_pdf_is_detected_as_scanned(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.pdf"
            document = pymupdf.open()
            document.new_page()
            document.save(path)
            document.close()
            result = extract_pdf_text(path)
        self.assertEqual(result["extraction_status"], "scanned")
        self.assertFalse(result["processed"])
        self.assertEqual(result["text"], "")


if __name__ == "__main__":
    unittest.main()
