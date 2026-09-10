import unittest
import importlib.util
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pymupdf

from app.services.metadata_extractor import extract_metadata


class MetadataExtractorTests(unittest.TestCase):
    def test_extracts_basic_metadata_with_provenance(self):
        pages = [{
            "page_number": 1,
            "text": (
                "Transparent Models for Research Papers\n"
                "John Smith, Jane Doe\n"
                "Department of Computer Science, Example University\n"
                "Published 2024\n"
                "Abstract: This paper presents a transparent method for research analysis.\n"
                "Keywords: research analysis, transparency, NLP\n"
                "1 Introduction\n"
            ),
        }]
        result = extract_metadata(pages, [])
        self.assertEqual(result["title"]["value"], "Transparent Models for Research Papers")
        self.assertEqual(result["authors"]["value"], ["John Smith", "Jane Doe"])
        self.assertEqual(result["publication_year"]["value"], 2024)
        self.assertIn("transparent method", result["abstract"]["value"])
        self.assertEqual(result["keywords"]["value"], ["research analysis", "transparency", "NLP"])
        self.assertEqual(result["title"]["page_number"], 1)
        self.assertIn("evidence", result["abstract"])

    def test_reuses_detected_abstract_section(self):
        pages = [{"page_number": 2, "text": "Title\nAuthor\nAbstract: fallback text"}]
        sections = [{"section_id": "abstract", "page_start": 3, "text": "The detected abstract text.", "subsections": []}]
        result = extract_metadata(pages, sections)
        self.assertEqual(result["abstract"]["value"], "The detected abstract text.")
        self.assertEqual(result["abstract"]["page_number"], 3)

    def test_missing_metadata_is_none(self):
        result = extract_metadata([{"page_number": 1, "text": "A short document without metadata."}], [])
        self.assertIsNone(result["authors"])
        self.assertIsNone(result["publication_year"])
        self.assertIsNone(result["keywords"])

    def test_publication_year_uses_front_matter_not_reference_years(self):
        pages = [{
            "page_number": 1,
            "text": (
                "Foundation Models for Remote Sensing and Earth Observation: A Survey\n"
                "Aoran Xiao, Weihao Xuan\n"
                "arXiv preprint, 2025\n"
                "Abstract: We review foundation models.\n"
                "References\n"
                "[1] Earlier work, 2023.\n"
            ),
        }]
        result = extract_metadata(pages, [])
        self.assertEqual(result["publication_year"]["value"], 2025)

    def test_flask_pipeline_passes_metadata_to_results(self):
        spec = importlib.util.spec_from_file_location("researchmate_flask", "app.py")
        application = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(application)

        document = pymupdf.open()
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "A Practical NLP Paper\nJohn Smith\nAbstract: We present a practical method.\nKeywords: NLP, research",
        )
        pdf_bytes = document.tobytes()
        document.close()
        with TemporaryDirectory() as directory, patch.object(application, "UPLOAD_DIR", Path(directory)):
            client = application.create_app().test_client()
            response = client.post(
                "/upload",
                data={"paper": (BytesIO(pdf_bytes), "metadata-test.pdf")},
                content_type="multipart/form-data",
            )
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("A Practical NLP Paper", html)
        self.assertIn("John Smith", html)


if __name__ == "__main__":
    unittest.main()
