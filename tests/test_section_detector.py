import unittest

from app.services.section_detector import detect_sections


class SectionDetectorTests(unittest.TestCase):
    def detect(self, text_pages):
        return detect_sections(
            [{"page_number": number, "text": text} for number, text in text_pages]
        )

    def test_standard_headings(self):
        result = self.detect([(1, "Abstract\nSummary\nIntroduction\nOpening")])
        self.assertEqual([item["section_id"] for item in result], ["abstract", "introduction"])

    def test_inline_front_matter_abstract_is_detected(self):
        result = self.detect([(
            1,
            "Paper title Authors Abstract: This study evaluates a transparent model on a public benchmark. "
            "Keywords: research, evaluation 1 Introduction The paper begins here.",
        )])
        abstract = next(item for item in result if item["section_id"] == "abstract")
        self.assertIn("This study evaluates", abstract["text"])
        self.assertNotIn("Paper title Authors", abstract["text"])
        self.assertNotIn("Keywords:", abstract["text"])

    def test_numbered_and_roman_headings(self):
        result = self.detect([(1, "1 Introduction\nText\nII. Methodology\nMethods")])
        self.assertEqual([item["section_id"] for item in result], ["introduction", "methodology"])

    def test_alternative_terminology(self):
        result = self.detect([(1, "Literature Review\nText\nExperimental Results\nFindings")])
        self.assertEqual([item["section_id"] for item in result], ["related_work", "results"])

    def test_false_positive_in_normal_paragraph(self):
        result = self.detect([(1, "This introduction describes results and the conclusion follows later.")])
        self.assertEqual(result, [])

    def test_subsections_remain_distinguishable(self):
        result = self.detect([(1, "3 Methodology\nText\n3.1 Dataset\nData\n3.2 Model Architecture\nModel")])
        self.assertEqual([item["section_id"] for item in result], ["methodology"])
        self.assertEqual([item["section_id"] for item in result[0]["subsections"]], ["dataset", "other"])
        self.assertEqual(result[0]["subsections"][0]["level"], "3.1")

    def test_major_section_hierarchy_and_parent(self):
        result = self.detect([(1, "3 Experiments\n3.1 Dataset\nData\n3.2 Results\nFindings")])
        self.assertTrue(result[0]["is_major"])
        self.assertEqual(result[0]["subsections"][0]["parent_section_id"], "experiments")

    def test_document_order_and_page_ranges(self):
        result = self.detect([(2, "2 Methods\nText"), (1, "1 Introduction\nIntro"), (3, "3 Results\nFindings")])
        self.assertEqual([item["section_id"] for item in result], ["methodology", "introduction", "results"])
        self.assertEqual(result[0]["page_start"], 2)
        self.assertEqual(result[0]["page_end"], 2)

    def test_missing_sections_are_allowed(self):
        result = self.detect([(1, "Abstract\nOnly an abstract.")])
        self.assertEqual(len(result), 1)
        self.assertNotIn("methodology", [item["section_id"] for item in result])

    def test_page_references_are_preserved(self):
        result = self.detect([(1, "Introduction\nFirst"), (2, "Results\nSecond")])
        sections = {item["section_id"]: item for item in result}
        self.assertEqual(sections["introduction"]["page_start"], 1)
        self.assertEqual(sections["introduction"]["page_end"], 1)
        self.assertEqual(sections["results"]["page_start"], 2)

    def test_unknown_heading_is_other(self):
        result = self.detect([(1, "1 Novel Perspective\nText")])
        self.assertEqual(result[0]["section_id"], "other")

    def test_confidence_scores_are_present(self):
        result = self.detect([(1, "1 Introduction\nText")])
        self.assertGreaterEqual(result[-1]["confidence"], 0.8)
        self.assertLessEqual(result[-1]["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
