import unittest

from app.services.summarizer import _tfidf_scores, generate_summaries


def section(section_id, text, page=1):
    return {"section_id": section_id, "text": text, "page_start": page, "page_end": page, "subsections": []}


def paged_section(section_id, records):
    return {
        "section_id": section_id,
        "text": "\n".join(record["text"] for record in records),
        "page_start": records[0]["page_number"],
        "page_end": records[-1]["page_number"],
        "page_records": records,
        "subsections": [],
    }


class SummarizerTests(unittest.TestCase):
    def test_extractive_behavior_and_provenance(self):
        text = "We propose a transparent classifier for scientific documents. The classifier uses lexical features and a linear model. It improves reproducibility for student researchers."
        result = generate_summaries([section("methodology", text, 3)])
        sentences = result["methodology"]["sentences"]
        self.assertTrue(sentences)
        self.assertTrue(all(sentence["text"] in text for sentence in sentences))
        self.assertEqual(sentences[0]["page_number"], 3)

    def test_section_aware_summary(self):
        result = generate_summaries([
            section("methodology", "We train a transformer model with a reproducible optimization procedure.", 2),
            section("results", "Accuracy reaches 94 percent on the held-out evaluation set.", 5),
        ])
        self.assertEqual(result["methodology"]["sentences"][0]["page_number"], 2)
        self.assertEqual(result["results"]["sentences"][0]["page_number"], 5)

    def test_sentence_page_provenance_is_preserved(self):
        result = generate_summaries([paged_section("results", [
            {"page_number": 4, "text": "The model improves accuracy on the validation benchmark by a substantial margin."},
            {"page_number": 5, "text": "The results show that the approach outperforms the baseline across all evaluation tasks."},
        ])])
        pages = {sentence["page_number"] for sentence in result["results"]["sentences"]}
        self.assertEqual(pages, {4, 5})

    def test_tfidf_scores_rare_terms_more_highly(self):
        sentences = [
            {"text": "The model uses common features."},
            {"text": "The model uses geospatial features."},
            {"text": "The model uses common features."},
        ]
        scores = _tfidf_scores(sentences)
        self.assertGreater(scores[1], scores[0])

    def test_methodology_prefers_method_over_background(self):
        result = generate_summaries([
            section("methodology", "Previous studies have explored related architectures in the literature for many years.", 2),
            section("methodology", "We propose a masked encoder architecture and train it with a contrastive reconstruction objective.", 3),
        ])
        self.assertIn("propose", result["methodology"]["sentences"][0]["text"].lower())

    def test_results_prefers_findings_over_setup(self):
        result = generate_summaries([
            section("experiments", "We use three public datasets and train the model for 100 epochs with a fixed learning rate.", 4),
            section("results", "Our method achieves 94 percent accuracy and outperforms the baseline on the held-out test set.", 5),
        ])
        self.assertIn("achieves", result["results"]["sentences"][0]["text"].lower())

    def test_executive_summary_covers_multiple_roles(self):
        result = generate_summaries([
            section("introduction", "Existing methods struggle with sparse labels and limited generalization in remote sensing.", 1),
            section("abstract", "We propose a multimodal representation framework for this research problem.", 2),
            section("methodology", "We train two encoders with complementary contrastive and reconstruction objectives.", 3),
            section("results", "The method achieves 92 percent accuracy and outperforms the strongest baseline.", 4),
            section("conclusion", "These findings support broader use of the representations in downstream applications.", 5),
        ])
        executive_text = " ".join(sentence["text"].lower() for sentence in result["executive"]["sentences"])
        self.assertIn("struggle", executive_text)
        self.assertIn("propose", executive_text)
        self.assertIn("achieves", executive_text)

    def test_redundancy_is_removed(self):
        result = generate_summaries([
            section("abstract", "The proposed model improves classification accuracy on scientific documents. The proposed model improves classification accuracy on scientific documents. A separate analysis reports stable performance across domains.", 1)
        ])
        texts = [sentence["text"] for sentence in result["abstract"]["sentences"]]
        self.assertEqual(len(texts), 2)

    def test_missing_section_is_explicitly_unavailable(self):
        result = generate_summaries([section("introduction", "This paper studies document analysis for academic applications.", 1)])
        self.assertFalse(result["abstract"]["available"])
        self.assertEqual(result["abstract"]["sentences"], [])

    def test_deterministic_output(self):
        sections = [section("conclusion", "We find that the method is robust in controlled experiments. The result supports practical use in academic settings.", 8)]
        self.assertEqual(generate_summaries(sections), generate_summaries(sections))

    def test_executive_summary_uses_relevant_sections(self):
        result = generate_summaries([
            section("references", "This citation sentence should never appear in a summary because references are irrelevant.", 10),
            section("results", "The evaluation demonstrates improved accuracy on the test benchmark for the proposed system.", 6),
        ])
        self.assertTrue(result["executive"]["available"])
        self.assertNotIn("citation", result["executive"]["sentences"][0]["text"])


if __name__ == "__main__":
    unittest.main()
