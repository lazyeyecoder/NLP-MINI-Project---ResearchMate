import unittest

from app.services.research_extractor import extract_research_insights


def section(section_id, text, page=1, subsections=None):
    return {
        "section_id": section_id,
        "text": text,
        "page_start": page,
        "page_end": page,
        "subsections": subsections or [],
    }


class ResearchExtractorTests(unittest.TestCase):
    def test_cue_phrase_problem_and_objective_with_provenance(self):
        result = extract_research_insights([
            section("introduction", "However, existing methods lack robust multimodal representations. This paper aims to improve remote sensing representations.", 2)
        ])
        self.assertIsNotNone(result["research_problem"])
        self.assertIsNotNone(result["research_objective"])
        self.assertEqual(result["research_problem"]["page_numbers"], [2])
        self.assertGreater(result["research_objective"]["confidence"], 0)

    def test_method_models_and_nested_sections(self):
        result = extract_research_insights([
            section("experiments", "We train a transformer and a masked autoencoder.", 3, [
                section("dataset", "We evaluate on the BigEarthNet dataset.", 4)
            ])
        ])
        self.assertEqual({item["value"].lower() for item in result["models_algorithms"]}, {"transformer", "masked autoencoder"})
        self.assertEqual(result["datasets"][0]["value"], "BigEarthNet dataset")

    def test_dataset_and_data_source_are_distinguished(self):
        result = extract_research_insights([
            section("experiments", "We evaluate the BigEarthNet dataset using Sentinel-2 satellite imagery and a transformer.", 3)
        ])
        self.assertIn("BigEarthNet dataset", [item["value"] for item in result["datasets"]])
        self.assertNotIn("Sentinel-2", [item["value"] for item in result["datasets"]])
        self.assertIn("Sentinel-2", [item["value"] for item in result["data_sources"]])

    def test_multiple_datasets_and_sources(self):
        result = extract_research_insights([
            section("dataset", "The EuroSAT dataset and BigEarthNet dataset use Sentinel-1 radar and Sentinel-2 optical sensors.", 4)
        ])
        self.assertGreaterEqual(len(result["datasets"]), 2)
        self.assertGreaterEqual(len(result["data_sources"]), 2)

    def test_ambiguous_mae_requires_context(self):
        metric = extract_research_insights([
            section("results", "Mean Absolute Error (MAE) was 0.12.", 5)
        ])
        model = extract_research_insights([
            section("methodology", "We use a Masked Autoencoder (MAE) model.", 6)
        ])
        ambiguous = extract_research_insights([
            section("results", "The MAE approach was discussed.", 7)
        ])
        self.assertEqual(metric["results_metrics"][0]["value"]["metric"], "MAE")
        self.assertIn("mae", [item["value"].lower() for item in model["models_algorithms"]])
        self.assertEqual(ambiguous["results_metrics"], [])

    def test_objective_prefers_abstract_over_conclusion_fallback(self):
        result = extract_research_insights([
            section("abstract", "We propose a transparent method for scientific classification.", 1),
            section("conclusion", "We propose that future systems should be evaluated broadly.", 8),
        ])
        self.assertEqual(result["research_objective"]["source_section"], "abstract")

    def test_conclusion_objective_fallback_has_lower_confidence(self):
        result = extract_research_insights([
            section("conclusion", "This work presents a useful framework for analysis.", 8),
        ])
        self.assertIsNotNone(result["research_objective"])
        self.assertLess(result["research_objective"]["confidence"], 0.8)

    def test_metrics_are_multiple_and_provenanced(self):
        result = extract_research_insights([
            section("results", "Accuracy was 94.2% and F1-score was 0.88 on the benchmark.", 5)
        ])
        self.assertEqual(len(result["results_metrics"]), 2)
        self.assertEqual(result["results_metrics"][0]["page_numbers"], [5])

    def test_limitations_and_future_work(self):
        result = extract_research_insights([
            section("limitations", "A limitation is the small dataset.", 6),
            section("future_work", "Future work will explore larger datasets.", 7),
        ])
        self.assertIsNotNone(result["limitations"])
        self.assertIsNotNone(result["future_work"])

    def test_missing_fields_are_null_without_hallucination(self):
        result = extract_research_insights([section("abstract", "This paper describes a study of text processing.", 1)])
        self.assertIsNone(result["research_problem"])
        self.assertIsNone(result["future_work"])
        self.assertEqual(result["datasets"], [])

    def test_low_confidence_candidates_remain_evidence_based(self):
        result = extract_research_insights([
            section("conclusion", "However, the method may not generalize to every setting.", 8)
        ])
        self.assertIsNotNone(result["limitations"])
        self.assertLessEqual(result["limitations"]["confidence"], 0.99)


if __name__ == "__main__":
    unittest.main()
