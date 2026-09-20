"""Corpus livré : provenance des preuves et absence d'entrées d'annuaire.

Ces tests valident le corpus fourni, pas l'éligibilité des utilisateurs ni
la disponibilité actuelle des programmes du kit de décembre 2025.
"""
from collections import Counter
from pathlib import Path
import unittest

from finrag.ingest import build_chunks, read_json


DATA = Path(__file__).resolve().parents[1] / "data"
TABLE_METHOD = "table_row_reconstructed_visual_verified"
# Couverture, sommaire, intercalaires et photographies ; les annuaires débutent p82.
EXCLUDED_PAGES = set(range(1, 8)) | {
    12, 24, 28, 31, 37, 42, 46, 49, 54, 58, 59, 74, 76, 78, 79,
} | set(range(82, 97))


def normalized(text):
    return " ".join(text.split())


class SuppliedCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = read_json(DATA / "pdf_pages.json")
        cls.catalog = read_json(DATA / "program_catalog.json")
        cls.page_map = {page["page"]: page["text"] for page in cls.pages}
        cls.chunks = build_chunks(cls.pages, cls.catalog, "kit_decembre_2025.pdf")

    def test_page_provenance_and_program_ids_are_complete_and_unique(self):
        self.assertEqual(sorted(self.page_map), list(range(1, 97)))
        self.assertEqual(len(self.pages), len(self.page_map))
        ids = [item["id"] for item in self.catalog]
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))
        for item in self.catalog:
            with self.subTest(program=item["id"]):
                self.assertTrue(item["name"].strip())
                self.assertTrue(item["pages"])
                self.assertEqual(len(item["pages"]), len(set(item["pages"])))
                self.assertTrue(set(item["pages"]) <= set(self.page_map))
                self.assertEqual({str(page) for page in item["pages"]}, set(item["page_texts"]))
                self.assertEqual({segment["page"] for segment in item["source_segments"]}, set(item["pages"]))
                for segment in item["source_segments"]:
                    self.assertEqual(segment["text"], item["page_texts"][str(segment["page"])])

    def test_quotes_are_extracted_or_explicitly_marked_verified_table_rows(self):
        for item in self.catalog:
            with self.subTest(program=item["id"]):
                method = item["source_text_method"]
                self.assertIn(method, {"exact_extracted_text", TABLE_METHOD})
                self.assertTrue(item["caveats"])
                for page, text in item["page_texts"].items():
                    self.assertTrue(text.strip())
                    if method == "exact_extracted_text":
                        self.assertIn(normalized(text), normalized(self.page_map[int(page)]))
                    else:
                        self.assertIn(int(page), {30, 61, 62, 75})
                        self.assertTrue(any("vérifié visuellement" in c for c in item["caveats"]))

    def test_index_has_no_empty_duplicate_or_orphan_chunks(self):
        by_id = {item["id"]: item for item in self.catalog}
        self.assertTrue(self.chunks)
        self.assertEqual(len(self.chunks), len({chunk["id"] for chunk in self.chunks}))
        represented = Counter(chunk["program_id"] for chunk in self.chunks)
        self.assertEqual(set(represented), set(by_id))
        for chunk in self.chunks:
            with self.subTest(chunk=chunk["id"]):
                self.assertTrue(chunk["text"].strip())
                item = by_id[chunk["program_id"]]
                self.assertIn(chunk["page"], item["pages"])
                self.assertEqual(chunk["program_name"], item["name"])
                self.assertIn(normalized(chunk["text"]), normalized(item["page_texts"][str(chunk["page"])]))

    def test_contents_directories_and_section_headings_are_not_programs(self):
        used_pages = {page for item in self.catalog for page in item["pages"]}
        self.assertFalse(used_pages & EXCLUDED_PAGES)
        generic_headings = {
            "sommaire", "structures d’accompagnement", "appui aux pme",
            "fonds généralistes", "fonds sectoriels", "appui à l’export",
            "ecosystème de l’investissement", "financement & régulation",
        }
        for item in self.catalog:
            self.assertNotIn(item["name"].strip().lower(), generic_headings)

    def test_shared_fund_pages_preserve_each_funds_own_evidence(self):
        by_id = {item["id"]: item for item in self.catalog}
        founders = by_id["212_founders"]["page_texts"]["63"]
        upline = by_id["upline_investments_fund"]
        self.assertIn("212 FOUNDERS", founders)
        self.assertNotIn("UPLINE INVESTMENTS FUND", founders)
        self.assertEqual(upline["pages"], [63, 64])
        for text in upline["page_texts"].values():
            self.assertIn("UPLINE INVESTMENTS FUND", text)
            self.assertNotIn("MOUSSAHAMA II", text)

    def test_diagnostic_queries_reference_real_programs(self):
        cases = read_json(DATA / "retrieval_cases.json")
        ids = {item["id"] for item in self.catalog}
        self.assertTrue(cases)
        for case in cases:
            self.assertTrue(case["query"].strip())
            self.assertTrue(case["relevant_program_ids"])
            self.assertTrue(set(case["relevant_program_ids"]) <= ids)


if __name__ == "__main__":
    unittest.main()
