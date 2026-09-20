from datetime import datetime
from io import BytesIO
import unittest

from finrag.profile import (
    EXCEL_HEADERS, build_profile, build_query, iter_excel_profiles,
    list_excel_users, load_excel_profile, load_excel_dataset, load_schema, normalize_answers,
)


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load_schema()

    def test_numbers_missing_and_french_formats(self):
        answers, warnings = normalize_answers({"Q31": "1 250 000,50", "Q32": None,
                                               "Q40": "", "Q48": 0, "Q2": "31"}, self.schema)
        self.assertEqual(answers["Q31"], 1250000.5)
        self.assertIsNone(answers["Q32"])
        self.assertIsNone(answers["Q40"])
        self.assertEqual(answers["Q48"], 0)
        self.assertEqual(answers["Q2"], 31)
        self.assertEqual(warnings, [])

    def test_invalid_values_never_become_plausible_numbers(self):
        for value in (float("nan"), float("inf"), -3, "1.234,50", True, "beaucoup"):
            with self.subTest(value=value):
                answers, warnings = normalize_answers({"Q32": value}, self.schema)
                self.assertIsNone(answers["Q32"])
                self.assertTrue(warnings)
        self.assertIsNone(normalize_answers({"Q51": 1.5}, self.schema)[0]["Q51"])

    def test_excel_dates_and_unknown_choices(self):
        answers, warnings = normalize_answers({"Q26": datetime(2020, 1, 2), "Q27": "03/01/2020",
                                               "Q19": "Secteur expérimental", "Q999": "ignored"}, self.schema)
        self.assertEqual(answers["Q26"], "2020-01-02")
        self.assertEqual(answers["Q27"], "2020-01-03")
        self.assertEqual(answers["Q19"], "Secteur expérimental")
        self.assertNotIn("Q999", answers)
        self.assertEqual(len(warnings), 2)

    def test_multiselect_excel_values(self):
        answers, warnings = normalize_answers({"Q115": "Subvention / prime ; Crowdfunding ; Crowdfunding",
                                               "Q133": []}, self.schema)
        self.assertEqual(answers["Q115"], ["Subvention / prime", "Crowdfunding"])
        self.assertIsNone(answers["Q133"])
        self.assertFalse(warnings)

    def test_profile_exposes_conflicts_without_eligibility(self):
        profile = build_profile({"Q31": 100, "Q32": 200, "Q52": 2, "Q53": 3,
                                 "Q9": "Auto-entrepreneur", "Q6": "Avec 1 associé",
                                 "Q26": "2020-01-02", "Q27": "2019-01-01"}, self.schema)
        self.assertEqual(len(profile["warnings"]), 4)
        self.assertIn("Quel montant recherchez-vous exactement", profile["summary"])
        self.assertTrue(profile["missing_information"])
        self.assertNotIn("eligibility", profile)

    def test_query_ignores_negative_topics(self):
        profile = build_profile({"Q19": "Industrie", "Q33": "Acheter une machine / équipement",
                                 "Q78": "Non", "Q87": "Non", "Q100": "Non", "Q88": "Oui"}, self.schema)
        query = build_query(profile, self.schema)
        self.assertIn("Industrie", query)
        self.assertIn("efficacité énergétique", query)
        self.assertNotIn("innovation", query)
        self.assertNotIn("solaires", query)
        self.assertNotIn("exportatrice", query)
        self.assertNotIn("Non", query)

    def test_optional_schema_and_full_questionnaire_support(self):
        raw = {}
        for question in self.schema["questions"]:
            key, kind = question["id"], question["type"]
            if key in {"Q26", "Q27"}:
                raw[key] = "2020-01-01"
            elif kind == "number":
                raw[key] = 10
            elif kind == "single":
                raw[key] = question["options"][0]
            elif kind == "multi":
                raw[key] = question["options"][:2]
            else:
                raw[key] = "Détail fourni"
        answers, warnings = normalize_answers(raw)
        self.assertEqual(len(answers), 134)
        self.assertEqual(warnings, [])
        profile = build_profile(answers)
        self.assertEqual(len(profile["labeled_answers"]), 134)
        self.assertTrue(build_query(profile))
        empty = build_profile({})
        self.assertEqual(empty["answers"], {})
        self.assertEqual(empty["summary"], "")
        self.assertTrue(empty["missing_information"])

    @staticmethod
    def workbook(headers=None, duplicate=False, empty=False):
        from openpyxl import Workbook
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Reponses_synthetiques"
        sheet.append(headers or EXCEL_HEADERS)
        data = [] if empty else [("User 1", 42000), ("User 1" if duplicate else "User 2", 81000)]
        for user, amount in data:
            row = [None] * 135
            row[0], row[2], row[26], row[32] = user, 30, datetime(2021, 3, 2), amount
            sheet.append(row)
        stream = BytesIO()
        workbook.save(stream)
        workbook.close()
        stream.seek(0)
        return stream

    def test_excel_mapping_and_reusable_stream(self):
        stream = self.workbook()
        self.assertEqual(list_excel_users(stream, limit=1), ["User 1"])
        row = load_excel_profile(stream, "User 2")
        self.assertEqual(row["Q32"], 81000)
        self.assertEqual(row["Q26"], datetime(2021, 3, 2))
        self.assertIsNone(row["Q31"])
        self.assertEqual(len(list(iter_excel_profiles(stream))), 2)
        with self.assertRaises(KeyError):
            load_excel_profile(stream, "Unknown")

    def test_excel_rejects_wrong_headers_and_duplicates(self):
        with self.assertRaises(ValueError):
            list(iter_excel_profiles(self.workbook(headers=["User", "Q1"])))
        with self.assertRaises(ValueError):
            list(iter_excel_profiles(self.workbook(duplicate=True)))

    def test_dataset_preserves_real_user_ids_and_first_last_values(self):
        stream = self.workbook()
        profiles = load_excel_dataset(stream)
        self.assertEqual(list(profiles), ["User 1", "User 2"])
        self.assertEqual(profiles["User 1"]["User"], "User 1")
        self.assertEqual(profiles["User 1"]["Q32"], 42000)
        self.assertEqual(profiles["User 2"]["Q32"], 81000)
        self.assertIsNone(profiles["User 2"]["Q134"])
        self.assertEqual(profiles["User 2"]["Q26"], datetime(2021, 3, 2))
        self.assertFalse(stream.closed)
        self.assertEqual(load_excel_dataset(stream)["User 2"]["Q32"], 81000)

    def test_dataset_rejects_duplicate_header_user_and_zero_rows(self):
        headers = EXCEL_HEADERS.copy()
        headers[2] = "Q1"
        with self.assertRaisesRegex(ValueError, "Colonnes attendues"):
            load_excel_dataset(self.workbook(headers=headers))
        with self.assertRaisesRegex(ValueError, "dupliqué"):
            load_excel_dataset(self.workbook(duplicate=True))
        with self.assertRaisesRegex(ValueError, "aucun profil User"):
            load_excel_dataset(self.workbook(empty=True))


if __name__ == "__main__":
    unittest.main()
