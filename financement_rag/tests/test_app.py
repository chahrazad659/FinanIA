"""Streamlit integration checks; no Ollama requests or index writes are allowed."""
from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def labelled(elements, label):
    return next(element for element in elements if element.label == label)


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        self.requests = patch("finrag.ollama_client.OllamaClient._request",
                              side_effect=AssertionError("Test must not call Ollama"))
        self.index = patch("finrag.retrieval.Retriever.build_embeddings",
                           side_effect=AssertionError("Test must not build an index"))
        self.requests_mock = self.requests.start()
        self.index_mock = self.index.start()
        self.addCleanup(self.requests.stop)
        self.addCleanup(self.index.stop)

    def assert_clean(self, app):
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual([error.value for error in app.error], [])
        self.requests_mock.assert_not_called()
        self.index_mock.assert_not_called()

    def test_empty_app_does_not_create_default_answers(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["answers"], {})
        metric = labelled(app.metric, "Questions du diagnostic")
        self.assertEqual(metric.value, "134")
        questionnaire_selects = [widget for widget in app.selectbox if str(widget.key).startswith("answer_")]
        self.assertTrue(questionnaire_selects)
        self.assertTrue(all(widget.value is None for widget in questionnaire_selects))
        labelled(app.button, "Enregistrer le profil").click().run()
        self.assert_clean(app)
        self.assertTrue(all(value is None for value in app.session_state["answers"].values()))
        self.assertNotIn("Rechercher les programmes", [button.label for button in app.button])

    def test_example_full_form_search_and_json_exports(self):
        with patch("streamlit.download_button", wraps=st.download_button) as download:
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            labelled(app.button, "Charger cet exemple").click().run()
            self.assert_clean(app)
            initial = dict(app.session_state["answers"])
            self.assertTrue(initial.get("Q17"))
            amount_widget = next(widget for widget in app.text_input if str(widget.key).endswith("_Q32"))
            amount_widget.set_value("250 000,50")
            labelled(app.button, "Enregistrer le profil").click().run()
            self.assert_clean(app)
            self.assertEqual(app.session_state["answers"]["Q32"], 250000.5)

            labelled(app.radio, "Parcours").set_value("Complet · 134 questions").run()
            self.assert_clean(app)
            controls = [element for collection in (app.selectbox, app.multiselect, app.text_input, app.text_area)
                        for element in collection if str(element.key).startswith("answer_")]
            self.assertEqual(len(controls), 134)
            self.assertEqual(len({element.key for element in controls}), 134)
            labelled(app.button, "Enregistrer le profil").click().run()
            self.assert_clean(app)
            self.assertEqual(app.session_state["answers"]["Q32"], 250000.5)
            self.assertEqual(app.session_state["answers"]["Q17"], initial["Q17"])

            labelled(app.radio, "Analyse").set_value("Extraits sans LLM").run()
            labelled(app.button, "Rechercher les programmes").click().run()
            self.assert_clean(app)
            result = app.session_state["result"]
            self.assertNotEqual(result["mode"], "llm")
            self.assertTrue(result["recommendations"])
            self.assertTrue(app.session_state["hits"])
            for recommendation in result["recommendations"]:
                self.assertTrue(recommendation["citations"])
                self.assertTrue(all(citation["page"] > 0 for citation in recommendation["citations"]))

            # AppTest exposes download controls, but does not emulate a browser
            # download click. Verify the offered bytes and the generated URLs.
            payloads = {call.args[0]: call.args[1] for call in download.call_args_list}
            exported_profile = json.loads(payloads["Exporter le profil JSON"])
            exported_result = json.loads(payloads["Exporter le résultat JSON"])
            self.assertEqual(exported_profile["answers"]["Q32"], 250000.5)
            self.assertEqual(exported_result["recommendations"], result["recommendations"])
            controls = {element.proto.label: element.proto.url for element in app.get("download_button")}
            self.assertTrue(controls["Exporter le profil JSON"])
            self.assertTrue(controls["Exporter le résultat JSON"])

    def test_invalid_amount_is_reported_and_left_unknown(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        amount_widget = next(widget for widget in app.text_input if str(widget.key).endswith("_Q32"))
        amount_widget.set_value("-500")
        labelled(app.button, "Enregistrer le profil").click().run()
        self.assert_clean(app)
        self.assertIsNone(app.session_state["answers"]["Q32"])
        self.assertTrue(any("Q32" in warning.value for warning in app.warning))


if __name__ == "__main__":
    unittest.main()
