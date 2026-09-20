"""Tests hors réseau : index, frontière locale et validation des preuves."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from finrag.generation import GenerationError, generate_recommendations
from finrag.ollama_client import OllamaClient, OllamaError
from finrag.retrieval import Retriever, RetrievalError


CHUNKS = [
    {"id": "a1", "program_id": "A", "program_name": "Programme A", "page": 4, "source": "kit.pdf", "category": "Garantie", "text": "Garantie destinée aux entreprises pour leurs investissements productifs."},
    {"id": "a2", "program_id": "A", "program_name": "Programme A", "page": 5, "source": "kit.pdf", "category": "Garantie", "text": "Les entreprises sollicitent une garantie auprès de la banque partenaire."},
    {"id": "a3", "program_id": "A", "program_name": "Programme A", "page": 5, "source": "kit.pdf", "category": "Garantie", "text": "La garantie accompagne les investissements des entreprises."},
    {"id": "b1", "program_id": "B", "program_name": "Programme B", "page": 8, "source": "kit.pdf", "category": "Formation", "text": "Formation et accompagnement des porteurs de projets à la création."},
]


class FakeClient:
    embedding_model = "test-embed"

    def embed(self, texts):
        return [[1.0, 0.1] if "garantie" in text.lower() else [0.1, 1.0] for text in texts]

    def chat(self, messages, schema):
        return deepcopy(self.response)


def valid_response():
    return {"recommendations": [{"program_id": "A", "reason": "Piste à examiner pour le besoin d'investissement déclaré.", "cautions": ["Les conditions détaillées restent à vérifier."], "citations": [{"chunk_id": "a1", "quote": CHUNKS[0]["text"]}]}], "missing_information": ["Vérifier le montant du besoin."]}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_bm25_accent_normalization_and_diversity(self):
        results = Retriever(CHUNKS, self.directory).search("garantie destinée entreprises", k=8)
        self.assertEqual(results[0]["id"], "a1")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(hit["retrieval_mode"] == "bm25" for hit in results))

    def test_unknown_term_does_not_return_arbitrary_lexical_results(self):
        self.assertEqual(Retriever(CHUNKS, self.directory).search("zyxwvutsrq"), [])

    def test_hybrid_index_and_corpus_mismatch(self):
        retriever = Retriever(CHUNKS, self.directory, FakeClient())
        self.assertEqual(retriever.build_embeddings(), 4)
        self.assertEqual(retriever.search("garantie", semantic=True)[0]["retrieval_mode"], "hybride")
        changed = deepcopy(CHUNKS)
        changed[0]["text"] += " Nouvelle règle."
        with self.assertRaises(RetrievalError):
            Retriever(changed, self.directory, FakeClient()).search("garantie", semantic=True)

    def test_model_mismatch_and_missing_cache_are_explicit(self):
        retriever = Retriever(CHUNKS, self.directory, FakeClient())
        with self.assertRaises(RetrievalError):
            retriever.search("garantie", semantic=True)
        retriever.build_embeddings()
        retriever.client.embedding_model = "other-model"
        with self.assertRaises(RetrievalError):
            retriever.search("garantie", semantic=True)

    def test_failed_rebuild_preserves_old_cache(self):
        retriever = Retriever(CHUNKS, self.directory, FakeClient())
        retriever.build_embeddings()
        original = retriever.cache_path.read_bytes()
        retriever.client.embed = MagicMock(side_effect=OllamaError("indisponible"))
        with self.assertRaises(OllamaError):
            retriever.build_embeddings()
        self.assertEqual(retriever.cache_path.read_bytes(), original)

    def test_invalid_cache_vector_is_rejected(self):
        retriever = Retriever(CHUNKS, self.directory, FakeClient())
        retriever.build_embeddings()
        data = json.loads(retriever.cache_path.read_text())
        data["vectors"][0] = [True, 0]
        retriever.cache_path.write_text(json.dumps(data))
        with self.assertRaises(RetrievalError):
            retriever.search("garantie", semantic=True)

    def test_nomic_prefixes_differ_for_documents_and_queries(self):
        client = FakeClient()
        client.embedding_model = "nomic-embed-text:latest"
        client.embed = MagicMock(wraps=client.embed)
        retriever = Retriever(CHUNKS, self.directory, client)
        retriever.build_embeddings()
        self.assertTrue(all(text.startswith("search_document: ") for text in client.embed.call_args.args[0]))
        retriever.search("garantie", semantic=True)
        self.assertEqual(client.embed.call_args.args[0], ["search_query: garantie"])


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.client.response = valid_response()

    def generate(self):
        return generate_recommendations({"besoin": "investissement"}, "garantie", CHUNKS, self.client)

    def test_valid_citation_maps_title_page_source_from_corpus(self):
        result = self.generate()["recommendations"][0]
        self.assertEqual(result["program_name"], "Programme A")
        self.assertEqual(result["citations"][0]["page"], 4)
        self.assertEqual(result["citations"][0]["source"], "kit.pdf")
        self.assertEqual(result["status"], "À vérifier")

    def test_invented_quote_and_cross_program_citations_rejected(self):
        for citation in ({"chunk_id": "a1", "quote": "Un montant inventé de 500 millions de dirhams."}, {"chunk_id": "b1", "quote": CHUNKS[-1]["text"]}):
            with self.subTest(citation=citation):
                self.client.response = valid_response()
                self.client.response["recommendations"][0]["citations"] = [citation]
                result = self.generate()
                self.assertEqual(result["recommendations"], [])
                self.assertTrue(any("écartée" in warning for warning in result["warnings"]))

    def test_unknown_program_and_invented_page_rejected(self):
        for field, value in (("program_id", "imaginaire"), ("page", 999)):
            self.client.response = valid_response()
            self.client.response["recommendations"][0][field] = value
            self.assertEqual(self.generate()["recommendations"], [])

    def test_wrong_top_level_types_rejected(self):
        for response in ([], {"recommendations": "texte", "missing_information": []}, {"recommendations": [], "missing_information": [None]}):
            self.client.response = response
            with self.assertRaises(GenerationError):
                self.generate()

    def test_definitive_eligibility_rejected(self):
        self.client.response["recommendations"][0]["reason"] = "Vous êtes éligible à ce programme."
        self.assertEqual(self.generate()["recommendations"], [])

    def test_evidence_only_mode_does_not_infer_compatibility(self):
        result = generate_recommendations({}, "garantie", CHUNKS, None)
        self.assertEqual(result["mode"], "extraits")
        self.assertEqual(len(result["recommendations"]), 2)
        self.assertIn("aucune compatibilité", result["recommendations"][0]["reason"])

    def test_no_hits_never_calls_llm_or_returns_recommendations(self):
        self.client.chat = MagicMock(side_effect=AssertionError("ne doit pas être appelé"))
        for client in (self.client, None):
            result = generate_recommendations({}, "garantie", [], client)
            self.assertEqual(result["recommendations"], [])
            self.assertTrue(any("Aucun extrait" in warning for warning in result["warnings"]))
        self.client.chat.assert_not_called()

    def test_wrong_nested_types_are_discarded(self):
        for field, value in (("program_id", []), ("reason", 12), ("cautions", "texte"), ("citations", {})):
            with self.subTest(field=field):
                self.client.response = valid_response()
                self.client.response["recommendations"][0][field] = value
                self.assertEqual(self.generate()["recommendations"], [])

    def test_citation_whitespace_normalization_preserves_french(self):
        self.client.response["recommendations"][0]["citations"][0]["quote"] = "Garantie destinée aux entreprises\n pour leurs investissements productifs."
        result = self.generate()["recommendations"][0]
        self.assertEqual(result["citations"][0]["quote"], CHUNKS[0]["text"])


class OllamaTests(unittest.TestCase):
    def test_cloud_model_names_are_rejected(self):
        for name in ("gpt-oss:120b-cloud", "qwen3:cloud", "cloud/model"):
            with self.subTest(name=name), self.assertRaises(OllamaError):
                OllamaClient(chat_model=name)

    def test_loopback_only(self):
        for url in ("https://example.com", "http://example.com", "http://192.168.1.20:11434", "http://localhost.evil.test", "http://user@localhost:11434", "http://localhost:11434/api"):
            with self.subTest(url=url), self.assertRaises(OllamaError):
                OllamaClient(url)
        self.assertEqual(OllamaClient("http://localhost:11434").base_url, "http://127.0.0.1:11434")
        self.assertEqual(OllamaClient("http://[::1]:11434").base_url, "http://[::1]:11434")

    @patch("finrag.ollama_client.build_opener")
    def test_transport_uses_structured_output_and_no_stream(self, opener):
        opener.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps({"message": {"content": '{"ok": true}'}}).encode()
        self.assertEqual(OllamaClient().chat([{"role": "user", "content": "test"}], {"type": "object"}), {"ok": True})
        request = opener.return_value.open.call_args.args[0]
        body = json.loads(request.data)
        self.assertFalse(body["stream"])
        self.assertFalse(body["think"])
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["format"], {"type": "object"})

    @patch("finrag.ollama_client.build_opener")
    def test_malformed_json_and_model_content(self, opener):
        response = opener.return_value.open.return_value.__enter__.return_value
        for raw in (b"not json", b'{"message":{"content":"not json"}}', b'{"message":{"content":"[]"}}'):
            response.read.return_value = raw
            with self.assertRaises(OllamaError):
                OllamaClient().chat([], {"type": "object"})

    @patch("finrag.ollama_client.build_opener")
    def test_embedding_disables_truncation(self, opener):
        opener.return_value.open.return_value.__enter__.return_value.read.return_value = b'{"embeddings":[[1,0],[0,1]]}'
        self.assertEqual(OllamaClient().embed(["premier", "second"]), [[1.0, 0.0], [0.0, 1.0]])
        body = json.loads(opener.return_value.open.call_args.args[0].data)
        self.assertFalse(body["truncate"])
        self.assertEqual(body["input"], ["premier", "second"])


if __name__ == "__main__":
    unittest.main()
