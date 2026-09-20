"""Commandes reproductibles pour l'index, la recherche et les vérifications."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import os

from finrag.ingest import DATA, load_corpus, read_json, rebuild_pages
from finrag.profile import load_schema, build_profile, build_query, iter_excel_profiles
from finrag.retrieval import Retriever, RetrievalError
from finrag.ollama_client import OllamaClient, OllamaError
from finrag.generation import generate_recommendations, GenerationError


def emit(value, path=None):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Résultat enregistré : {path}")
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--chat-model", default=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct-q4_K_M"))
    parser.add_argument("--embed-model", default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"))
    parser.add_argument("--timeout", type=int, default=600)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Vérifier les modèles Ollama installés")
    sub.add_parser("index", help="Construire les embeddings du guide uniquement")
    reextract = sub.add_parser("ingest", help="Réextraire le PDF de l'édition cataloguée")
    reextract.add_argument("--pdf", type=Path)
    recommend = sub.add_parser("recommend", help="Exécuter questionnaire → profil → RAG → LLM")
    recommend.add_argument("--profile", type=Path, help="Fichier {answers:{Q1:...}} ou réponses Q1…Q134")
    recommend.add_argument("--example", default="Création d'une petite entreprise")
    recommend.add_argument("--excerpts-only", action="store_true", help="Recherche et extraits, sans LLM")
    recommend.add_argument("--semantic", action="store_true", help="Index hybride requis")
    recommend.add_argument("--top", type=int, default=3)
    recommend.add_argument("--passages", type=int, default=8)
    recommend.add_argument("--out", type=Path)
    evaluation = sub.add_parser("evaluate", help="Tests de recherche illustratifs, sans décision utilisateur")
    evaluation.add_argument("--semantic", action="store_true")
    evaluation.add_argument("--k", type=int, default=5)
    evaluation.add_argument("--out", type=Path)
    audit = sub.add_parser("audit-excel", help="Audit de profils synthétiques ; aucun appel LLM")
    audit.add_argument("--file", type=Path, default=DATA / "raw" / "Questionnaire_10000_users_synthetiques.xlsx")
    audit.add_argument("--limit", type=int, default=10000)
    audit.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        client = OllamaClient(args.host, args.chat_model, args.embed_model, args.timeout)
        if args.command == "check":
            emit({"models": client.tags()})
            return
        if args.command == "ingest":
            manifest = read_json(DATA / "manifest.json")
            emit(rebuild_pages(args.pdf or DATA / "raw" / manifest["pdf_name"]))
            return
        if args.command == "audit-excel":
            if args.limit < 1:
                parser.error("--limit doit être positif")
            schema = load_schema()
            warnings = Counter()
            rows = 0
            missing = 0
            iterator = iter_excel_profiles(args.file)
            try:
                for record in iterator:
                    profile = build_profile(record, schema)
                    rows += 1
                    missing += sum(v is None for v in profile["answers"].values())
                    warnings.update(profile["warnings"])
                    if rows >= args.limit:
                        break
            finally:
                iterator.close()
            emit({"rows_checked": rows, "missing_answers": missing,
                  "warnings": dict(warnings), "note": "Données synthétiques sans vérité terrain de recommandation. Aucun appel LLM, aucune décision d'éligibilité."}, args.out)
            return
        chunks, catalog, manifest = load_corpus()
        retriever = Retriever(chunks, DATA / "index", client)
        if args.command == "index":
            emit({"indexed_passages": retriever.build_embeddings(), "model": args.embed_model})
        elif args.command == "recommend":
            raw = read_json(args.profile) if args.profile else read_json(DATA / "example_profiles.json")[args.example]
            profile = build_profile(raw.get("answers", raw))
            query = build_query(profile)
            hits = retriever.search(query, k=args.passages, semantic=args.semantic)
            llm_profile = {"reponses": profile["labeled_answers"], "incoherences": profile["warnings"]}
            result = generate_recommendations(llm_profile, query, hits, None if args.excerpts_only else client, args.top)
            emit({"query": query, "profile_warnings": profile["warnings"], "result": result}, args.out)
        elif args.command == "evaluate":
            if not 1 <= args.k <= 50:
                parser.error("--k doit être entre 1 et 50")
            cases = read_json(DATA / "retrieval_cases.json")
            details = []
            for case in cases:
                hits = retriever.search(case["query"], k=min(100, args.k * 2), semantic=args.semantic)
                ids = list(dict.fromkeys(h["program_id"] for h in hits))[:args.k]
                relevant = set(case["relevant_program_ids"])
                recalls = len(relevant.intersection(ids)) / len(relevant)
                reciprocal = next((1 / (i + 1) for i, item in enumerate(ids) if item in relevant), 0)
                details.append({**case, "retrieved": ids, "recall_at_k": recalls, "reciprocal_rank": reciprocal})
            emit({"mode": "hybride" if args.semantic else "bm25", "k": args.k, "cases": len(cases),
                  "recall_at_k": sum(d["recall_at_k"] for d in details) / len(details),
                  "mrr_at_k": sum(d["reciprocal_rank"] for d in details) / len(details),
                  "limitation": "Tests illustratifs comportant les noms des programmes, pas une mesure de pertinence pour des profils réels. Aucun score de précision sur les 10 000 profils non annotés.",
                  "details": details}, args.out)
    except (OllamaError, RetrievalError, GenerationError, ValueError, KeyError, OSError, TypeError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
