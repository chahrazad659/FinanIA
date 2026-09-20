"""Recherche BM25 hors ligne et hybride BM25 + embeddings locaux Ollama."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata

from .ollama_client import OllamaClient


class RetrievalError(RuntimeError):
    pass


_STOP = set("a au aux avec ce ces cette dans de des du en et est la le les leur leurs ou par pour que qui se ses son sur un une vos votre vous nous notre nos il elle ils elles d l s n qu plus ainsi".split())


def tokenize(text: str) -> list[str]:
    normalized = "".join(char for char in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(char))
    return [word for word in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE) if word not in _STOP]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RetrievalError("Dimension des embeddings modifiée. Reconstruisez l'index sémantique.")
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


class Retriever:
    def __init__(self, chunks: list[dict], index_dir: Path, client: OllamaClient | None = None):
        if not isinstance(chunks, list):
            raise RetrievalError("Le corpus doit être une liste de passages.")
        seen = set()
        self.chunks = []
        for chunk in chunks:
            if not isinstance(chunk, dict) or any(not isinstance(chunk.get(key), str) or not chunk[key].strip() for key in ("id", "program_id", "program_name", "text", "source")):
                raise RetrievalError("Un passage doit fournir id, program_id, program_name, text et source.")
            if chunk["id"] in seen:
                raise RetrievalError("Identifiants de passages dupliqués dans le corpus.")
            seen.add(chunk["id"])
            self.chunks.append(dict(chunk))
        self.index_dir = Path(index_dir)
        self.client = client
        self._terms = [Counter(tokenize(f"{chunk['program_name']} {chunk.get('category', '')} {chunk['text']}")) for chunk in self.chunks]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._average = sum(self._lengths) / max(len(self._lengths), 1)
        self._df = Counter(word for terms in self._terms for word in terms)

    @property
    def cache_path(self) -> Path:
        return self.index_dir / "embeddings.json"

    def _fingerprint(self) -> str:
        data = {
            "embedding_format_version": 2,
            "model": self.client.embedding_model if self.client else None,
            "chunks": [{key: chunk.get(key) for key in ("id", "program_id", "program_name", "page", "text", "source", "category")} for chunk in self.chunks],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()

    def _embedding_text(self, text: str, *, query: bool = False) -> str:
        # Préfixes requis par Nomic : https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
        model = self.client.embedding_model.lower().rsplit("/", 1)[-1] if self.client else ""
        if model.startswith("nomic-embed-text"):
            return ("search_query: " if query else "search_document: ") + text
        return text

    def build_embeddings(self) -> int:
        """Ne remplace l'ancien cache qu'une fois le nouveau corpus entièrement vectorisé."""
        if self.client is None:
            raise RetrievalError("Configurez Ollama avant de construire l'index sémantique.")
        if not self.chunks:
            raise RetrievalError("Le corpus est vide : importez d'abord le PDF.")
        vectors = []
        for start in range(0, len(self.chunks), 32):
            texts = [self._embedding_text(f"{chunk['program_name']}\n{chunk['text']}") for chunk in self.chunks[start:start + 32]]
            vectors.extend(self.client.embed(texts))
        self._validate_vectors(vectors)
        payload = {"version": 2, "model": self.client.embedding_model, "fingerprint": self._fingerprint(), "ids": [chunk["id"] for chunk in self.chunks], "vectors": vectors}
        self.index_dir.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.index_dir, prefix="embeddings-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.cache_path)
        except OSError as exc:
            raise RetrievalError("Impossible d'enregistrer l'index. Vérifiez les droits du dossier de cache.") from exc
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return len(vectors)

    def _validate_vectors(self, vectors) -> list[list[float]]:
        if not isinstance(vectors, list) or len(vectors) != len(self.chunks):
            raise RetrievalError("Index sémantique incomplet. Reconstruisez l'index.")
        dimension = None
        for row in vectors:
            if not isinstance(row, list) or not row:
                raise RetrievalError("Vecteurs invalides. Reconstruisez l'index sémantique.")
            if dimension is None:
                dimension = len(row)
            if len(row) != dimension or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in row) or not any(row):
                raise RetrievalError("Vecteurs incohérents. Reconstruisez l'index sémantique.")
        return vectors

    def _load_embeddings(self) -> list[list[float]]:
        if self.client is None:
            raise RetrievalError("Ollama doit être configuré pour la recherche sémantique.")
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise RetrievalError("Index sémantique absent ou illisible. Cliquez sur « Construire l'index sémantique ».") from exc
        if not isinstance(data, dict) or data.get("version") != 2 or data.get("fingerprint") != self._fingerprint() or data.get("model") != self.client.embedding_model or data.get("ids") != [chunk["id"] for chunk in self.chunks]:
            raise RetrievalError("L'index sémantique ne correspond plus au corpus ou au modèle. Reconstruisez-le.")
        return self._validate_vectors(data.get("vectors"))

    def _bm25(self, query: str) -> list[float]:
        words = set(tokenize(query))
        total = len(self.chunks)
        scores = []
        for terms, length in zip(self._terms, self._lengths):
            score = 0.0
            for word in words:
                frequency = terms.get(word, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (total - self._df[word] + 0.5) / (self._df[word] + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / max(self._average, 1)))
            scores.append(score)
        return scores

    def search(self, query: str, k: int = 8, semantic: bool = False) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            return []
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
            raise RetrievalError("Le nombre de passages doit être compris entre 1 et 100.")
        if not self.chunks:
            return []
        scores = self._bm25(query)
        lexical = sorted((index for index, score in enumerate(scores) if score > 0), key=lambda index: (-scores[index], self.chunks[index]["id"]))
        if semantic:
            vectors = self._load_embeddings()
            query_vectors = self.client.embed([self._embedding_text(query, query=True)])
            if len(query_vectors) != 1:
                raise RetrievalError("Ollama n'a pas vectorisé la requête correctement.")
            semantic_scores = [_cosine(query_vectors[0], vector) for vector in vectors]
            semantic_order = sorted(range(len(vectors)), key=lambda index: (-semantic_scores[index], self.chunks[index]["id"]))
            # Reciprocal Rank Fusion : échelles BM25 et cosinus non comparables.
            scores = [0.0] * len(self.chunks)
            for ranking in (lexical, semantic_order):
                for rank, index in enumerate(ranking, 1):
                    scores[index] += 1.0 / (60 + rank)
            ranking = sorted(range(len(scores)), key=lambda index: (-scores[index], self.chunks[index]["id"]))
        else:
            ranking = lexical
        result = []
        counts = Counter()
        for index in ranking:
            chunk = self.chunks[index]
            if counts[chunk["program_id"]] >= 2:
                continue
            result.append({**chunk, "score": round(scores[index], 8), "retrieval_mode": "hybride" if semantic else "bm25"})
            counts[chunk["program_id"]] += 1
            if len(result) >= k:
                break
        return result
