"""Petit client Ollama : exclusivement local, sans dépendance HTTP externe."""

from __future__ import annotations

import ipaddress
import json
import math
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class OllamaError(RuntimeError):
    """Erreur exploitable directement par l'interface."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise OllamaError("Redirection HTTP refusée : Ollama doit rester sur votre ordinateur.")


def _local_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        host = parts.hostname
        port = parts.port
    except (TypeError, ValueError) as exc:
        raise OllamaError("Adresse Ollama invalide. Exemple : http://127.0.0.1:11434") from exc
    if (
        parts.scheme != "http"
        or not host
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise OllamaError("Utilisez une adresse HTTP locale, sans chemin ni identifiants.")
    if host.lower() == "localhost":
        host = "127.0.0.1"  # Pas de résolution DNS et pas de proxy système.
    else:
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("not loopback")
        except ValueError as exc:
            raise OllamaError("Seules les adresses locales localhost, 127.0.0.1 ou ::1 sont autorisées.") from exc
    host = f"[{host}]" if ":" in host else host
    return f"http://{host}" + (f":{port}" if port is not None else "")


def _vectors(value, expected: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != expected:
        raise OllamaError("Ollama a retourné un nombre de vecteurs inattendu.")
    dimension = None
    result = []
    for row in value:
        if not isinstance(row, list) or not row:
            raise OllamaError("Ollama a retourné un vecteur vide ou invalide.")
        if dimension is None:
            dimension = len(row)
        if len(row) != dimension or any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
            for x in row
        ):
            raise OllamaError("Les vecteurs Ollama sont incohérents ou contiennent des valeurs invalides.")
        if not any(row):
            raise OllamaError("Ollama a retourné un vecteur nul.")
        result.append([float(x) for x in row])
    return result


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        chat_model: str = "qwen3:4b",
        embedding_model: str = "embeddinggemma",
        timeout: float = 180,
    ):
        self.base_url = _local_url(base_url)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise OllamaError("Le délai Ollama doit être un nombre positif de secondes.")
        for name in (chat_model, embedding_model):
            if not isinstance(name, str) or not name.strip():
                raise OllamaError("Les noms des modèles Ollama doivent être renseignés.")
            if re.search(r"(?:^|[-:/])cloud(?:$|[-:/])", name.lower()):
                raise OllamaError("Les modèles Ollama cloud sont désactivés. Choisissez un modèle téléchargé sur votre ordinateur.")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    def _request(self, endpoint: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = Request(
            self.base_url + endpoint,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            opener = build_opener(ProxyHandler({}), _NoRedirect())
            with opener.open(request, timeout=self.timeout) as response:
                raw = response.read(32 * 1024 * 1024 + 1)
            if len(raw) > 32 * 1024 * 1024:
                raise OllamaError("Réponse Ollama trop volumineuse. Réduisez le nombre de documents.")
            parsed = json.loads(raw)
        except HTTPError as exc:
            if exc.code == 404:
                raise OllamaError("Modèle ou route Ollama introuvable. Vérifiez 'ollama list' et téléchargez le modèle indiqué.") from exc
            raise OllamaError(f"Ollama a renvoyé une erreur HTTP {exc.code}. Consultez son journal local.") from exc
        except (TimeoutError, URLError, ConnectionError, OSError) as exc:
            raise OllamaError("Ollama est inaccessible ou trop lent. Démarrez Ollama et vérifiez l'adresse et le délai.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaError("Ollama a renvoyé une réponse JSON illisible.") from exc
        if not isinstance(parsed, dict):
            raise OllamaError("Réponse Ollama inattendue : un objet JSON est requis.")
        if parsed.get("error"):
            raise OllamaError("Ollama signale une erreur. Vérifiez le modèle installé et les ressources disponibles.")
        return parsed

    def tags(self) -> list[str]:
        models = self._request("/api/tags").get("models")
        if not isinstance(models, list):
            raise OllamaError("La liste des modèles Ollama est invalide.")
        return [model["name"] for model in models if isinstance(model, dict) and isinstance(model.get("name"), str)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise OllamaError("Les textes à vectoriser doivent être des chaînes non vides.")
        if not texts:
            return []
        response = self._request("/api/embed", {"model": self.embedding_model, "input": texts, "truncate": False})
        return _vectors(response.get("embeddings"), len(texts))

    def chat(self, messages: list[dict], schema: dict) -> dict:
        response = self._request("/api/chat", {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2500},
        })
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaError("Le modèle n'a pas renvoyé de contenu textuel JSON.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaError("Le modèle a produit un JSON invalide. Relancez la génération ou utilisez les extraits.") from exc
        if not isinstance(parsed, dict):
            raise OllamaError("Le modèle doit produire un objet JSON, pas une liste ni du texte libre.")
        return parsed
