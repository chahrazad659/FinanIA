"""Extraction et découpage traçable. Aucun texte source n'est une instruction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = [{"page": i + 1, "text": page.extract_text() or ""}
             for i, page in enumerate(reader.pages)]
    if not any(len(p["text"].strip()) > 100 for p in pages):
        raise ValueError("PDF sans texte exploitable. Effectuez un OCR avant l'indexation.")
    return pages


def split_text(text: str, size: int = 1800, overlap: int = 200) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("Le chevauchement doit être inférieur à la taille du passage.")
    text = re.sub(r"[ \t]+", " ", text).strip()
    parts, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return parts


def build_chunks(pages: list[dict], catalog: list[dict], source: str) -> list[dict]:
    """A page is never assigned to a neighbouring programme implicitly.

    A catalog item may supply page-specific excerpts in `page_texts` when
    several entries share one PDF page. Text is checked against the PDF.
    """
    page_map = {p["page"]: p["text"] for p in pages}
    chunks = []
    for program in catalog:
        for page in program["pages"]:
            if page not in page_map:
                raise ValueError(f"Page absente du PDF : {page}")
            text = program.get("page_texts", {}).get(str(page), page_map[page])
            normalize = lambda s: " ".join(s.split())
            reconstructed = program.get("source_text_method") == "table_row_reconstructed_visual_verified"
            if not reconstructed and normalize(text) not in normalize(page_map[page]):
                raise ValueError(f"Extrait non retrouvé dans la page {page}")
            for i, part in enumerate(split_text(text)):
                chunks.append({
                    "id": f"{program['id']}-p{page}-c{i + 1}",
                    "program_id": program["id"], "program_name": program["name"],
                    "category": program["category"], "page": page,
                    "text": part, "source": source,
                    "source_text_method": program.get("source_text_method", "exact_extracted_text"),
                    "caveats": program.get("caveats", []),
                })
    if not chunks:
        raise ValueError("Le catalogue ne contient aucun passage indexable.")
    return chunks


def load_corpus() -> tuple[list[dict], list[dict], dict]:
    manifest = read_json(DATA / "manifest.json")
    pages = read_json(DATA / "pdf_pages.json")
    catalog = read_json(DATA / "program_catalog.json")
    return build_chunks(pages, catalog, manifest["pdf_name"]), catalog, manifest


def rebuild_pages(pdf_path: Path) -> dict:
    """Reextract the supplied edition only; catalog remapping is explicit."""
    manifest = read_json(DATA / "manifest.json")
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if digest != manifest["pdf_sha256"]:
        raise ValueError("Ce PDF diffère de l'édition cataloguée. Mettez à jour le catalogue, "
                         "les pages et le manifeste ensemble avant de l'utiliser.")
    pages = extract_pdf(pdf_path)
    build_chunks(pages, read_json(DATA / "program_catalog.json"), manifest["pdf_name"])
    (DATA / "pdf_pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"pages": len(pages), "sha256": digest}
