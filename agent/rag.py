"""RAG MITRE ATT&CK — base vectorielle locale (Phase 3).

Charge le sous-ensemble curaté produit par ``ingest_attack.py``
(``knowledge/attack_techniques.json``), le vectorise avec l'embedding local
Ollama ``nomic-embed-text`` et l'indexe dans ChromaDB (persistant sur disque).

Points d'entrée :
    build_index(force=False)      -> construit (ou réutilise) l'index
    search(query, k=4)            -> liste de techniques pertinentes
    format_results(hits)          -> texte prêt à injecter dans le prompt

Les embeddings sont TOUJOURS fournis explicitement à ChromaDB : on ne veut
jamais que Chroma déclenche son embedding ONNX par défaut (téléchargement
HuggingFace, hors du 100 % local).
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
TECHNIQUES_PATH = KNOWLEDGE_DIR / "attack_techniques.json"
CHROMA_DIR = KNOWLEDGE_DIR / "chroma"

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "mitre_attack"
EMBED_DIM = 768          # dimension de nomic-embed-text

DEFAULT_K = 4

class _NoAutoEmbed:
    """Embedding function factice : refuse tout appel automatique.

    ChromaDB installe par défaut un embedder ONNX qui télécharge un modèle
    depuis HuggingFace. On fournit toujours nos propres embeddings : si cette
    classe est appelée, c'est qu'un embed implicite s'est glissé quelque part.
    """

    def __call__(self, input):  # noqa: A002 - signature ChromaDB
        raise RuntimeError(
            "ChromaDB a tenté un embedding automatique (modèle ONNX/HuggingFace). "
            "Passez toujours `embeddings=` explicitement."
        )

    @staticmethod
    def name() -> str:
        return "no-auto-embed"

# Composants paresseux : une seule instance par process.
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_embedder = None

def _get_embedder():
    """Embedder Ollama (lazy)."""
    global _embedder
    if _embedder is None:
        from langchain_ollama import OllamaEmbeddings

        _embedder = OllamaEmbeddings(model=EMBED_MODEL)
    return _embedder

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Vectorise une liste de textes via Ollama (local)."""
    vectors = _get_embedder().embed_documents(texts)
    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding mismatch : {len(vectors)} != {len(texts)}")
    return vectors

def embed_query(text: str) -> list[float]:
    """Vectorise une requête via Ollama (local)."""
    return _get_embedder().embed_query(text)

def _get_collection(create: bool = True) -> chromadb.Collection:
    global _client, _collection
    if _collection is not None:
        return _collection

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_NoAutoEmbed(),
        metadata={"hnsw:space": "cosine"},
    )
    return _collection

def chunk_text(t: dict) -> str:
    """Construit le texte indexé pour une technique.

    Le texte mêle nom, tactique, description, guidance de détection et
    mitigations : une seule requête peut nommer l'attaque et retrouver la
    procédure de remédiation.
    """
    lines = [f"{t['id']} — {t['name']}"]

    if t.get("tactics"):
        lines.append(f"Tactic(s): {', '.join(t['tactics'])}")

    if t.get("description"):
        lines.append(f"Description: {t['description']}")

    for s in t.get("detection", [])[:2]:
        lines.append(f"Detection ({s.get('strategy_id', '')}): {s.get('strategy', '')}")
        for a in s.get("analytics", [])[:2]:
            lines.append(f"  - {a['description']}")
            if a.get("logs"):
                lines.append(f"  Logs: {', '.join(a['logs'])}")

    if t.get("mitigations"):
        lines.append(f"Mitigations: {'; '.join(t['mitigations'])}")

    if t.get("url"):
        lines.append(f"Ref: {t['url']}")

    return "\n".join(lines)

def _load_techniques() -> list[dict]:
    if not TECHNIQUES_PATH.exists():
        raise FileNotFoundError(
            f"{TECHNIQUES_PATH} introuvable — lancez d'abord `ingest_attack.py`."
        )
    return json.loads(TECHNIQUES_PATH.read_text(encoding="utf-8"))

def build_index(force: bool = False) -> dict:
    """Indexe les techniques dans ChromaDB (idempotent). Retourne des stats."""
    techniques = _load_techniques()
    col = _get_collection()

    if force:
        col.delete(where={})          # vide la collection

    existing = col.get()
    have = set(existing["ids"])

    todo = [t for t in techniques if t["id"] not in have]
    if todo:
        texts = [chunk_text(t) for t in todo]
        vectors = embed_texts(texts)
        col.add(
            ids=[t["id"] for t in todo],
            documents=texts,
            embeddings=vectors,
            metadatas=[{
                "id": t["id"],
                "name": t["name"],
                "tactics": ", ".join(t.get("tactics", [])),
                "n_detection": len(t.get("detection", [])),
                "n_mitigations": len(t.get("mitigations", [])),
            } for t in todo],
        )

    count = col.count()
    return {
        "techniques_source": len(techniques),
        "indexed": count,
        "added_this_run": len(todo),
        "collection": COLLECTION_NAME,
        "path": str(CHROMA_DIR),
        "embed_model": EMBED_MODEL,
    }

def search(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Recherche sémantique dans l'index ATT&CK.

    Retourne une liste de dicts : {id, name, tactics, text, distance}.
    """
    col = _get_collection()
    if col.count() == 0:
        build_index()

    vec = embed_query(query)
    res = col.query(
        query_embeddings=[vec],
        n_results=min(k, max(col.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    for i, doc_id in enumerate(res["ids"][0]):
        meta = res["metadatas"][0][i]
        hits.append({
            "id": doc_id,
            "name": meta.get("name", ""),
            "tactics": meta.get("tactics", ""),
            "text": res["documents"][0][i],
            "distance": round(float(res["distances"][0][i]), 4),
        })
    return hits

def format_results(hits: list[dict]) -> str:
    """Formate les hits pour injection dans un prompt d'agent."""
    if not hits:
        return "Aucune technique ATT&CK correspondante dans la base locale."

    blocks = []
    for h in hits:
        # Le texte complet contient déjà ID + name + tactics + description +
        # détection + mitigations : pas besoin de le reformater.
        blocks.append(h["text"])
    return "\n\n---\n\n".join(blocks)

def stats() -> dict:
    """État de la base (diagnostic)."""
    col = _get_collection()
    return {
        "indexed": col.count(),
        "collection": COLLECTION_NAME,
        "path": str(CHROMA_DIR),
        "exists": TECHNIQUES_PATH.exists(),
    }

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Index RAG MITRE ATT&CK\n")
    s = build_index()
    print(f"  techniques source : {s['techniques_source']}")
    print(f"  indexees          : {s['indexed']} (+{s['added_this_run']} ce run)")
    print(f"  collection        : {s['collection']}")
    print(f"  chemin            : {s['path']}")
    print(f"  embedding         : {s['embed_model']}\n")

    print("Requête de test : 'failed logon attempts then success from one ip'\n")
    for h in search("failed logon attempts then success from one ip", k=3):
        print(f"  {h['id']:10} {h['name'][:44]:44} d={h['distance']}")

    print("\nRequête de test : 'slow scheduled data exfiltration over time'\n")
    for h in search("slow scheduled data exfiltration over time", k=3):
        print(f"  {h['id']:10} {h['name'][:44]:44} d={h['distance']}")
