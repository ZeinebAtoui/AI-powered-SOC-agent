"""Outils exposés à l'agent agentic IA."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

MAX_FILE_BYTES = 1_000_000
MAX_OUTPUT_CHARS = 4_000

def _safe_path(relative: str) -> Path:
    """Résout un chemin relatif en s'assurant qu'il reste dans le workspace."""
    target = (WORKSPACE / relative).resolve()
    if not str(target).startswith(str(WORKSPACE.resolve())):
        raise ValueError(f"Accès refusé : {relative} est hors du workspace.")
    return target

def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Borne une sortie d'outil pour ne pas saturer le contexte du LLM."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    return (
        f"{head}\n\n[TRONQUÉ — {len(text)} caractères au total. "
        f"Affiche les {limit} premiers. Refine ta requête pour obtenir le reste "
        f"(filtre, agrégation, comptage…)]."
    )

@tool
def web_search(query: str) -> str:
    """Recherche une information sur le web via DuckDuckGo."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Erreur : ddgs n'est pas installé (pip install ddgs)."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as exc:  # noqa: BLE001 - l'erreur doit revenir au LLM
        return f"Erreur de recherche web : {exc}"

    if not results:
        return "Aucun résultat trouvé."

    return _truncate(
        json.dumps(
            [{"titre": r.get("title"), "url": r.get("href"), "extrait": r.get("body")} for r in results],
            ensure_ascii=False,
            indent=2,
        ),
        limit=3_000,
    )

@tool
def read_file(filename: str) -> str:
    """Lit le contenu d'un fichier du workspace (refuse les fichiers > 1 Mo)."""
    path = _safe_path(filename)
    if not path.exists():
        return f"Fichier introuvable : {filename}"
    if path.stat().st_size > MAX_FILE_BYTES:
        return (
            f"Fichier trop volumineux (> {MAX_FILE_BYTES} octets) : {filename}. "
            f"Utilise run_python pour l'analyser par agrégation (comptage, filtres, "
            f"top N…) sans le charger en mémoire."
        )
    return _truncate(path.read_text(encoding="utf-8", errors="replace"))

@tool
def write_file(filename: str, content: str) -> str:
    """Écrit dans un fichier du workspace (chemin relatif, crée si absent)."""
    path = _safe_path(filename)
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        return f"Contenu trop volumineux (> {MAX_FILE_BYTES} octets)."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Fichier écrit : {filename} ({len(content)} caractères)."

@tool
def list_files() -> str:
    """Liste tous les fichiers présents dans le workspace, avec leur taille."""
    entries = [
        f"{p.relative_to(WORKSPACE)} ({p.stat().st_size} o)"
        for p in sorted(WORKSPACE.rglob("*"))
        if p.is_file()
    ]
    return "\n".join(entries) if entries else "Le workspace est vide."

@tool
def calculator(expression: str) -> str:
    """Évalue une expression arithmétique simple (ex : \"2 + 2 * 10\")."""
    allowed = set("0123456789+-*/().% ")
    if not expression or any(ch not in allowed for ch in expression):
        return "Expression invalide : seuls les chiffres et opérateurs + - * / ( ) % sont autorisés."
    try:
        return str(eval(compile(expression, "<calc>", "eval"), {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001
        return f"Erreur de calcul : {exc}"

@tool
def run_python(code: str) -> str:
    """Exécute un snippet Python et retourne la sortie standard.

    Sers-to-en pour tout traitement de données : les autres outils ne
    couvrent ni l'agrégation ni le filtrage. Fais toujours des print() —
    un appel sans sortie ne prouve rien.
    """
    import io
    import contextlib

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "<agent>", "exec"), {"__builtins__": __builtins__})  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return f"Erreur Python : {type(exc).__name__}: {exc}"
    output = buffer.getvalue()
    if not output.strip():
        return "(exécution terminée, aucune sortie)"
    return _truncate(output.strip())

@tool
def detect_anomalies(contamination: float = 0.02) -> str:
    """Détection hybride : l'Isolation Forest détecte, le XGBoost nomme, la RAG habille.

    - Isolation Forest (non supervisé, entraîné sur le normal seul) score les
      profils (IP, heure) : c'est le seul décideur, lui seul crée une alerte.
    - XGBoost (supervisé sur UNSW-NB15) ne fait que promouvoir une alerte
      existante en priorité haute (p_attack >= 0.90), jamais en créer une.
    - RAG ATT&CK traduit la famille prédite en technique officielle.

    À utiliser pour : faible volume anormal, combinaison de features
    inhabituelles, attaque lente et discrète que des règles simples ne
    repèrent pas.

    Le split d'évaluation n'est pas paramétrable : c'est un protocole de
    mesure (frontière juste avant la 1re attaque, entraînement sur jours
    benignes) et non un réglage d'analyste — une frontière choisie ailleurs
    ferait tomber des attaques côté entraînement, où le modèle ne peut pas
    les détecter.

    Returns:
        JSON ``{summary, evaluation, alerts[]}``. Chaque alerte porte ip,
        hour, anomaly_score, priority ('haute' = les deux modèles convergent,
        'normale' = seul l'IF alerte, typique des attaques subtiles),
        agreement, features, xgboost {family, p_attack_max} et
        attack_technique {id, name, tactics} si la famille a un équivalent.
    """
    from agent.detect import DEFAULT_SPLIT_DATE
    from agent.hybrid import detect_hybrid, models_available

    logs_dir = WORKSPACE / "logs"
    if not logs_dir.exists():
        return "Aucun répertoire workspace/logs à analyser."

    sd = DEFAULT_SPLIT_DATE

    if not models_available():
        return (
            "XGBoost non entraîné : lancer `python train_xgboost` puis "
            "relancer. (L'Isolation Forest seule reste accessible via "
            "agent.detect.fit_and_score.)"
        )

    try:
        payload = detect_hybrid(logs_dir, contamination=float(contamination),
                                split_date=sd)
    except Exception as exc:  # noqa: BLE001
        return f"Erreur lors de la détection hybride : {type(exc).__name__}: {exc}"

    # JSON compact, jamais coupé au milieu : on retire des alertes par la fin
    # (scores les plus bas, l'IF étant trié décroissant) jusqu'à tenir dans
    # la limite. Plancher à 12 : les 4 attaques y sont toutes, la plus basse
    # étant au rang 11.
    limit = 12_000
    text = json.dumps(payload, ensure_ascii=False)
    while len(text) > limit and len(payload["alerts"]) > 12:
        payload["alerts"] = payload["alerts"][:-1]
        payload["summary"]["shown"] = len(payload["alerts"])
        text = json.dumps(payload, ensure_ascii=False)
    return _truncate(text, limit=limit)

@tool
def search_attack_knowledge(query: str, k: int = 4) -> str:
    """Recherche une technique MITRE ATT&CK dans la base locale (63 techniques).

    À utiliser après avoir identifié un comportement suspect, pour nommer
    l'attaque avec le bon identifiant (ex. T1110), récupérer la guidance de
    détection et les mitigations.

    Décris les faits vus dans les logs en langage naturel (ex. "failed logon
    attempts then a success from one external IP").
    """
    from agent.rag import format_results, search

    if not query or not query.strip():
        return "Requête vide : décris le comportement observé."

    try:
        hits = search(query.strip(), k=max(1, min(int(k), 8)))
    except FileNotFoundError as exc:
        return f"Base ATT&CK absente : {exc}"
    except Exception as exc:  # noqa: BLE001 - l'erreur doit revenir au LLM
        return f"Erreur de recherche ATT&CK : {type(exc).__name__}: {exc}"

    return _truncate(format_results(hits), limit=6_000)

TOOLS = [
    web_search,
    read_file,
    write_file,
    list_files,
    calculator,
    run_python,
    detect_anomalies,
    search_attack_knowledge,
]
