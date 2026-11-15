"""Interface en ligne de commande de l'agent.

Usage :
    py main.py                                   # mode conversationnel (général)
    py main.py "ma question"                     # exécution unique (général)
    py main.py --soc                             # mode analyste SOC, conversationnel
    py main.py --soc "ma question"               # mode analyste SOC, exécution unique
    py generate_logs.py                          # Phase 1 : générer les logs de test
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Console Windows : forcer UTF-8 pour les accents
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage

from agent.graph import build_graph

def _trace(result) -> None:
    """Affiche la trajectoire de l'agent : quels outils, dans quel ordre."""
    for msg in result["messages"]:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                args = ", ".join(f"{k}={str(v)[:70]}" for k, v in tc["args"].items())
                print(f"  > {tc['name']}({args})")
        elif msg.type == "tool":
            content = str(msg.content).replace("\n", " ")
            print(f"      {content[:110]}{'...' if len(content) > 110 else ''}")

def _is_empty_answer(result) -> bool:
    """Vrai si l'agent s'est arrêté sur une réponse vide (ni contenu, ni outil).

    qwen3:8b émet parfois un tour sans tool_call ni contenu : le graphe
    s'arrête alors prématurément et le rapport n'est jamais écrit.
    """
    last = result["messages"][-1]
    if last.type != "ai":
        return False
    content = last.content if isinstance(last.content, str) else str(last.content)
    return not content.strip() and not getattr(last, "tool_calls", None)

_EMPTY_NUDGE = (
    "Ta réponse précédente était vide : ni contenu, ni appel d'outil. "
    "C'est une erreur. Reprends exactement là où tu en étais dans ta méthode "
    "et appelle l'outil suivant. N'écris JAMAIS de réponse vide."
)

def _invoke(graph, payload, config, max_retries: int = 2):
    """Invoque le graphe et relance s'il rend une réponse vide.

    La relance réutilise le même thread : la mémoire conversationnelle
    conserve le travail déjà fait.
    """
    result = graph.invoke(payload, config=config)
    for attempt in range(1, max_retries + 1):
        if not _is_empty_answer(result):
            return result
        print(f"  [!] réponse vide — relance {attempt}/{max_retries}...")
        result = graph.invoke(
            {"messages": [HumanMessage(content=_EMPTY_NUDGE)]}, config=config
        )
    return result

def run_once(graph, question: str, mode: str = "general") -> None:
    """Exécute une seule question et affiche la réponse."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    try:
        result = _invoke(
            graph, {"messages": [HumanMessage(content=question)]}, config
        )
    except Exception as exc:  # noqa: BLE001
        _report_error(exc)
        return

    print("\n--- Trajet de l'agent ---")
    _trace(result)
    print("\n--- Réponse ---")
    last = result["messages"][-1]
    content = last.content if isinstance(last.content, str) else str(last.content)
    if not content.strip() and _is_empty_answer(result):
        print("\nL'agent n'a rien produit, même après relance. "
              "Relancez la commande.")
        return
    print("\n" + content + "\n")

    if mode == "soc":
        _ensure_report(content)

def _ensure_report(content: str) -> None:
    """Garantit workspace/incident_report.md en mode SOC.

    Le livrable est le fichier, pas le texte affiché : si l'agent n'a pas
    appelé write_file mais que sa réponse est complète, on la sauvegarde.
    Les deux cas sont signalés distinctement — sauvegarder en silence
    masquerait un échec de l'agent.
    """
    report = Path(__file__).resolve().parent / "workspace" / "incident_report.md"

    if report.exists():
        print(f"Rapport rédigé par l'agent : {report.name} "
              f"({report.stat().st_size} o)")
        return

    try:
        report.write_text(content, encoding="utf-8")
        print(f"[!] write_file non appelé — réponse sauvegardée "
              f"automatiquement dans {report.name} ({report.stat().st_size} o)")
    except OSError as exc:
        print(f"[erreur] Impossible d'écrire {report.name} : {exc}")

def _report_error(exc: Exception) -> None:
    """Affiche un message d'erreur lisible selon la cause."""
    text = str(exc)
    if "not found" in text and "status code: 404" in text:
        print("\nModèle introuvable dans Ollama.\n   Installez-le :  ollama pull qwen3:8b")
    elif "Could not connect" in text or "Connection refused" in text:
        print("\nOllama ne répond pas. Lancez-le :  ollama serve")
    else:
        print(f"\nErreur : {exc}")

def chat(graph, mode: str = "general") -> None:
    """Boucle conversationnelle interactive."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    titre = "SOC — Analyste de sécurité" if mode == "soc" else "Agent IA polyvalent"
    print("=" * 60)
    print(f"  {titre} — LangGraph + Ollama (local)")
    print("  Tape 'quit' ou 'exit' pour quitter.")
    print("=" * 60)

    while True:
        try:
            question = input("\nVous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Au revoir !")
            break

        try:
            result = _invoke(
                graph,
                {"messages": [HumanMessage(content=question)]},
                config,
            )
            print("\nAgent > " + (result["messages"][-1].content or ""))
        except Exception as exc:  # noqa: BLE001
            _report_error(exc)

def main() -> None:
    args = sys.argv[1:]
    mode = "general"

    if args and args[0] == "--soc":
        mode = "soc"
        args = args[1:]

    graph = build_graph(mode=mode)

    if args:
        run_once(graph, " ".join(args), mode=mode)
    else:
        chat(graph, mode=mode)

if __name__ == "__main__":
    main()
