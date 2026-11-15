"""Démonstration de l'agent polyvalent : enchaînement automatique d'outils.

Chaque cas d'usage montre un chemin différent à travers le graphe :
  1. Recherche web seule
  2. Calcul seul
  3. Écriture puis lecture de fichier (2 appels outils)
  4. Tâche combinée web + fichier + calcul (3+ appels outils)

Usage : .venv\\Scripts\\python.exe demo.py [numero_cas]
"""

from __future__ import annotations

import sys
import uuid

# Console Windows : forcer UTF-8 pour les accents
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage

from agent.graph import build_graph

DEMOS = [
    (
        "1. Recherche web",
        "Qu'est-ce que LangGraph en une phrase ? Utilise la recherche web pour être précis.",
    ),
    (
        "2. Calcul",
        "Calcule (1234 * 5678) / 42 et donne-moi le résultat arrondi à 2 décimales.",
    ),
    (
        "3. Fichiers (écriture puis lecture)",
        "Écris un fichier 'plan.md' contenant un plan d'étude de 5 lignes sur les agents IA, "
        "puis relis-le et confirme-moi son contenu.",
    ),
    (
        "4. Tâche combinée multi-outils",
        " Cherche 3 faits marquants récents sur l'IA, écris-les dans 'faits.md', "
        "puis calcule combien de caractères fait en moyenne chaque fait.",
    ),
]

def run(index: int) -> None:
    graph = build_graph()
    title, question = DEMOS[index]

    print("=" * 70)
    print(f"  CAS {title}")
    print(f"  Question : {question}")
    print("=" * 70)

    # On écoute les nœuds du graphe pour visualiser les appels d'outils
    print("\nExécution...\n")
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )

    # Récapitulatif des étapes
    print("\n--- Trajet de l'agent ---")
    for msg in result["messages"]:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"  > {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            content = str(msg.content)
            preview = content[:80].replace("\n", " ")
            print(f"      {preview}{'...' if len(content) > 80 else ''}")

    print("\n--- Réponse finale ---")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    n = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 3
    if not 0 <= n < len(DEMOS):
        print(f"Choisissez un cas entre 1 et {len(DEMOS)}")
        sys.exit(1)
    run(n)
