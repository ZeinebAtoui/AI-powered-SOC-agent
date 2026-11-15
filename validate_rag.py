"""Validation Phase 3 — l'agent utilise-t-il vraiment le RAG ATT&CK ?

Critère d'acceptation, en 2 niveaux :

  A. L'outil seul renvoie la bonne technique (aucun LLM, instantané).
  B. L'agent l'appelle de lui-même et cite une technique officielle
     dans sa réponse — sans l'inventer.

B est le vrai test : c'est lui qui prouve que la Phase 3 est branchée dans
le pipeline, et pas seulement construite en isolé. On teste avec une consigne
courte qui ne demande pas d'analyser les journaux : le critère porte sur le
RAG, pas sur la capacité à parser 3 x 2 Mo de logs (couvert par la Phase 1/2).

Usage :  .\\.venv\\Scripts\\python.exe validate_rag.py
"""

from __future__ import annotations

import re
import sys
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from agent.rag import search as rag_search

SEP = "=" * 88

# Prompt volontairement court : aucune lecture de fichier, aucun calcul.
# On veut isoler la question « l'agent appelle-t-il le RAG ? ».
PROMPT = (
    "Utilise search_attack_knowledge pour identifier la technique ATT&CK d'un "
    "attaquant qui exfiltre des données lentement et régulièrement, sur "
    "plusieurs jours, pour passer inaperçu. Donne l'identifiant, le nom et la "
    "tactique."
)

# Plusieurs réponses acceptables par cas : une technique admet souvent deux
# lectures selon le contexte (scan externe -> T1595 Active Scanning en
# Reconnaissance ; découverte depuis l'intérieur du réseau -> T1046 Network Service
# Discovery en Discovery). Exiger UNE seule id ferait échouer un RAG juste.
CASES = [
    ("slow scheduled data exfiltration over time", {"T1029", "T1041", "T1048"}),
    ("failed logon attempts then success from one ip", {"T1110", "T1078", "T1021"}),
    ("port scan many ports on a single host", {"T1046", "T1595"}),
]

def _norm(tid: str) -> str:
    """T1595.001 -> T1595 : compare au parent pour un id de haut niveau."""
    return tid.split(".")[0]

def check_a() -> bool:
    """A. La recherche retourne-t-elle une technique acceptable ? (sans LLM)

    On interroge `rag.search` (structurel) plutôt que le texte formaté de
    l'outil : `format_results` ajoute une section « techniques liées » qui
    polluerait une regex posée sur la sortie complète.
    """
    print(SEP)
    print("A. RECHERCHE SEULE — une technique acceptable ressort-elle en top-5 ?")
    print(SEP)

    ok = True
    for query, acceptable in CASES:
        hits = rag_search(query, k=5)
        returned = {_norm(h.get("technique_id") or h.get("id") or "") for h in hits}
        want = {_norm(t) for t in acceptable}
        found = returned & want
        ok &= bool(found)
        print(f"  {'OK ' if found else 'KO '} « {query[:50]:50s} »")
        print(f"       acceptable {sorted(want)} | top-5 {sorted(returned)}")
        print(f"       intersection : {sorted(found) or 'VIDE'}")
    return ok

def check_b() -> bool:
    """B. L'AGENT appelle l'outil et cite une technique officielle."""
    print("\n" + SEP)
    print("B. AGENT — appelle-t-il search_attack_knowledge() et cite-t-il une technique ?")
    print(SEP)

    graph = build_graph(mode="soc")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke({"messages": [HumanMessage(content=PROMPT)]}, config=config)

    msgs = result["messages"]

    # Quels outils l'agent a-t-il réellement appelés ?
    called: list[str] = []
    for m in msgs:
        if m.type == "ai":
            for tc in (getattr(m, "tool_calls", None) or []):
                called.append(tc["name"])

    # Sortie réelle de l'outil (pas juste l'intention de l'appeler)
    rag_outputs = [str(m.content) for m in msgs if m.type == "tool"
                   and "T1" in str(m.content)]

    last = msgs[-1]
    final = last.content if isinstance(last.content, str) else str(last.content)

    cited = set(re.findall(r"\bT\d{4}(?:\.\d{3})?\b", final))
    # Une technique citée doit être étayée par une vraie sortie d'outil
    grounded = cited & set(re.findall(r"\bT\d{4}(?:\.\d{3})?\b", " ".join(rag_outputs)))

    n_rag = called.count("search_attack_knowledge")
    print(f"  outils appelés      : {called or 'AUCUN'}")
    print(f"  appels RAG          : {n_rag}")
    print(f"  sorties RAG utiles  : {len(rag_outputs)}")
    print(f"  techniques citées   : {sorted(cited) or 'aucune'}")
    print(f"  étayées par un outil: {sorted(grounded) or 'aucune'}")
    print(f"  réponse finale ({len(final)} car.) : {final[:150]!r}")

    ok = n_rag >= 1 and bool(cited) and bool(grounded)
    print(f"\n  {'>> OK' if ok else '>> ECHEC'} — "
          + ("l'agent a appelé le RAG et cité une technique prouvée."
             if ok else
             "l'agent n'a pas démontré l'usage du RAG (appel manquant, "
             "technique absente, ou technique citée sans sortie d'outil)."))
    return ok

def main() -> None:
    a = check_a()
    b = check_b()

    print("\n" + SEP)
    print("SYNTHÈSE")
    print(SEP)
    print(f"  A. outil seul            : {'PASS' if a else 'FAIL'}")
    print(f"  B. intégration agent     : {'PASS' if b else 'FAIL'}")
    print(f"\n  Phase 3 : {'VALIDÉE' if (a and b) else 'NON VALIDÉE'}")
    if not (a and b):
        sys.exit(1)

if __name__ == "__main__":
    main()
