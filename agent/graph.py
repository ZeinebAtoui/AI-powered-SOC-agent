"""Graphe de l'agent agentic IA basé sur LangGraph.

Architecture :
    start -> agent (LLM + outils) --(appel d'outil)--> tools -> agent
                      |
                      +--(réponse finale)--> end
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agent.soc import SOC_PROMPT
from agent.tools import TOOLS

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """Tu es un agent IA autonome et polyvalent, francophone.

Tu dispose de plusieurs outils. Utilise-les systématiquement quand la tâche l'exige :
- web_search : toute information factuelle récente ou inconnue de toi.
- read_file / write_file / list_files : tout ce qui touche aux fichiers du workspace.
- calculator : les calculs arithmétiques (ne fait jamais de calcul mental sur des valeurs non triviales).
- run_python : les tâches de traitement de données ou de logique complexe.

Règles :
1. Décompose les tâches complexes en étapes et appelle les outils un par un.
2. Ne jamais inventer le résultat d'un appel d'outil : lis toujours la vraie sortie.
3. Réponds en français, de façon concise et structurée.
4. Si une information manque, cherche-la plutôt que de l'inventer.
5. Termine toujours par un résumé clair de ce que tu as fait."""

PROMPTS = {"general": SYSTEM_PROMPT, "soc": SOC_PROMPT}

def _patch_chat_transport(ChatOllama_cls=None) -> None:
    """Force le non-stream et retire le ``thinking`` de l'historique.

    Deux réglages propres à qwen3:8b :

    1. ``stream=False`` — en mode streamé, qwen3 n'émet aucun ``tool_call``
       (mesuré : 0/3 réponses utiles contre 4/4 en non-stream). LangChain
       streame par défaut même pour ``invoke()``, d'où ce patch de
       ``_chat_params``.

    2. Suppression du ``thinking`` dans l'historique renvoyé — ``reasoning=True``
       est nécessaire à la génération, mais conserver ce raisonnement coûterait
       ~600 tokens par tour et déborderait la fenêtre sur une longue enquête.
    """
    cls = ChatOllama_cls or ChatOllama
    if getattr(cls, "_transport_patched", False):
        return

    orig_params = cls._chat_params

    def _params(self, messages, stop=None, **kwargs):
        kwargs.setdefault("stream", False)
        return orig_params(self, messages, stop, **kwargs)

    cls._chat_params = _params

    orig_convert = cls._convert_messages_to_ollama_messages

    def _converted(self, messages, *args, **kwargs):
        out = orig_convert(self, messages, *args, **kwargs)
        for m in out:
            if isinstance(m, dict):
                m.pop("thinking", None)
            elif getattr(m, "thinking", None):
                try:
                    m.thinking = ""
                except (AttributeError, ValueError):  # pragma: no cover
                    pass
        return out

    cls._convert_messages_to_ollama_messages = _converted
    cls._transport_patched = True

_patch_chat_transport()

def build_llm() -> ChatOllama:
    """Crée le LLM local branché sur Ollama.

    - ``num_ctx=16384`` : fenêtre de contexte (défaut Ollama 4096, trop juste
      pour enchaîner les appels d'outils d'une enquête SOC).
    - ``reasoning=True`` : indispensable avec qwen3 — en ``False``, Ollama jette
      le raisonnement que le modèle génère quand même et la réponse sort vide.
    - ``num_predict=8192`` : évite la troncature d'un appel d'outil long, qui
      produirait un JSON invalide et arrêterait le graphe.
    """
    return ChatOllama(
        model=MODEL,
        base_url=BASE_URL,
        temperature=0.1,
        num_ctx=16384,
        num_predict=8192,
        reasoning=True,
    )

def build_graph(mode: str = "general"):
    """Construit le graphe réactif de l'agent avec mémoire conversationnelle.

    Args:
        mode: "general" (assistant polyvalent) ou "soc" (analyste sécurité).
    """
    if mode not in PROMPTS:
        raise ValueError(f"Mode inconnu : {mode!r} (attendu : {', '.join(PROMPTS)})")

    agent = create_react_agent(
        build_llm(),
        tools=TOOLS,
        prompt=PROMPTS[mode],
        checkpointer=MemorySaver(),
    )
    return agent
