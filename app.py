"""Dashboard SOC — FastAPI + page web.

Expose l'architecture hybride (Isolation Forest + XGBoost + RAG) derrière une
interface : c'est le point de restitution du pipeline complet.

Endpoints :
    GET  /               -> page du dashboard (static/index.html)
    GET  /api/status     -> état des modèles, des journaux, du rapport
    GET  /api/detection  -> payload de detect_hybrid() (mis en cache)
    POST /api/refresh    -> invalide le cache et recalcule
    GET  /api/report     -> workspace/incident_report.md, s'il existe
    POST /api/ask        -> lance une enquête de l'agent en arrière-plan
    GET  /api/job/{id}   -> état d'une enquête en cours

Usage :
    python app.py          # puis ouvrir http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
WORKSPACE = ROOT / "workspace"
REPORT = WORKSPACE / "incident_report.md"
PORT = 5000

app = FastAPI(title="Dashboard SOC — agent hybride", version="1.0")

# detect_hybrid() coûte quelques secondes : on ne le recalcule pas à chaque
# rafraîchissement de page, d'où le cache partagé.
_cache: dict | None = None
_lock = threading.Lock()

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")

@app.get("/api/status")
def status() -> dict:
    """État des trois briques + des entrées/sorties du pipeline."""
    from agent.hybrid import models_available

    logs_dir = WORKSPACE / "logs"
    logs = {}
    if logs_dir.exists():
        for name in ("auth.log", "firewall.log", "app.log"):
            p = logs_dir / name
            logs[name] = p.stat().st_size if p.exists() else None

    ollama = False
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        ollama = True
    except Exception:  # noqa: BLE001
        ollama = False

    rag_ok = (ROOT / "knowledge" / "chroma").exists()
    xgb_ok = models_available()

    return {
        "components": {
            "isolation_forest": {"ready": True,
                                 "note": "fit à chaque appel, entraîné sur le normal seul"},
            "xgboost": {"ready": xgb_ok,
                        "note": "models/xgboost.json — entraîné sur UNSW-NB15"},
            "rag_attack": {"ready": rag_ok,
                           "note": "knowledge/chroma — 63 techniques MITRE"},
            "ollama": {"ready": ollama, "note": "serveur local qwen3:8b"},
        },
        "logs": logs,
        "report_exists": REPORT.exists(),
        "report_size": REPORT.stat().st_size if REPORT.exists() else 0,
    }

@app.get("/api/detection")
def detection(refresh: bool = False) -> JSONResponse:
    """Payload de la détection hybride, mis en cache."""
    global _cache

    from agent.hybrid import detect_hybrid, models_available

    if not models_available():
        return JSONResponse(
            status_code=503,
            content={"error": "XGBoost non entraîné",
                     "hint": "lancer : python train_xgboost"},
        )

    with _lock:
        if _cache is None or refresh:
            try:
                _cache = detect_hybrid()
            except Exception as exc:  # noqa: BLE001
                return JSONResponse(
                    status_code=500,
                    content={"error": f"{type(exc).__name__}: {exc}"},
                )
    return JSONResponse(content=_cache)

@app.post("/api/refresh")
def refresh() -> dict:
    """Invalide le cache : la prochaine lecture /api/detection recalcule."""
    global _cache
    with _lock:
        _cache = None
    return {"ok": True, "message": "cache invalidé"}

@app.get("/api/report")
def report() -> JSONResponse:
    """Le rapport d'incident produit par l'agent, s'il existe."""
    if not REPORT.exists():
        return JSONResponse(
            status_code=404,
            content={"exists": False,
                     "hint": "aucun workspace/incident_report.md — "
                             "lancer : python main.py --soc \"...\""},
        )
    return JSONResponse(content={
        "exists": True,
        "size": REPORT.stat().st_size,
        "content": REPORT.read_text(encoding="utf-8", errors="replace"),
    })

# Une enquête SOC dure 15 à 45 minutes (chaque tour de LLM coûte 30-60 s) :
# le graphe tourne donc dans un thread, et le front interroge l'état du job
# toutes les 2 s plutôt que d'attendre la fin de la requête HTTP.

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_MAX_LOG = 500

def _new_job(question: str, mode: str) -> tuple[dict | None, str | None]:
    """Crée un job, de façon atomique.

    Retourne ``(job, None)`` si la création a réussi, ou ``(None, id)`` si une
    enquête tourne déjà. Le verrou n'est acquis qu'une fois ici : ``_jobs_lock``
    n'est pas réentrant, un appel imbriqué bloquerait l'appelant.
    """
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "question": question,
        "mode": mode,
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "turns": 0,
        "log": [],
        "answer": None,
        "error": None,
    }
    with _jobs_lock:
        for existing in _jobs.values():
            if existing["status"] == "running":
                return None, existing["id"]
        _jobs[job_id] = job
    return job, None

def _describe(msg) -> list[dict]:
    """Traduit un message LangChain en entrées de journal.

    Un message AI peut porter plusieurs ``tool_calls`` : on en retourne une
    entrée par appel.
    """
    mtype = getattr(msg, "type", "")

    if mtype == "ai":
        out = []
        for call in (getattr(msg, "tool_calls", None) or []):
            args = json.dumps(call.get("args", {}), ensure_ascii=False)
            if len(args) > 400:
                args = args[:400] + "…"
            out.append({"kind": "tool", "name": call.get("name", "?"),
                        "args": args})
        # Un AI sans tool_call est la réponse finale : elle est exposée dans
        # job["answer"], pas dans le journal.
        return out

    if mtype == "tool":
        text = msg.content or ""
        if len(text) > 500:
            text = text[:500] + "…"
        return [{"kind": "result", "name": getattr(msg, "name", ""), "text": text}]

    return []

def _run_agent(job: dict) -> None:
    """Exécute le graphe dans un thread et journalise chaque nœud."""
    try:
        from langchain_core.messages import HumanMessage

        from agent.graph import build_graph

        graph = build_graph(mode=job["mode"])
        config = {"configurable": {"thread_id": job["id"]}}
        payload = {"messages": [HumanMessage(content=job["question"])]}

        # stream_mode="values" rend l'état complet après chaque nœud : on ne
        # journalise que les messages pas encore vus.
        seen = 0
        for state in graph.stream(payload, config, stream_mode="values"):
            messages = list((state or {}).get("messages") or [])
            for msg in messages[seen:]:
                for entry in _describe(msg):
                    with _jobs_lock:
                        if len(job["log"]) < _MAX_LOG:
                            job["log"].append(entry)
                        if entry["kind"] == "tool":
                            job["turns"] += 1
            seen = len(messages)

        # Réponse finale = dernier message AI sans tool_call.
        state = graph.get_state(config)
        messages = list(state.values.get("messages") or [])
        answer = ""
        for msg in reversed(messages):
            if (getattr(msg, "type", "") == "ai"
                    and not (getattr(msg, "tool_calls", None) or [])):
                answer = msg.content or ""
                break
        if not answer and messages:
            answer = getattr(messages[-1], "content", "") or ""

        job["answer"] = answer

        # Le livrable est le fichier, pas l'écran : même filet de sécurité
        # que la CLI, en cas d'oubli de write_file().
        if job["mode"] == "soc" and answer:
            from main import _ensure_report
            _ensure_report(answer)

        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["finished_at"] = time.time()

@app.post("/api/ask")
def ask(payload: dict | None = None) -> JSONResponse:
    """Lance une question à l'agent, en arrière-plan (réponse immédiate)."""
    payload = payload or {}
    question = str(payload.get("question") or "").strip()
    mode = str(payload.get("mode") or "soc")
    if mode not in {"soc", "general"}:
        mode = "soc"

    if not question:
        return JSONResponse(status_code=400, content={"error": "Question vide."})

    # _new_job acquiert _jobs_lock lui-même : ne pas le tenir ici, sinon
    # appel imbriqué sur un verrou non réentrant.
    job, running_id = _new_job(question, mode)
    if job is None:
        return JSONResponse(status_code=409, content={
            "error": "Une enquête est déjà en cours.",
            "running": running_id,
        })

    threading.Thread(target=_run_agent, args=(job,), daemon=True).start()
    return JSONResponse(content={"id": job["id"], "status": "running"})

@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    """État d'un job : journal vivant, réponse, erreur."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "Job inconnu."})
        # Copie sous verrou : le thread continue d'écrire pendant la lecture.
        finished = job["finished_at"]
        snapshot = {
            "id": job["id"],
            "question": job["question"],
            "mode": job["mode"],
            "status": job["status"],
            "turns": job["turns"],
            "elapsed": round((finished or time.time()) - job["started_at"], 1),
            "log": list(job["log"]),
            "answer": job["answer"],
            "error": job["error"],
        }
    return JSONResponse(content=snapshot)

def main() -> None:
    import uvicorn

    print("=" * 68)
    print("DASHBOARD SOC")
    print("=" * 68)
    print(f"  URL      : http://127.0.0.1:{PORT}")
    print(f"  statique : {STATIC}")
    print(f"  rapport  : {REPORT}")
    print("\n  Ctrl+C pour arrêter.")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

if __name__ == "__main__":
    main()
