"""Ingestion MITRE ATT&CK — sous-ensemble ciblé (Phase 3, RAG).

Télécharge le STIX officiel d'Enterprise ATT&CK (mitre/cti), extrait une
liste curatée de techniques pertinentes pour notre scénario (intrusion
réseau : scan → bruteforce → compromission → exfiltration), y joint les
mitigations et la guidance de détection, et écrit
knowledge/attack_techniques.json.

Particularités du STIX récent (constatées sur enterprise-attack.json) :
  · la prose `x_mitre_detection` a été retirée des attack-pattern, remplacée
    par attack-pattern --(rel "detects")--> x-mitre-detection-strategy -->
    x_mitre_analytic_refs --> x-mitre-analytic (ce sont les analytics qui
    portent le texte et les sources de log) ;
  · "Defense Evasion" est scindée en "Stealth" et "Defense Impairment",
    les phase_name sont donc `stealth` et `defense-impairment` ;
  · certaines familles sont revoked (ex. T1562 "Impair Defenses").

Le STIX brut (~47 Mo) reste dans le cache temporaire : seul le sous-ensemble
enrichi (~250 Ko) est conservé dans le projet.

Usage :  .\\.venv\\Scripts\\python.exe ingest_attack.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)
RAW_PATH = Path(tempfile.gettempdir()) / "enterprise-attack.json"
OUT_PATH = Path(__file__).resolve().parent / "knowledge" / "attack_techniques.json"

MAX_STRATEGIES = 3     # bornage : ne pas gonfler les chunks d'embedding
MAX_ANALYTICS = 3
MAX_DESC_CHARS = 1400

# Sous-ensemble curaté — couvre nos 4 attaques + le contexte d'une intrusion
CURATED_IDS: list[str] = [
    # Reconnaissance / découverte (attaque 2 : scan)
    "T1595", "T1595.001",
    "T1046",                                    # Network Service Discovery <-- scan
    "T1049", "T1016",
    "T1083", "T1087", "T1057", "T1082",

    # Accès aux identifiants (attaque 1 : bruteforce)
    "T1110", "T1110.001", "T1110.003", "T1110.004",
    "T1078", "T1078.002", "T1078.003",
    "T1552", "T1555",

    # Accès initial / mouvement latéral
    "T1133", "T1021", "T1021.004",
    "T1190", "T1199",

    # Exécution / persistance
    "T1059", "T1059.004",
    "T1053", "T1053.005",
    "T1105", "T1106",

    # Évasion / impairment des défenses
    # (T1562 est revoked dans le STIX actuel → remplaçants ci-dessous)
    "T1070", "T1070.004", "T1070.006",
    "T1036", "T1036.005",
    "T1027",
    "T1564", "T1564.001", "T1140", "T1497",

    # Collecte / exfiltration (attaques 3 et 4)
    "T1005", "T1074", "T1074.001",
    "T1560", "T1560.001",
    "T1020",
    "T1029",                                   # Scheduled Transfer    <-- ATTAQUE 4
    "T1030",                                   # Data Transfer Size Limits
    "T1041",                                   # Exfil over C2         <-- ATTAQUE 3
    "T1048", "T1048.001", "T1048.002", "T1048.003",
    "T1011",
    "T1567", "T1567.002",

    # Commande et contrôle (beaconing)
    "T1071", "T1071.001", "T1071.004",
    "T1572", "T1571",
    "T1090", "T1090.003",
    "T1568",
]

def download(force: bool = False) -> Path:
    """Télécharge le STIX Enterprise ATT&CK (une fois, mis en cache)."""
    if RAW_PATH.exists() and not force and RAW_PATH.stat().st_size > 10_000_000:
        print(f"  STIX en cache : {RAW_PATH.name} "
              f"({RAW_PATH.stat().st_size / 1e6:.0f} Mo)")
        return RAW_PATH

    print(f"  Téléchargement STIX ({STIX_URL.split('/')[-1]})…")
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAW_PATH.with_suffix(".part")
    with urllib.request.urlopen(STIX_URL, timeout=180) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(RAW_PATH)
    print(f"  OK : {RAW_PATH.stat().st_size / 1e6:.0f} Mo")
    return RAW_PATH

def technique_id(obj: dict) -> str | None:
    e = next((r for r in obj.get("external_references", [])
              if r.get("source_name") == "mitre-attack" and r.get("external_id")), None)
    return e["external_id"] if e else None

def build_subset(data: dict) -> list[dict]:
    objects = data["objects"]
    by_id = {o["id"]: o for o in objects}

    # Tactique : shortname -> nom affichable
    tactic_name = {
        t.get("x_mitre_shortname"): t.get("name", "")
        for t in objects if t.get("type") == "x-mitre-tactic"
    }

    # Mitigations : relationship "mitigates"
    coa = {o["id"]: o for o in objects if o.get("type") == "course-of-action"}
    mitigations_for: dict[str, list[str]] = {}
    for o in objects:
        if o.get("type") == "relationship" and o.get("relationship_type") == "mitigates":
            src = coa.get(o.get("source_ref", ""))
            if src:
                mitigations_for.setdefault(o.get("target_ref", ""), []).append(
                    src.get("name", ""))

    # Détection : technique -> strategies -> analytics
    # rel "detects" : source = x-mitre-detection-strategy, target = attack-pattern
    strategies_for: dict[str, list[dict]] = {}
    for o in objects:
        if o.get("type") != "relationship" or o.get("relationship_type") != "detects":
            continue
        strat = by_id.get(o.get("source_ref", ""))
        if not strat or strat.get("type") != "x-mitre-detection-strategy":
            continue
        if strat.get("revoked"):
            continue

        ref = (strat.get("external_references") or [{}])[0]
        analytics = []
        for aid in (strat.get("x_mitre_analytic_refs") or [])[:MAX_ANALYTICS]:
            a = by_id.get(aid)
            if not a or not a.get("description"):
                continue
            logs = [r.get("name") for r in
                    (a.get("x_mitre_log_source_references") or []) if r.get("name")]
            analytics.append({
                "name": a.get("name", ""),
                "description": a["description"].strip(),
                "logs": logs[:4],
            })
        if not analytics:
            continue
        strategies_for.setdefault(o.get("target_ref", ""), []).append({
            "strategy": strat.get("name", ""),
            "strategy_id": ref.get("external_id", ""),
            "analytics": analytics,
        })

    # Sélection des techniques curatées
    wanted = set(CURATED_IDS)
    subset: list[dict] = []

    for o in objects:
        if o.get("type") != "attack-pattern" or o.get("revoked"):
            continue
        tid = technique_id(o)
        if tid not in wanted:
            continue

        tactics = sorted({
            tactic_name.get(p.get("phase_name"), p.get("phase_name", "").title())
            for p in o.get("kill_chain_phases", [])
            if p.get("kill_chain_name") == "mitre-attack"
        })

        subset.append({
            "id": tid,
            "name": o.get("name", ""),
            "tactics": tactics,
            "is_subtechnique": bool(o.get("x_mitre_is_subtechnique")),
            "description": (o.get("description") or "").strip()[:MAX_DESC_CHARS],
            "mitigations": sorted(set(mitigations_for.get(o["id"], [])))[:8],
            "detection": strategies_for.get(o["id"], [])[:MAX_STRATEGIES],
            "url": next((r.get("url", "") for r in o.get("external_references", [])
                         if r.get("source_name") == "mitre-attack"), ""),
        })

    subset.sort(key=lambda t: (t["id"].split(".")[0], t["id"]))
    return subset

def main() -> None:
    print("Ingestion MITRE ATT&CK — sous-ensemble curaté\n")
    data = json.loads(download().read_text(encoding="utf-8"))
    subset = build_subset(data)

    found = {t["id"] for t in subset}
    missing = [i for i in CURATED_IDS if i not in found]
    if missing:
        print(f"  ! Absentes (revoked ou introuvables) : {missing}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(subset, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    n_sub = sum(1 for t in subset if t["is_subtechnique"])
    n_det = sum(1 for t in subset if t["detection"])
    n_mit = sum(1 for t in subset if t["mitigations"])
    n_an = sum(len(s["analytics"]) for t in subset for s in t["detection"])
    print(f"\n  {OUT_PATH.name} -> {OUT_PATH.parent}")
    print(f"    {len(subset)} techniques ({n_sub} sous-techniques) "
          f"sur {len(CURATED_IDS)} demandées")
    print(f"    {n_det} avec guidance de détection ({n_an} analytics)")
    print(f"    {n_mit} avec mitigations")
    print(f"    {OUT_PATH.stat().st_size / 1024:.0f} Ko")

    print("\n  Aperçu :")
    for t in subset[:5]:
        det = f" [det:{len(t['detection'])}]" if t["detection"] else ""
        print(f"    {t['id']:10} {t['name'][:46]:46} {','.join(t['tactics'][:2])}{det}")

if __name__ == "__main__":
    main()
