"""Architecture hybride : Isolation Forest (détection) + XGBoost (nommage).

Rôles asymétriques — les deux modèles ne répondent pas à la même question et
ne sont pas en concurrence. La fusion est à sens unique : sur nos journaux le
XGBoost a un taux de faux positifs de 24,6 % (contre 3,1 % sur UNSW), parce que
notre distribution d'octets est ~100× celle d'UNSW. Lui donner le droit
d'alerter seul produirait ~3 400 fausses alertes.

    L'IF décide QUOI est suspect  ->  il est le seul détecteur
    Le XGBoost dit QUOI C'EST     ->  il ne fait que confirmer et nommer

    IF seul            -> alerte, priorité "normale"
    IF + XGBoost       -> alerte, priorité "haute", famille nommée
    XGBoost seul       -> pas d'alerte, mais consigné (xgb_solo)

Le désaccord ne supprime jamais une alerte de l'IF : sinon on perdrait les
attaques subtiles (A2 scan, A4 low & slow) que le XGBoost rate.

Complémentarité par granularité :
    IF       profil (IP, heure)  -> voit A2 (269 ports en 1 h = agrégat)
                                     voit A4 (4 conn -> 1 hôte = pattern)
    XGBoost  flux individuel     -> voit A3 (10,5 Mo/flux = volumétrie)

Usage :
    python -m agent.hybrid
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.detect import (
    DEFAULT_CONTAMINATION,
    DEFAULT_SEED,
    DEFAULT_SPLIT_DATE,
    fit_and_score,
)
from agent.features import FEATURE_COLS
from agent.flow_features import flows_from_firewall
from agent.preprocess import Preprocessor

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "xgboost.json"
CLASSES_PATH = MODELS_DIR / "classes.json"
PRE_PATH = MODELS_DIR / "preprocess.json"

# Seuil d'accord entre les deux modèles : à 0,5 le FPR est de 24,6 % (inutilisable),
# à 0,9 on ne garde que des flux quasi certains. Seuil de confiance, pas de
# détection : il ne crée jamais d'alerte, il ne fait que promouvoir une
# alerte existante en priorité haute.
AGREEMENT_THRESHOLD = 0.90

DEFAULT_LIMIT = 15
NORMAL_CLASS = "Normal"

def models_available() -> bool:
    """Le XGBoost est-il entraîné et prêt ?"""
    return all(p.exists() for p in (MODEL_PATH, CLASSES_PATH, PRE_PATH))

def _load_models():
    """Charge le classifieur, son vocabulaire de classes et le prétraitement."""
    import xgboost as xgb

    meta = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    classes: list[str] = list(meta["classes"])
    normal: str = meta.get("normal", NORMAL_CLASS)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    pre = Preprocessor.load(PRE_PATH)
    return model, classes, normal, pre

def score_flows(flows: pd.DataFrame) -> pd.DataFrame:
    """Ajoute à chaque flux : p_attack, famille prédite et sa probabilité.

    p_attack = 1 - P(Normal) : signal de confirmation.
    family   = argmax des classes d'attaque : le nom.
    """
    model, classes, normal, pre = _load_models()
    normal_idx = classes.index(normal)

    proba = model.predict_proba(pre.transform(flows))
    p_attack = 1.0 - proba[:, normal_idx]

    # Nom : meilleure classe d'attaque (on exclut Normal).
    attack_idx = [i for i, c in enumerate(classes) if c != normal]
    sub = proba[:, attack_idx]
    best_local = sub.argmax(axis=1)
    family = np.array([classes[attack_idx[i]] for i in best_local])
    family_prob = sub[np.arange(len(sub)), best_local]

    out = flows.copy()
    out["p_attack"] = p_attack
    out["family"] = family
    out["family_prob"] = family_prob
    return out

def _hour_of(ts: str) -> str:
    """'2026-10-10T00:00:28+01:00' -> '2026-10-10 00:00:00'.

    Même format que `_hour_key()` de agent/features.py : sans cette égalité
    stricte, la jointure IF / XGBoost ne trouverait aucune ligne commune et
    le hybride serait silencieusement vide.
    """
    return f"{ts[:10]} {ts[11:13]}:00:00"

def aggregate_by_profile(scored: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les scores de flux en signaux par profil (IP, heure).

    Un profil IF peut contenir des flux où l'IP est source OU destination :
    on déplie les deux côtés, puis on agrège. La famille retenue est celle du
    flux le plus confiant (p_attack max) — les flux neutres le pollueraient.
    """
    parts = []
    for side in ("src", "dst"):
        part = pd.DataFrame({
            "ip": scored[side].to_numpy(),
            "hour": [_hour_of(t) for t in scored["ts"]],
            "p_attack": scored["p_attack"].to_numpy(),
            "family": scored["family"].to_numpy(),
            "family_prob": scored["family_prob"].to_numpy(),
        })
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)

    agg = long.groupby(["ip", "hour"]).agg(
        n_flows=("p_attack", "size"),
        xgb_p_attack_max=("p_attack", "max"),
        xgb_p_attack_mean=("p_attack", "mean"),
    )

    top_idx = long.groupby(["ip", "hour"])["p_attack"].idxmax()
    top = long.loc[top_idx].set_index(["ip", "hour"])
    agg["xgb_family"] = top["family"]
    agg["xgb_family_prob"] = top["family_prob"]
    return agg

def fuse(if_table: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Joint l'IF et le XGBoost, puis applique les règles de priorité.

        IF + accord      -> priorité "haute"
        IF sans accord   -> priorité "normale"   (le XGBoost ne peut pas veto)
        XGBoost sans IF  -> xgb_solo=True, PAS d'alerte
    """
    fused = if_table.join(agg, how="left")

    for col, fill in (("n_flows", 0), ("xgb_p_attack_max", 0.0),
                      ("xgb_p_attack_mean", 0.0),
                      ("xgb_family_prob", 0.0)):
        fused[col] = fused[col].fillna(fill)
    fused["xgb_family"] = fused["xgb_family"].fillna("-")

    agree = fused["xgb_p_attack_max"] >= AGREEMENT_THRESHOLD
    flagged = fused["anomaly"].astype(bool)

    fused["agreement"] = flagged & agree
    fused["xgb_solo"] = (~flagged) & agree
    fused["priority"] = np.where(
        ~flagged, "aucune",
        np.where(fused["agreement"], "haute", "normale"),
    )
    return fused

def detect_hybrid(
    logs_dir: Path | str = "workspace/logs",
    contamination: float = DEFAULT_CONTAMINATION,
    split_date: str | None = DEFAULT_SPLIT_DATE,
    limit: int = DEFAULT_LIMIT,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Pipeline hybride complet -> dictionnaire structuré (prêt pour l'UI)."""
    if_table = fit_and_score(logs_dir, contamination=contamination,
                             seed=seed, split_date=split_date)
    if if_table.empty:
        return {"alerts": [], "summary": {"profiles": 0},
                "message": "Aucun profil construit depuis les journaux."}

    flows = flows_from_firewall(logs_dir)
    scored = score_flows(flows)
    agg = aggregate_by_profile(scored)
    fused = fuse(if_table, agg)

    flagged = fused[fused["anomaly"].astype(bool)]
    shown = flagged.head(limit)

    alerts = []
    for (ip, hour), row in shown.iterrows():
        active = {
            col: (int(row[col]) if float(row[col]).is_integer()
                  else round(float(row[col]), 2))
            for col in FEATURE_COLS
            if col in row.index and float(row[col]) != 0
        }
        alerts.append({
            "ip": ip,
            "hour": hour,
            "anomaly_score": float(row["anomaly_score"]),
            "priority": str(row["priority"]),
            "agreement": bool(row["agreement"]),
            "features": active,
            "xgboost": {
                "family": str(row["xgb_family"]),
                "family_probability": round(float(row["xgb_family_prob"]), 4),
                "p_attack_max": round(float(row["xgb_p_attack_max"]), 4),
                "p_attack_mean": round(float(row["xgb_p_attack_mean"]), 4),
                "n_flows": int(row["n_flows"]),
                "contributed": bool(row["agreement"] or row["xgb_solo"]),
            },
        })

    enrich_with_attack(alerts)

    payload = {
        "architecture": "hybride IsolationForest (détection) + XGBoost (nommage)",
        "fusion_rule": (
            "L'IF est le seul détecteur. Le XGBoost ne promeut une alerte en "
            f"priorité haute que si p_attack >= {AGREEMENT_THRESHOLD} ; il ne "
            "peut jamais en créer une seule (FPR 24,6 % sur nos journaux)."
        ),
        "evaluation": if_table.attrs.get(
            "split", {"mode": "unknown"},
        ),
        "summary": {
            "hourly_profiles": int(len(fused)),
            "flagged": int(fused["anomaly"].sum()),
            "shown": len(alerts),
            "priority_haute": int((fused["priority"] == "haute").sum()),
            "priority_normale": int((fused["priority"] == "normale").sum()),
            "xgb_solo_not_alerted": int(fused["xgb_solo"].sum()),
            "xgb_flows_scored": int(len(scored)),
            "with_technique": sum(1 for a in alerts if a["attack_technique"]),
        },
        "alerts": alerts,
        "interpretation": (
            "priority 'haute' = les deux modèles convergent (IF + XGBoost). "
            "priority 'normale' = seul l'IF alerte — c'est le cas des attaques "
            "subtiles que le XGBoost ne sait pas voir. xgb_solo_not_alerted = "
            "signaux XGBoost non retenus, pour ne pas noyer le rapport."
        ),
    }
    return payload

# Le XGBoost produit une famille (vocabulaire UNSW, 9 classes), la RAG une
# technique ATT&CK : les deux vocabulaires ne coïncident pas, d'où la table.
# 3 familles sur 9 n'ont pas d'équivalent, et on le dit plutôt que d'inventer
# une technique :
#   Fuzzers  -> technique de test, aucun comportement d'adversaire associé
#   Analysis -> catégorie propre à UNSW, pas une tactique MITRE
#   Generic  -> trop vague : l'agent affinera avec les faits concrets
FAMILY_RAG_QUERY = {
    "Reconnaissance": "active scanning probing for open ports and network services",
    "DoS": "denial of service flood overwhelming the target service",
    "Exploits": "exploitation of a public-facing application vulnerability",
    "Backdoor": "backdoor web shell persistent unauthorized access",
    "Shellcode": "shellcode execution injected into memory",
    "Worms": "self-propagating worm spreading laterally between hosts",
    "Fuzzers": "",
    "Analysis": "",
    "Generic": "",
    "Normal": "",
}

def enrich_with_attack(alerts: list[dict], k: int = 2) -> list[dict]:
    """Ajoute à chaque alerte la technique ATT&CK de sa famille XGBoost.

    Une seule requête RAG par famille unique (3 à 6 en pratique), donc quelques
    appels d'embedding et non un par alerte. ``suggested_attack_query`` est
    exposé même sans résultat : c'est ce que l'agent peut relancer lui-même
    via search_attack_knowledge() s'il veut affiner avec les faits.
    """
    from agent.rag import search

    cache: dict[str, list[dict]] = {}
    for alert in alerts:
        family = alert["xgboost"]["family"]
        query = FAMILY_RAG_QUERY.get(family, "")
        alert["suggested_attack_query"] = query or None
        alert["attack_technique"] = None

        if not query:
            continue

        if family not in cache:
            try:
                cache[family] = search(query, k=k)
            except Exception:  # noqa: BLE001 - la RAG ne doit pas casser l'outil
                cache[family] = []

        hits = cache[family]
        if not hits:
            continue
        top = hits[0]
        alert["attack_technique"] = {
            "id": top.get("id", ""),
            "name": top.get("name", ""),
            "tactics": top.get("tactics", ""),
            "distance": top.get("distance"),
            "source": "famille XGBoost -> RAG ATT&CK",
        }
    return alerts

def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not models_available():
        print("XGBoost absent — lancer d'abord : python train_xgboost")
        raise SystemExit(1)

    payload = detect_hybrid()

    print("=" * 78)
    print("DÉTECTION HYBRIDE")
    print("=" * 78)
    print(f"  {payload['architecture']}")
    print(f"  split : {payload['evaluation']}")
    s = payload["summary"]
    print(f"\n  profils horaires : {s['hourly_profiles']:,}")
    print(f"  retenus (IF)     : {s['flagged']:,}")
    print(f"  priorité haute   : {s['priority_haute']:,}")
    print(f"  priorité normale : {s['priority_normale']:,}")
    print(f"  XGBoost seul (non alerté) : {s['xgb_solo_not_alerted']:,}")

    print(f"\n  {'ip':16s} {'heure':17s} {'score':>6s} {'priorité':10s} "
          f"{'accord':6s}  famille XGBoost")
    print("  " + "-" * 82)
    for a in payload["alerts"]:
        print(f"  {a['ip']:16s} {a['hour']:17s} {a['anomaly_score']:6.1f} "
              f"{a['priority']:10s} {str(a['agreement']):6s}  "
              f"{a['xgboost']['family']} "
              f"({a['xgboost']['p_attack_max']:.2f})")

if __name__ == "__main__":
    main()
