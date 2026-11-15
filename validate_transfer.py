"""Test de transfert — le XGBoost entraîné sur UNSW vaut-il sur nos logs ?

Entraîner sur UNSW et scorer UNSW prouve seulement que le modèle a appris
UNSW. La question utile : sur des journaux qu'il n'a jamais vus et dont la
distribution diffère (sbytes x236, ct_dst_ltm x7,5), remonte-t-il nos 3
attaques firewall tout en restant silencieux sur le normal ?

Trois écueils mesurés ici :
  1. décalage de distribution — le seuil 0.5 devient inadapté (score médian
     du normal >> seuil).
  2. taux de faux positifs — un FPR de 4,5 % sur UNSW peut exploser chez
     nous ; à base réelle, la précision s'effondre.
  3. classement — les attaques doivent remonter en tête, sinon le modèle
     est inutilisable pour prioriser un rapport SOC.

Les 3 attaques firewall sont repérées par une IP qui n'existe jamais en
normal (attaquant ou destination exfiltre = RFC 5737).

Usage :
    python validate_transfer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.flow_features import flows_from_firewall
from agent.preprocess import Preprocessor

import xgboost as xgb

MODEL_PATH = Path("models/xgboost.json")
PRE_PATH = Path("models/preprocess.json")
METRICS_PATH = Path("models/metrics.json")
LOGS_DIR = Path("workspace/logs")
SEP = "=" * 78

# Signatures d'attaque : une IP suffit, elle est unique au scénario.
#   91.219.236.18  = scanner externe (jamais en normal)
#   91.234.83.77   = destination d'exfiltration (jamais en normal)
#   203.0.113.77   = RFC 5737, réservée au scénario low & slow
ATTACKS = {
    "A2_scan": ("src", "91.219.236.18"),
    "A3_exfil": ("dst", "91.234.83.77"),
    "A4_lowslow": ("dst", "203.0.113.77"),
}

def label_flows(flows: pd.DataFrame) -> np.ndarray:
    """1 = flux d'attaque, 0 = normal. Basé sur les IP de scénario."""
    is_att = np.zeros(len(flows), dtype=int)
    for _, (col, ip) in ATTACKS.items():
        is_att[(flows[col] == ip).to_numpy()] = 1
    return is_att

def main() -> int:
    print(SEP)
    print("TEST DE TRANSFERT — XGBoost (UNSW) -> nos journaux")
    print(SEP)

    for p in (MODEL_PATH, PRE_PATH):
        if not p.exists():
            print(f"  ABSENT : {p} — lancer d'abord : python train_xgboost.py")
            return 1

    # chargement
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    pre = Preprocessor.load(PRE_PATH)

    flows = flows_from_firewall(LOGS_DIR)
    y = label_flows(flows)
    print(f"\n1. {len(flows):,} flux, dont {int(y.sum()):,} d'attaque "
          f"({100 * y.mean():.2f} %)")

    # score
    X = pre.transform(flows)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    n_att, n_norm = int(y.sum()), int((y == 0).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    print(f"\n2. seuil 0.5 — matrice de confusion")
    print(f"     TN={tn:>6,}  FP={fp:>6,}   <- faux positifs sur {n_norm:,} normaux")
    print(f"     FN={fn:>6,}  TP={tp:>6,}")
    fpr = fp / max(n_norm, 1)
    recall = tp / max(n_att, 1)
    precision = tp / max(tp + fp, 1)
    print(f"     FPR réel       : {fpr:.4f}  ({100 * fpr:.2f} % des normaux signalés)")
    print(f"     rappel réel    : {recall:.4f}")
    print(f"     précision réelle: {precision:.4f}")

    # décalage de distribution
    print(f"\n3. décalage de distribution (médiane des scores)")
    med_n = float(np.median(proba[y == 0]))
    med_a = float(np.median(proba[y == 1]))
    print(f"     normal  : {med_n:.4f}")
    print(f"     attaque : {med_a:.4f}")
    if med_n > 0.5:
        print(f"     !! le normal MEDIAN dépasse 0.5 : le seuil UNSW "
              f"est inadapté à nos données")
    else:
        print(f"     le seuil 0.5 reste du bon côté pour le normal")

    # classement (priorisation SOC)
    order = np.argsort(-proba)
    ranks = np.empty(len(proba), dtype=int)
    ranks[order] = np.arange(1, len(proba) + 1)
    att_ranks = ranks[y == 1]
    print(f"\n4. classement des {n_att:,} flux d'attaque (1 = plus suspect)")
    print(f"     rang médian    : {int(np.median(att_ranks)):,} / {len(proba):,}")
    print(f"     rang minimum   : {int(att_ranks.min()):,}")
    for k in (50, 100, 500, 1000):
        hit = int((att_ranks <= k).sum())
        print(f"     top-{k:<5,} contient {hit:>4,}/{n_att:,} attaques "
              f"({100 * hit / n_att:.1f} %)")

    # détail par attaque
    print(f"\n5. rappel par attaque (seuil 0.5) + rang médian")
    per_attack = []
    for name, (col, ip) in ATTACKS.items():
        mask = (flows[col] == ip).to_numpy()
        if mask.sum() == 0:
            print(f"     {name:12s} ABSENT des logs")
            continue
        r05 = float((pred[mask] == 1).mean())
        rmed = float(np.median(ranks[mask]))
        # fraction des flux de cette attaque présents dans le top 10%
        top10 = int((ranks[mask] <= len(proba) * 0.10).sum())
        print(f"     {name:12s} n={int(mask.sum()):>4,}  rappel={r05:.4f}  "
              f"rang médian={int(rmed):>6,}  dans top10%={top10:>4,}/{int(mask.sum())}")
        per_attack.append({"attack": name, "n": int(mask.sum()),
                           "recall": r05, "median_rank": int(rmed),
                           "in_top10pct": top10})

    # comparaison avec le score UNSW
    unsup = {}
    if METRICS_PATH.exists():
        unsup = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print(f"\n6. comparaison du seuil 0.5")
    if unsup.get("test"):
        print(f"     FPR sur UNSW (test) : {1 - unsup['test']['recall']:.4f} "
              f"-> rappel normal {1 - unsup['test']['recall']:.4f}")
        print(f"     FPR sur NOS logs    : {fpr:.4f}  "
              f"({'pire' if fpr > 1 - unsup['test']['recall'] else 'meilleur'} "
              f"qu'UNSW)")
    print(f"     précision UNSW (55 % d'attaques) : "
          f"{unsup.get('test', {}).get('precision', float('nan')):.4f}")
    print(f"     précision NOS logs ({100 * y.mean():.1f} % d'attaques) : "
          f"{precision:.4f}")

    # synthèse
    ok_rank = bool((att_ranks <= len(proba) * 0.10).sum() > 0)
    ok_fpr = fpr <= 0.10
    print(SEP)
    print("VERDICT")
    print(SEP)
    print(f"  [{'OK ' if ok_fpr else 'KO '}] FPR maîtrisé (<= 10 %) : {fpr:.4f}")
    print(f"  [{'OK ' if ok_rank else 'KO '}] au moins 1 attaque dans le top 10 %")
    print(f"  [{'OK ' if precision > 0.5 else 'KO '}] précision > 0.5")
    transfer_ok = ok_fpr and ok_rank
    print(f"\n  TRANSFERT : {'FONCTIONNEL' if transfer_ok else 'DÉGRADÉ'}")

    Path("models").mkdir(exist_ok=True)
    Path("models/transfer.json").write_text(json.dumps({
        "fpr": fpr, "recall": recall, "precision": precision,
        "median_score_normal": med_n, "median_score_attack": med_a,
        "n_flows": len(flows), "n_attack_flows": n_att,
        "per_attack": per_attack,
        "unsw_test_precision": unsup.get("test", {}).get("precision"),
    }, indent=2), encoding="utf-8")

    return 0 if transfer_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
