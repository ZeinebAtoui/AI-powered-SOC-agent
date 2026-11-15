"""Validation Phase 2 — 4 attaques, règles à seuil vs Isolation Forest.

Vérifie les 3 points de la Phase 2 :
  A. l'attaque « low & slow » reste dans le range normal de CHAQUE feature
  B. les règles à seuil classiques la manquent
  C. l'Isolation Forest la classe dans le top 15 avec les 3 autres attaques

Usage :  .\\.venv\\Scripts\\python.exe validate_detection.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.detect import fit_and_score, format_report
from agent.features import FEATURE_COLS, build_features

# IP et journée de l'attaque 4 (doit correspondre à generate_logs.py)
TARGET_IP, TARGET_DAY = "10.0.1.46", "2026-11-06"
SEP = "=" * 90

def main() -> None:
    logs = Path("workspace/logs")
    X = build_features(logs)

    atq4_all = np.asarray(
        (X.index.get_level_values("ip") == TARGET_IP)
        & (X.index.get_level_values("hour").str.startswith(TARGET_DAY))
    )
    # Heures de l'attaque 4 qui portent réellement du trafic
    active4 = atq4_all & (X["fw_src_conn"].to_numpy() > 0)
    known = np.asarray(
        (X["auth_failed"] >= 100) | (X["fw_src_conn"] >= 100)
        | (X["fw_dst_conn"] >= 100) | (X["fw_src_bytes"] >= 1e8)
    )
    normal = ~known & ~atq4_all
    Xn, sub = X[normal], X[active4]

    # A. Plausibilité individuelle des features
    print(SEP)
    print("A. ATTAQUE 4 — chaque feature reste-t-elle dans le range normal ?")
    print(SEP)
    print(f"{'feature':22s} {'max normal':>11} {'max atq4':>10}  {'ok ?':>6}")
    print("-" * 90)
    ok_all = True
    for c in FEATURE_COLS:
        nz = Xn[Xn[c] > 0][c]
        nmax = float(nz.max()) if len(nz) else 0.0
        amax = float(sub[c].max())
        ok = amax <= nmax or amax == 0
        ok_all &= ok
        print(f"{c:22s} {nmax:11.0f} {amax:10.0f}  {'OUI' if ok else 'NON !':>6}")
    print(f"\n>> Features BRUTES toutes plausibles : "
          f"{'OUI' if all(X[c][active4].max() <= (Xn[Xn[c] > 0][c].max() if len(Xn[Xn[c] > 0]) else 0) or X[c][active4].max() == 0 for c in FEATURE_COLS[:11]) else 'NON'}")
    print("   (fw_conn_per_dst dépasse légèrement : c'est la feature DÉRIVÉE")
    print("    conçue pour exposer la jointure 4 conn → 1 destination.)")

    # B. Règles à seuil
    print("\n" + SEP)
    print("B. RÈGLES À SEUIL — détectent-elles l'attaque 4 ?")
    print(SEP)
    rules = [
        ("auth_failed > 100", X["auth_failed"] > 100),
        ("fw_src_conn > 50", X["fw_src_conn"] > 50),
        ("fw_src_unique_dports > 10", X["fw_src_unique_dports"] > 10),
        ("fw_src_bytes > 100 Mo", X["fw_src_bytes"] > 100_000_000),
        ("deny_ratio==1 et conn>3",
         (X["fw_src_deny_ratio"] == 1) & (X["fw_src_conn"] > 3)),
        ("bytes_per_conn > 1 Mo", X["fw_bytes_per_conn"] > 1_000_000),
        # règle sur la feature DÉRIVÉe — transparence sur ce qu'elle révèle
        ("[dérivée] conn_per_dst > 3", X["fw_conn_per_dst"] > 3),
    ]
    missed = 0
    for name, mask in rules:
        m = np.asarray(mask)
        hit = int((m & active4).sum())
        missed += hit == 0
        print(f"  {name:32s} -> {int(m.sum()):>5} alertes | atq4 : "
              f"{'MANQUÉE' if hit == 0 else f'DÉTECTÉE ({hit} h)'}")
    n_raw = sum(1 for n, _ in rules if not n.startswith("["))
    print(f"\n>> {missed}/{len(rules)} règles manquent l'attaque 4 "
          f"(dont {n_raw} des règles sur features brutes)")

    # C. Classement Isolation Forest (holdout)
    print("\n" + SEP)
    print("C. ISOLATION FOREST — évaluation HOLDOUT (split temporel)")
    print(SEP)

    def signatures(tbl):
        """Les 4 signatures d'attaque, évaluées SUR LE TABLEAU FOURNI.

        Restreindre au tableau est essentiel : compter une signature sur
        toutes les lignes alors qu'on ne classe que le jeu de test fausserait
        le dénominateur (« lignes ») et donc le verdict.
        """
        ip = tbl.index.get_level_values("ip")
        hr = tbl.index.get_level_values("hour")
        return {
            "1. bruteforce 487 échecs": (tbl["auth_failed"] >= 100).to_numpy(),
            "2. scan 269 ports": (
                (tbl["fw_src_conn"] >= 100) | (tbl["fw_dst_conn"] >= 100)
            ).to_numpy(),
            "3. exfiltration 4,2 Go": (tbl["fw_src_bytes"] >= 1e8).to_numpy(),
            "4. LOW & SLOW (6 h)": (
                (ip == TARGET_IP) & hr.str.startswith(TARGET_DAY)
                & (tbl["fw_src_conn"].to_numpy() > 0)
            ),
        }

    def ranks_of(tbl):
        """(rang, flaggées, lignes) par signature — rangs DANS ce tableau."""
        rank = tbl["anomaly_score"].rank(ascending=False, method="min")
        out = {}
        for label, m in signatures(tbl).items():
            n = int(m.sum())
            if n == 0:
                out[label] = (None, 0, 0)
                continue
            out[label] = (int(rank.to_numpy()[m].min()),
                          int(tbl["anomaly"].to_numpy()[m].sum()), n)
        return out

    table = fit_and_score(logs)                      # holdout par défaut
    split = table.attrs.get("split", {})
    print(f"  mode          : {split.get('mode')}")
    print(f"  frontière     : {split.get('split_date')} "
          f"(la veille de la 1re attaque)")
    print(f"  ENTRAÎNEMENT  : {split.get('train_period')}")
    print(f"                  {split.get('train_rows')} profils — 0 attaque")
    print(f"  ÉVALUATION    : {split.get('test_period')}")
    print(f"                  {split.get('test_rows')} profils — les 4 attaques")
    print("\n  Le modèle n'a JAMAIS vu le jeu d'évaluation, et le seuil")
    print("  `contamination` est calibré sur l'entraînement seul.")

    holdout = ranks_of(table)
    print(f"\n{len(table)} profils évalués | "
          f"{int(table['anomaly'].sum())} flaggés (top 2%)\n")
    print(f"{'signature':26s} {'lignes':>7} {'meilleur rang':>14} "
          f"{'flaggées':>10}  verdict")
    print("-" * 90)
    for label, (r, f, n) in holdout.items():
        verdict = "ABSENT" if r is None else ("OK" if f == n else f"{f}/{n}")
        print(f"{label:26s} {n:>7} {'-' if r is None else r:>14} "
              f"{f:>10}  {verdict}")

    # C'. Comparaison honnête : ce que donnait le protocole sans split
    print("\n" + SEP)
    print("C'. CONTRÔLE — même modèle SANS split (resubstitution)")
    print(SEP)
    print("  fit et score sur les MÊMES lignes : le seuil est calibré sur le")
    print("  jeu évalué. Ce chiffre n'est PAS une mesure de détection — on le")
    print("  publie pour quantifier l'écart, pas pour valider le modèle.\n")
    weak = fit_and_score(logs, split_date=None)
    print(f"{'signature':26s} {'lignes':>7} {'meilleur rang':>14}  comparaison")
    print("-" * 90)
    weak_ranks = ranks_of(weak)
    for label in holdout:
        h_r, _, _ = holdout[label]
        w_r, _, _ = weak_ranks[label]
        delta = ("" if h_r is None or w_r is None
                 else f"holdout {h_r} vs sans-split {w_r}")
        print(f"{label:26s} {'-' if w_r is None else w_r:>14}  {delta}")

    # D. Top 15 = sortie réelle de l'outil
    print("\n" + SEP)
    print("D. TOP 15 — exactement ce que renvoie l'outil detect_anomalies()")
    print(SEP)
    report = json.loads(format_report(table))
    tags = {
        "185.220.101.7": "ATTAQUE 1",
        "91.219.236.18": "ATTAQUE 2",
    }
    for i, e in enumerate(report["anomalies"], 1):
        tag = ""
        if e["ip"] == TARGET_IP and e["hour"].startswith(TARGET_DAY):
            tag = "ATTAQUE 4"
        elif e["ip"] in tags:
            tag = tags[e["ip"]]
        elif e["ip"] == "10.0.10.25" and e["hour"].startswith("2026-11-07"):
            tag = "ATTAQUE 2"
        elif e["ip"] == "10.0.10.25" and e["hour"].startswith("2026-11-08"):
            tag = "ATTAQUE 3"
        suffix = f"  <== {tag}" if tag else ""
        print(f"{i:2}. {e['ip']:>16} {e['hour']}  "
              f"score={e['anomaly_score']:5.1f}{suffix}")

    present = {
        1: any(e["ip"] == "185.220.101.7" for e in report["anomalies"]),
        2: any(e["ip"] == "91.219.236.18" for e in report["anomalies"]),
        3: any(e["ip"] == "10.0.10.25" and e["hour"].startswith("2026-11-08")
               for e in report["anomalies"]),
        4: any(e["ip"] == TARGET_IP and e["hour"].startswith(TARGET_DAY)
               for e in report["anomalies"]),
    }
    print("\n>> Présence des 4 attaques dans le top 15 : "
          + ", ".join(f"A{k}={'OUI' if v else 'NON'}" for k, v in present.items()))
    if not all(present.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
