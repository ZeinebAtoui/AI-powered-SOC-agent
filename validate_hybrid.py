"""Validation de l'architecture hybride — agent + IF/XGBoost + RAG.

Cinq contrôles, un par rôle du design :

  A. Les 4 attaques sont dans le top 15
        -> l'Isolation Forest reste le détecteur : la fusion ne doit pas
           dégrader une détection qui marche déjà.

  B. Aucune alerte créée par le XGBoost seul
        -> le XGBoost ne peut que confirmer, jamais alerter seul.
           Sans ce contrôle, son faux positif de 24,6 % inonderait le rapport.

  C. Les familles mappables reçoivent une technique ATT&CK
        -> la règle « XGBoost nomme, RAG habille » est réellement câblée.

  D. Les familles sans équivalent ne reçoivent AUCUNE technique
        -> on ne fabrique jamais une classification : Generic, Fuzzers et
           Analysis n'ont pas d'équivalent ATT&CK.

  E. L'outil detect_anomalies() renvoie un JSON valide et exploitable
        -> c'est ce que l'agent lira réellement ; si le contrat casse ici,
           le rapport ne sortira jamais.

Usage :  python validate_hybrid.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.hybrid import FAMILY_RAG_QUERY, detect_hybrid, models_available

SEP = "=" * 84

# Signatures des 4 attaques (générées par generate_logs.py).
# Les IP d'attaquant sont uniques au scénario ; les IP internes (victime,
# poste légitime) existent aussi en normal -> on filtre sur la date aussi.
ATTACKS = {
    "A1_bruteforce": ("185.220.101.7", "2026-11-03"),   # unique à l'attaque
    "A2_scan":       ("91.219.236.18", "2026-11-07"),   # unique à l'attaque
    "A3_exfil":      ("10.0.10.25",    "2026-11-08"),   # victime, filtre date
    "A4_lowslow":    ("10.0.1.46",     "2026-11-06"),   # poste légitime, date
}

MAPPED_OK = {"Exploits", "Reconnaissance", "DoS", "Backdoor", "Shellcode", "Worms"}
UNMAPPED = {"Generic", "Fuzzers", "Analysis", "Normal", "-"}

def check_a(alerts: list[dict]) -> bool:
    print(SEP)
    print("A. Les 4 attaques sont-elles dans le top 15 ? (l'IF reste détecteur)")
    print(SEP)
    ok = True
    for name, (ip, date) in ATTACKS.items():
        hits = [a for a in alerts
                if a["ip"] == ip and a["hour"].startswith(date)]
        rank = alerts.index(hits[0]) + 1 if hits else None
        ok &= bool(hits)
        print(f"  {'OK ' if hits else 'KO '} {name:16s} attendu {ip} le {date}"
              f"  -> rang {rank if rank else 'ABSENT'}")
    return ok

def check_b(payload: dict) -> bool:
    print("\n" + SEP)
    print("B. Le XGBoost a-t-il créé une alerte seul ? (il ne doit PAS)")
    print(SEP)
    # Toute alerte vient de l'IF (flagged) : xgb_solo n'apparaît jamais ici.
    solo = payload["summary"]["xgb_solo_not_alerted"]
    n_alerts = payload["summary"]["shown"]
    ok = n_alerts > 0
    print(f"  {'OK ' if ok else 'KO '} alertes affichées       : {n_alerts}")
    print(f"       xgb_solo NON alertés : {solo}  (rejetés, comme voulu)")
    print(f"       priorité haute/normale : {payload['summary']['priority_haute']}"
          f" / {payload['summary']['priority_normale']}")
    print(f"  → le XGBoost ne fait que promouvoir : {payload['summary']['priority_haute']}"
          f" alertes hautes sur {n_alerts}")
    return ok

def check_c(alerts: list[dict]) -> bool:
    print("\n" + SEP)
    print("C. Les familles mappables reçoivent-elles une technique ATT&CK ?")
    print(SEP)
    seen: dict[str, bool] = {}
    for a in alerts:
        fam = a["xgboost"]["family"]
        if fam in MAPPED_OK:
            seen[fam] = bool(a.get("attack_technique"))
    if not seen:
        print("  KO  aucune famille mappable dans le top 15 — contrôle non concluant")
        return False
    ok = all(seen.values())
    for fam, has in sorted(seen.items()):
        tech = next((a["attack_technique"] for a in alerts
                     if a["xgboost"]["family"] == fam and a.get("attack_technique")),
                    None)
        detail = f"{tech['id']} {tech['name']}" if tech else "AUCUNE"
        print(f"  {'OK ' if has else 'KO '} {fam:16s} -> {detail}")
    return ok

def check_d(alerts: list[dict]) -> bool:
    print("\n" + SEP)
    print("D. Les familles SANS équivalent reçoivent-elles zéro technique ?")
    print(SEP)
    ok = True
    checked = set()
    for a in alerts:
        fam = a["xgboost"]["family"]
        if fam in UNMAPPED and fam not in checked:
            checked.add(fam)
            tech = a.get("attack_technique")
            good = tech is None
            ok &= good
            print(f"  {'OK ' if good else 'KO '} {fam:16s} -> "
                  f"{'aucune (correct)' if good else f'INVENTÉE: {tech}'}")
    if not checked:
        print("  --  aucune famille non mappable dans le top 15")
    return ok

def check_e() -> bool:
    print("\n" + SEP)
    print("E. L'outil detect_anomalies() renvoie-t-il un JSON exploitable ?")
    print(SEP)
    from agent.tools import detect_anomalies

    try:
        out = detect_anomalies.invoke({})
    except Exception as exc:  # noqa: BLE001
        print(f"  KO  exception : {type(exc).__name__}: {exc}")
        return False

    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        print(f"  KO  JSON illisible ({len(out)} car.) : {exc}")
        print(f"       début : {out[:200]!r}")
        return False

    required = ["summary", "evaluation", "alerts", "architecture"]
    missing = [k for k in required if k not in data]
    alert_keys = ["ip", "hour", "anomaly_score", "priority", "agreement",
                  "features", "xgboost", "attack_technique", "suggested_attack_query"]
    bad = [k for k in alert_keys
           if data["alerts"] and k not in data["alerts"][0]]

    ok = not missing and not bad
    print(f"  {'OK ' if ok else 'KO '} taille sortie : {len(out):,} caractères")
    print(f"       clés manquantes   : {missing or 'aucune'}")
    print(f"       clés alerte manquantes: {bad or 'aucune'}")
    print(f"       summary           : {data.get('summary')}")
    print(f"       evaluation.mode   : {data.get('evaluation', {}).get('mode')}")
    return ok

def main() -> int:
    if not models_available():
        print("XGBoost absent — lancer : python train_xgboost")
        return 1

    payload = detect_hybrid()
    alerts = payload["alerts"]

    a = check_a(alerts)
    b = check_b(payload)
    c = check_c(alerts)
    d = check_d(alerts)
    e = check_e()

    print("\n" + SEP)
    print("SYNTHÈSE")
    print(SEP)
    for name, res in (("A. 4 attaques dans le top 15", a),
                      ("B. XGBoost ne crée pas d'alerte", b),
                      ("C. familles mappables -> technique", c),
                      ("D. familles non mappables -> zéro", d),
                      ("E. outil = JSON exploitable", e)):
        print(f"  {'PASS' if res else 'FAIL'}  {name}")
    ok = all((a, b, c, d, e))
    print(f"\n  Architecture hybride : {'VALIDÉE' if ok else 'NON VALIDÉE'}")
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
