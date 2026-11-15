"""Features de flux communes entre nos journaux et UNSW-NB15.

Le pipeline hybride combine deux granularités :

    Isolation Forest  ->  profils (IP, heure)   [agent/features.py]
    XGBoost           ->  FLUX individuels      [ce module]

Le XGBoost est entraîné sur UNSW-NB15 puis appliqué à nos journaux : ce
transfert n'est honnête que si les colonnes ont la même sémantique des deux
côtés.

Un pont partiel, pas les 45 colonnes d'UNSW : ~20 features exigent une
capture paquets (dur, spkts, sjit, tcprtt, swin...) que notre pare-feu ne
produit pas — les inventer ou les combler de zéros fausserait l'entraînement.
On ne garde que ce qui se calcule à l'identique des deux sources, soit
11 features, dont les ct_* qui détectent scan, beaconing et latéral.

Usage :
    python -m agent.flow_features       # diagnostique d'alignement
"""

from __future__ import annotations

import re
from collections import Counter, deque
from pathlib import Path

import pandas as pd

# Colonnes présentes des deux côtés, avec la même sémantique.
COMMON_FEATURES = [
    "proto",
    "service",
    "sbytes",
    "ct_srv_src",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]

CATEGORICALS = ["proto", "service"]
NUMERICS = [c for c in COMMON_FEATURES if c not in CATEGORICALS]

# UNSW définit ses ct_* comme « nombre de connexions parmi les 100
# dernières » : on reproduit exactement cette fenêtre.
WINDOW = 100

# Mappage port destination -> service dans le vocabulaire UNSW. Les ports
# absents (dont ceux générés par le scan) tombent sur "-" — même réflexe
# qu'UNSW, où "-" représente 57 % des flux.
DPT_TO_SERVICE = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 25: "smtp", 53: "dns",
    67: "dhcp", 68: "dhcp", 80: "http", 110: "pop3", 161: "snmp",
    443: "ssl", 6667: "irc", 1812: "radius", 8080: "http", 8443: "ssl",
}

# SPT est présent dans le log mais absent du regex de features.py (inutile
# là-bas) — indispensable ici pour ct_dst_sport_ltm.
FW_FLOW_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\S*\s+\S+\s+"
    r"(?P<verdict>ACCEPT|DENY)\s+.*?"
    r"SRC=(?P<src>\S+)\s+DST=(?P<dst>\S+)\s+.*?"
    r"PROTO=(?P<proto>\S+)\s+SPT=(?P<spt>\d+)\s+DPT=(?P<dpt>\d+)\s+"
    r"BYTES=(?P<bytes>\d+)"
)

def parse_firewall_flows(logs_dir: Path | str) -> pd.DataFrame:
    """Extrait les flux individuels de firewall.log (une ligne = un flux).

    Returns:
        DataFrame trié chronologiquement, colonnes de travail :
        ts, src, dst, spt, dpt, proto, service, bytes, verdict.
    """
    logs_dir = Path(logs_dir)
    fw_path = logs_dir / "firewall.log"
    if not fw_path.exists():
        raise FileNotFoundError(f"firewall.log introuvable : {fw_path}")

    rows: list[dict] = []
    for line in fw_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = FW_FLOW_RE.match(line)
        if not m:
            continue
        d = m.groupdict()
        dpt = int(d["dpt"])
        rows.append({
            "ts": d["ts"],
            "src": d["src"],
            "dst": d["dst"],
            "spt": int(d["spt"]),
            "dpt": dpt,
            # UNSW écrit en minuscules : on aligne pour que les vocabulaires
            # coïncident (TCP -> tcp).
            "proto": d["proto"].lower(),
            "service": DPT_TO_SERVICE.get(dpt, "-"),
            "sbytes": int(d["bytes"]),
            "verdict": d["verdict"],
        })

    if not rows:
        raise ValueError(f"Aucun flux parsé dans {fw_path} — regex obsolète ?")

    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

def add_rolling_counts(flows: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Ajoute les 7 colonnes ct_* — comptages sur les `window` derniers flux.

    Sémantique reprise de la documentation UNSW-NB15 :

        ct_src_ltm        : même IP source
        ct_dst_ltm        : même IP destination
        ct_srv_src        : même (source, service)
        ct_srv_dst        : même (destination, service)
        ct_src_dport_ltm  : même (source, port destination)
        ct_dst_sport_ltm  : même (destination, port source)
        ct_dst_src_ltm    : même (source, destination)

    La fenêtre inclut le flux courant (comme dans UNSW : le premier flux
    d'une source affiche 1, pas 0).
    """
    flows = flows.sort_values("ts").reset_index(drop=True)
    n = len(flows)

    seqs: dict[str, list] = {
        "ct_src_ltm": list(zip(flows["src"])),
        "ct_dst_ltm": list(zip(flows["dst"])),
        "ct_srv_src": list(zip(flows["src"], flows["service"])),
        "ct_srv_dst": list(zip(flows["dst"], flows["service"])),
        "ct_src_dport_ltm": list(zip(flows["src"], flows["dpt"])),
        "ct_dst_sport_ltm": list(zip(flows["dst"], flows["spt"])),
        "ct_dst_src_ltm": list(zip(flows["src"], flows["dst"])),
    }

    for name, seq in seqs.items():
        counts: Counter = Counter()
        window_q: deque = deque()
        col = [0] * n
        for i, key in enumerate(seq):
            window_q.append(key)
            counts[key] += 1
            if len(window_q) > window:
                old = window_q.popleft()
                counts[old] -= 1
                if counts[old] == 0:
                    del counts[old]
            col[i] = counts[key]
        flows[name] = col

    # is_sm_ips_ports : src==dst ET spt==dpt (définition UNSW).
    flows["is_sm_ips_ports"] = (
        (flows["src"] == flows["dst"]) & (flows["spt"] == flows["dpt"])
    ).astype(int)

    return flows

def flows_from_firewall(logs_dir: Path | str) -> pd.DataFrame:
    """Pipeline complet : firewall.log -> flux + features de flux communes."""
    return add_rolling_counts(parse_firewall_flows(logs_dir))

def load_unsw(path: Path | str) -> pd.DataFrame:
    """Charge UNSW-NB15 et ne garde que le schéma commun (+ libellés)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset UNSW introuvable : {path}")

    df = pd.read_csv(path)
    df["proto"] = df["proto"].astype(str).str.lower()
    df["service"] = df["service"].astype(str).str.lower()

    missing = [c for c in COMMON_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes absentes d'UNSW : {missing}")

    keep = COMMON_FEATURES + ["label", "attack_cat"]
    return df[keep].copy()

def _compare(ours: pd.DataFrame, unsw: pd.DataFrame) -> None:
    """Affiche côte à côte la distribution de chaque feature commune.

    Test honnête du pont : si les ordres de grandeur divergent fortement, le
    XGBoost entraîné sur UNSW ne se transférera pas à nos journaux.
    """
    print(f"{'feature':22s} {'nos flux (min/med/max)':>32s}   "
          f"{'UNSW (min/med/max)':>32s}")
    print("-" * 92)
    for col in NUMERICS:
        a, b = ours[col], unsw[col]
        fmt = lambda s: f"{s.min():.0f} / {s.median():.0f} / {s.max():.0f}"
        print(f"{col:22s} {fmt(a):>32s}   {fmt(b):>32s}")

    print()
    print(f"{'feature':22s} {'nos flux (top valeurs)':>32s}   "
          f"{'UNSW (top valeurs)':>32s}")
    print("-" * 92)
    for col in CATEGORICALS:
        ours_top = ", ".join(ours[col].value_counts().head(4).index)
        unsw_top = ", ".join(unsw[col].value_counts().head(4).index)
        print(f"{col:22s} {ours_top:>32s}   {unsw_top:>32s}")

def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = Path(__file__).resolve().parent.parent
    unsw_path = Path(r"D:\datasets\unsw_nb15\UNSW_NB15_testing-set.csv")

    ours = flows_from_firewall(root / "workspace" / "logs")
    unsw = load_unsw(unsw_path)

    print("=" * 92)
    print("PONT DE FEATURES — nos flux vs UNSW-NB15")
    print("=" * 92)
    print(f"  nos flux      : {ours.shape[0]:>7} lignes x {ours.shape[1]} colonnes")
    print(f"  UNSW          : {unsw.shape[0]:>7} lignes x {unsw.shape[1]} colonnes")
    print(f"  features communes ({len(COMMON_FEATURES)}) : {COMMON_FEATURES}")
    print()

    _compare(ours, unsw)

    # Min attendu = 1 pour les ct_* (la fenêtre inclut le flux courant) :
    # 0 indiquerait une fenêtre décalée d'une unité face à UNSW.
    mins_unsw = unsw[NUMERICS].min().to_dict()
    mins_ours = ours[NUMERICS].min().to_dict()
    print()
    print("Contrôle de sens (min attendu = 1 pour les ct_*, 0 sinon) :")
    for col in NUMERICS:
        print(f"  {col:22s} nos={mins_ours[col]:>8.0f}   UNSW={mins_unsw[col]:>8.0f}")

if __name__ == "__main__":
    main()
