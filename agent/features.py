"""Extraction de features numériques à partir des logs SOC.

Isolation Forest ne comprend pas du texte : il faut transformer les journaux
bruts en une matrice numérique. On agrège par couple (IP, heure) — c'est la
granularité à laquelle les comportements anormaux se voient.

Chaque feature est conçue pour qu'une attaque évidente ressorte sur une seule
dimension, tandis qu'une attaque « low & slow » ne ressort que sur la
combinaison de plusieurs dimensions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Nov 03 03:12:07 srv-auth-01 sshd[24113]: Failed password for ... from 1.2.3.4 port 51234 ssh2
AUTH_RE = re.compile(
    r"^(\w{3}) (\d{2}) (\d{2}:\d{2}:\d{2}) .*?: "
    r"(Failed password|Accepted \w+) .*?from (\d+\.\d+\.\d+\.\d+) "
)

# 2026-11-07T04:41:10+01:00 fw-edge-01 DENY IN=eth0 SRC=a DST=b PROTO=TCP SPT=x DPT=y BYTES=z
FW_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T(\d{2}):\d{2}:\d{2}.*? (ACCEPT|DENY) .*?"
    r"SRC=(\S+) DST=(\S+) .*?DPT=(\d+) BYTES=(\d+)"
)

# 2026-11-05T14:23:11+01:00 app-server-01 INFO req=... user=x ip=1.2.3.4 action=login ...
APP_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}):\d{2}:\d{2}")
APP_IP_RE = re.compile(r"\bip=(\d+\.\d+\.\d+\.\d+)")
APP_ACTION_RE = re.compile(r"\baction=(\S+)")
APP_ROWS_RE = re.compile(r"\brows=(\d+)")

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

FEATURE_COLS = [
    "fw_src_conn", "fw_src_bytes", "fw_src_deny_ratio",
    "fw_src_unique_dports", "fw_src_unique_dsts",
    "fw_dst_conn", "fw_dst_unique_srcs",
    "auth_failed", "auth_accepted",
    "app_actions", "app_export_rows",
    # Features relationnelles : sans elles, une exfiltration « low & slow »
    # (4 conn → 1 hôte → 1 port) reste dans le range normal de chaque
    # feature isolée et passe inaperçue.
    "fw_conn_per_dst", "fw_conn_per_dport", "fw_bytes_per_conn",
]

def _add_relational(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features qui capturent des jointures, pas des valeurs seules.

    fw_src_unique_dsts=1 est banal et fw_src_conn=4 est rare mais plausible :
    c'est leur rapport qui est inhabituel. 4 connexions toutes vers le même
    hôte, aucune règle sur une seule colonne ne la voit.
    """
    conn = df["fw_src_conn"]
    safe = conn.clip(lower=1)

    df["fw_conn_per_dst"] = (conn / df["fw_src_unique_dsts"].clip(lower=1)).where(conn > 0, 0.0)
    df["fw_conn_per_dport"] = (conn / df["fw_src_unique_dports"].clip(lower=1)).where(conn > 0, 0.0)
    df["fw_bytes_per_conn"] = (df["fw_src_bytes"] / safe).where(conn > 0, 0.0)
    return df

def _hour_key(year: int, month: int, day: int, hour: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:00:00"

def build_features(logs_dir: Path | str) -> pd.DataFrame:
    """Transforme les 3 logs en une table (IP, heure) × features numériques.

    Returns:
        DataFrame indexé par ["ip", "hour"], colonnes = FEATURE_COLS.
    """
    logs_dir = Path(logs_dir)
    rows: dict[tuple[str, str], dict[str, float]] = {}

    def cell(ip: str, hour: str) -> dict:
        key = (ip, hour)
        if key not in rows:
            # _dports/_dsts/_srcs sont des sets intermédiaires (cardinalité
            # calculée plus bas) — initialisés pour éviter des NaN.
            rows[key] = {c: 0.0 for c in FEATURE_COLS}
            rows[key].update({"_dports": set(), "_dsts": set(), "_srcs": set()})
        return rows[key]

    # auth.log : échecs / succès par IP source
    auth_path = logs_dir / "auth.log"
    if auth_path.exists():
        for line in auth_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = AUTH_RE.match(line)
            if not m:
                continue
            mon, day, hms, action, ip = m.groups()
            month = MONTHS.get(mon)
            if month is None:
                continue
            hour = hms[:2]
            c = cell(ip, _hour_key(2026, month, int(day), int(hour)))
            if action.startswith("Failed"):
                c["auth_failed"] += 1
            else:
                c["auth_accepted"] += 1

    # firewall.log : connexions, volume, diversité (source ET destination)
    fw_path = logs_dir / "firewall.log"
    if fw_path.exists():
        for line in fw_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = FW_RE.match(line)
            if not m:
                continue
            date, hour_s, verdict, src, dst, dpt, nbytes = m.groups()
            y, mo, d = (int(x) for x in date.split("-"))
            hour = _hour_key(y, mo, d, int(hour_s))

            csrc = cell(src, hour)
            csrc["fw_src_conn"] += 1
            csrc["fw_src_bytes"] += int(nbytes)
            if verdict == "DENY":
                csrc["fw_src_deny_ratio"] += 1
            csrc.setdefault("_dports", set()).add(dpt)
            csrc.setdefault("_dsts", set()).add(dst)

            cdst = cell(dst, hour)
            cdst["fw_dst_conn"] += 1
            cdst.setdefault("_srcs", set()).add(src)

    # app.log : actions et volume d'export par IP client
    app_path = logs_dir / "app.log"
    if app_path.exists():
        for line in app_path.read_text(encoding="utf-8", errors="replace").splitlines():
            ts = APP_TS_RE.match(line)
            ip_m = APP_IP_RE.search(line)
            act_m = APP_ACTION_RE.search(line)
            if not (ts and ip_m and act_m):
                continue
            date, hour_s = ts.groups()
            y, mo, d = (int(x) for x in date.split("-"))
            c = cell(ip_m.group(1), _hour_key(y, mo, d, int(hour_s)))
            c["app_actions"] += 1
            rows_m = APP_ROWS_RE.search(line)
            if rows_m and act_m.group(1).startswith("export"):
                c["app_export_rows"] += int(rows_m.group(1))

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["ip", "hour"])

    # Set -> cardinalité, compteur deny -> proportion
    for col, src_set in (("_dports", "fw_src_unique_dports"),
                         ("_dsts", "fw_src_unique_dsts"),
                         ("_srcs", "fw_dst_unique_srcs")):
        if col in df.columns:
            df[src_set] = df[col].apply(len)
            df = df.drop(columns=[col])

    if "fw_src_deny_ratio" in df.columns and "fw_src_conn" in df.columns:
        df["fw_src_deny_ratio"] = (
            df["fw_src_deny_ratio"] / df["fw_src_conn"].clip(lower=1)
        )

    df = df.fillna(0.0)
    df = _add_relational(df)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df[FEATURE_COLS].sort_index()

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("workspace/logs")
    table = build_features(logs)
    print(f"Table : {table.shape[0]} lignes (IP, heure) × {table.shape[1]} features\n")
    print(table.describe().T[["count", "mean", "min", "max"]].round(2).to_string())
