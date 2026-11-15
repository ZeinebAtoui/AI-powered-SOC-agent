#!/usr/bin/env python3
"""Phase 1 — Générateur de logs synthétiques réalistes (SOC Agent).

Produit 3 journaux dans workspace/logs/ (~1,5 à 2 Mo chacun, soit volontairement
au-dessus de la limite de 1 Mo de read_file → l'agent est forcé d'utiliser
run_python pour agréger, comme un vrai analyste face à des logs volumineux).

Scénario d'attaque injecté (corrélation inter-fichiers à découvrir) :
  1. 03/11 03:12→03:19 — bruteforce SSH depuis 185.220.101.7
                         → 487 échecs PUIS succès sur le compte "jdupont"
                         → puis connexions applicatives depuis cette même IP
  2. 07/11 22:14       — scan de ports depuis 91.219.236.18 vers 10.0.10.25
  3. 08/11 02:47→08:15 — exfiltration ~4,2 Go depuis 10.0.10.25 vers
                         91.234.83.77  + exports massifs dans app.log
  4. 06/11 09:00→14:00 — exfiltration "low & slow" depuis 10.0.1.46 vers
                         203.0.113.77 (nouvelle IP jamais vue)

L'attaque 4 reste dans le range observé en normal (≤ 5 conn/heure, ≤ 797 Ko,
≤ 3 ports, ≤ 4 destinations) : aucune règle à seuil ne la voit. C'est la
combinaison qui est rare — 4 connexions toutes vers le même hôte sur le même
port, alors qu'en normal un hôte qui parle 4 fois dialogue avec 2 à 4
destinations. C'est cette sparsity que l'isole l'IF.

Données 100 % synthétiques et reproductibles (seed=42) — aucune donnée réelle.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Console Windows : forcer UTF-8 pour les accents
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

random.seed(42)

OUT_DIR = Path(__file__).resolve().parent / "workspace" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY0 = datetime(2026, 10, 10, 0, 0, 0)
NDAYS = 30

# Éléments réalistes
USERS = [
    "jdupont", "mlegrand", "kbenali", "tmoreau", "snguyen",
    "apetit", "rdurand", "croux", "lbernard", "fmoreau",
]
INTERNAL_IPS = [f"10.0.1.{i}" for i in range(10, 60)] + [f"10.0.2.{i}" for i in range(10, 40)]
EXT_BENIGN = [
    "104.16.132.229", "151.101.1.140", "142.250.172.14",
    "13.107.42.14", "23.55.60.12", "195.181.160.10", "8.8.8.8",
]
AUTH_HOSTS = ["srv-auth-01", "srv-auth-02", "srv-bastion-01"]
FW_HOST = "fw-edge-01"
APP_HOST = "app-server-01"

# Acteurs de l'attaque
ATT_BRUTE_IP = "185.220.101.7"
ATT_SCAN_IP = "91.219.236.18"
ATT_EXFIL_IP = "91.234.83.77"
VICTIM_HOST = "10.0.10.25"
BREACHED_USER = "jdupont"

# Attaque 4 — low & slow (indétectable par règles à seuil)
ATT_SLOW_IP = "203.0.113.77"   # RFC 5737 — jamais présent dans EXT_BENIGN
ATT_SLOW_SRC = "10.0.1.46"     # poste interne légitime (déjà présent en normal)
ATT_SLOW_DAY = DAY0 + timedelta(days=27)   # 06/11/2026
ATT_SLOW_HOURS = range(9, 15)              # heures de bureau, 09:00→14:59
ATT_SLOW_PER_HOUR = 4                      # ≤ max normal (5)
ATT_SLOW_BYTES_HOUR = 350_000              # < p99 normal (406 K), < max (797 K)

def iso(t: datetime) -> str:
    """Timestamp ISO 8601 avec fuseau +01:00."""
    return t.strftime("%Y-%m-%dT%H:%M:%S+01:00")

def syslog(t: datetime) -> str:
    """Timestamp style syslog : 'Nov 03 03:12:07'."""
    return t.strftime("%b %d %H:%M:%S")

# 1. auth.log — authentifications SSH
def gen_auth() -> list[tuple[datetime, str]]:
    rows: list[tuple[datetime, str]] = []

    for d in range(NDAYS):
        day = DAY0 + timedelta(days=d)

        # Connexions légitimes (heures de bureau + un peu aléatoire)
        for _ in range(random.randint(170, 230)):
            t = day + timedelta(
                hours=random.randint(6, 20),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )
            user = random.choice(USERS)
            ip = random.choice(INTERNAL_IPS)
            host = random.choice(AUTH_HOSTS)
            pid = random.randint(20000, 39999)
            port = random.randint(32768, 65535)
            rows.append((t, f"{host} sshd[{pid}]: Accepted publickey for {user} from {ip} port {port} ssh2"))

            # Déconnexion
            t2 = min(t + timedelta(minutes=random.randint(20, 300)), day + timedelta(days=1))
            rows.append((t2, f"{host} systemd-logind[1321]: Session {random.randint(1000, 9999)} closed for user {user}"))

        # Bruit normal d'échecs (scanurs/robots isolés)
        for _ in range(random.randint(8, 30)):
            t = day + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
            host = random.choice(AUTH_HOSTS)
            pid = random.randint(20000, 39999)
            if random.random() < 0.5:
                msg = f"Failed password for invalid user {random.choice(['test', 'oracle', 'ubuntu', 'git', 'postgres'])}"
            else:
                msg = f"Failed password for {random.choice(USERS)}"
            rows.append((t, f"{host} sshd[{pid}]: {msg} from {random.choice(EXT_BENIGN)} port {random.randint(1024, 65535)} ssh2"))

        # Tâches cron / journal système (volume + réalisme)
        for _ in range(random.randint(25, 45)):
            t = day + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
            rows.append((t, f"{random.choice(AUTH_HOSTS)} CRON[{random.randint(1000, 9999)}]: (root) CMD (/usr/local/bin/backup.sh)"))

    # ATTAQUE 1 : bruteforce puis compromission du compte jdupont
    brute_day = DAY0 + timedelta(days=24)  # 03/11/2026
    start = brute_day + timedelta(hours=3, minutes=12, seconds=7)
    span = 397  # 03:12:07 → 03:18:44
    for i in range(487):
        t = start + timedelta(seconds=span * i / 487 + random.uniform(0, 0.4))
        pid = 24113 + i // 2
        if random.random() < 0.6:
            msg = f"Failed password for invalid user {random.choice(['admin', 'root', 'test', 'oracle', 'user'])}"
        else:
            msg = f"Failed password for {random.choice([BREACHED_USER, 'mlegrand', 'admin'])}"
        rows.append((t, f"srv-auth-01 sshd[{pid}]: {msg} from {ATT_BRUTE_IP} port {random.randint(51000, 51990)} ssh2"))

    # Succès : le compte jdupont est compromis
    ok = start + timedelta(seconds=span + 7)
    rows.append((ok, f"srv-auth-01 sshd[24891]: Accepted password for {BREACHED_USER} from {ATT_BRUTE_IP} port 51892 ssh2"))
    rows.append((ok + timedelta(seconds=1), "srv-auth-01 systemd-logind[1321]: New session 8841 of user jdupont."))
    rows.append((ok + timedelta(hours=1, minutes=22), "srv-auth-01 systemd-logind[1321]: Session 8841 closed for user jdupont."))

    rows.sort(key=lambda r: r[0])
    return rows

# 2. firewall.log — connexions pare-feu
def gen_firewall() -> list[tuple[datetime, str]]:
    rows: list[tuple[datetime, str]] = []

    for d in range(NDAYS):
        day = DAY0 + timedelta(days=d)
        for _ in range(random.randint(400, 520)):
            t = day + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
            src = random.choice(INTERNAL_IPS)
            dst = random.choice(EXT_BENIGN)
            proto = random.choice(["TCP", "TCP", "TCP", "UDP"])
            dpt = random.choice([443, 443, 443, 80, 53, 123, 8080])
            verdict = "ACCEPT" if random.random() < 0.9 else "DENY"
            rows.append((
                t,
                f"{FW_HOST} {verdict} IN=eth0 SRC={src} DST={dst} PROTO={proto} "
                f"SPT={random.randint(49152, 65535)} DPT={dpt} BYTES={random.randint(400, 250000)}",
            ))

    # ATTAQUE 2 : scan de ports vers l'hôte victime
    scan_day = DAY0 + timedelta(days=28)  # 07/11/2026
    scan_start = scan_day + timedelta(hours=22, minutes=14, seconds=3)
    ports = random.sample(range(1, 1024), 260) + [3389, 445, 22, 8080, 8443, 3306, 5432, 6379, 27017]
    for i, port in enumerate(ports):
        t = scan_start + timedelta(seconds=38 * i / len(ports))
        rows.append((
            t,
            f"{FW_HOST} DENY IN=eth0 SRC={ATT_SCAN_IP} DST={VICTIM_HOST} PROTO=TCP "
            f"SPT={random.randint(40000, 50000)} DPT={port} BYTES=60",
        ))

    # ATTAQUE 3 : exfiltration ~4,2 Go depuis l'hôte victime
    exfil_day = DAY0 + timedelta(days=29)  # 08/11/2026
    ex_start = exfil_day + timedelta(hours=2, minutes=47)
    ex_end = exfil_day + timedelta(hours=8, minutes=15)
    n = 400
    total_target = 4_200_000_000  # ~4,2 Go
    per_line = total_target // n
    for i in range(n):
        t = ex_start + (ex_end - ex_start) * i / n
        rows.append((
            t,
            f"{FW_HOST} ACCEPT IN=eth0 SRC={VICTIM_HOST} DST={ATT_EXFIL_IP} PROTO=TCP "
            f"SPT={random.randint(49152, 65535)} DPT=443 BYTES={per_line + random.randint(-400_000, 400_000)}",
        ))

    # ATTAQUE 4 : exfiltration low & slow. Chaque valeur reste dans le range
    # normal (4 conn/heure, 350 Ko/heure, 1 port, 1 destination) : ce qui est
    # rare, c'est la combinaison — 4 connexions vers UN seul hôte sur UN seul
    # port, alors qu'en normal 4 connexions touchent 2 à 4 destinations.
    per_line_slow = ATT_SLOW_BYTES_HOUR // ATT_SLOW_PER_HOUR
    for h in ATT_SLOW_HOURS:
        for k in range(ATT_SLOW_PER_HOUR):
            t = ATT_SLOW_DAY + timedelta(
                hours=h,
                minutes=7 * k + random.randint(0, 3),   # cadence régulière ~7 min
                seconds=random.randint(0, 59),
            )
            rows.append((
                t,
                f"{FW_HOST} ACCEPT IN=eth0 SRC={ATT_SLOW_SRC} DST={ATT_SLOW_IP} PROTO=TCP "
                f"SPT={random.randint(49152, 65535)} DPT=8443 "
                f"BYTES={per_line_slow + random.randint(-6_000, 6_000)}",
            ))

    rows.sort(key=lambda r: r[0])
    return rows

# 3. app.log — activité applicative
def gen_app() -> list[tuple[datetime, str]]:
    rows: list[tuple[datetime, str]] = []
    actions = ["login", "view_dashboard", "list_orders", "list_customers",
               "view_report", "export_csv", "logout", "search"]

    for d in range(NDAYS):
        day = DAY0 + timedelta(days=d)
        for _ in range(random.randint(360, 440)):
            t = day + timedelta(hours=random.randint(7, 19), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
            rows.append((
                t,
                f"{APP_HOST} INFO req={random.randint(10000, 99999)} user={random.choice(USERS)} "
                f"ip={random.choice(INTERNAL_IPS)} action={random.choice(actions)} status=200 "
                f"duration={random.randint(8, 900)}ms",
            ))

    # ATTAQUE 1 (suite) : le compte compromis s'active depuis l'IP attaquante
    brute_day = DAY0 + timedelta(days=24)
    t0 = brute_day + timedelta(hours=3, minutes=19, seconds=22)
    attack1_actions = [
        ("login", 200, 38),
        ("list_customers", 200, 1240),
        ("view_report", 200, 890),
        ("export_csv", 200, 8421),
        ("export_csv", 200, 9102),
        ("list_orders", 200, 2310),
        ("export_csv", 200, 10877),
        ("logout", 200, 12),
    ]
    for i, (act, status, dur) in enumerate(attack1_actions):
        rows.append((
            t0 + timedelta(minutes=4 * i + random.randint(0, 2)),
            f"{APP_HOST} WARN req={70000 + i} user={BREACHED_USER} ip={ATT_BRUTE_IP} "
            f"action={act} status={status} duration={dur}ms",
        ))

    # ATTAQUE 3 (suite) : exports massifs depuis l'hôte victime
    exfil_day = DAY0 + timedelta(days=29)
    t1 = exfil_day + timedelta(hours=2, minutes=52)
    for i in range(26):
        rows.append((
            t1 + timedelta(minutes=13 * i + random.randint(0, 4)),
            f"{APP_HOST} WARN req={80000 + i} user={BREACHED_USER} ip={VICTIM_HOST} "
            f"action=export_records status=200 rows={random.randint(12000, 26000)} "
            f"duration={random.randint(6000, 24000)}ms",
        ))

    rows.sort(key=lambda r: r[0])
    return rows

def write(name: str, rows: list[tuple[datetime, str]], prefix=None) -> Path:
    path = OUT_DIR / name
    lines = [prefix(t) + body for t, body in rows] if prefix else [body for _, body in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def main() -> None:
    print("Génération des logs synthétiques (seed=42)...\n")

    auth = gen_auth()
    p1 = write("auth.log", auth, prefix=lambda t: f"{syslog(t)} ")
    print(f"  {p1.name:14} {len(auth):>7} lignes  {p1.stat().st_size / 1e6:.2f} Mo")

    fw = gen_firewall()
    p2 = write("firewall.log", fw, prefix=lambda t: f"{iso(t)} ")
    print(f"  {p2.name:14} {len(fw):>7} lignes  {p2.stat().st_size / 1e6:.2f} Mo")

    app = gen_app()
    p3 = write("app.log", app, prefix=lambda t: f"{iso(t)} ")
    print(f"  {p3.name:14} {len(app):>7} lignes  {p3.stat().st_size / 1e6:.2f} Mo")

    print("\nScénario injecté à retrouver :")
    print("  1. 03/11 03:12  bruteforce SSH 185.220.101.7 → compte jdupont compromis")
    print("  2. 07/11 22:14  scan de ports 91.219.236.18 → 10.0.10.25")
    print("  3. 08/11 02:47  exfiltration ~4,2 Go 10.0.10.25 → 91.234.83.77")
    print("  4. 06/11 09:00  exfiltration low & slow 10.0.1.46 → 203.0.113.77")
    print("     (la n°4 reste dans les seuils normaux : seule l'Isolation Forest")
    print("      la voit, sur la combinaison 4 conn → 1 hôte → 1 port)")
    print("\n→ Lancez :  py main.py --soc \"Analyse les logs et rédige le rapport d'incident\"")

if __name__ == "__main__":
    main()
