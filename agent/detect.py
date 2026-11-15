"""Détection d'anomalies par Isolation Forest (Phase 2).

Non supervisé : le modèle apprend la forme du « normal » sur les données
d'entraînement, puis isole les points rares dans l'espace des features. La
forêt (200 arbres sur sous-échantillons) fait ressortir ce qui n'anormal que
sur quelques dimensions — c'est ce qui permet de voir les attaques
« low & slow » invisibles sur une seule feature.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

from agent.features import FEATURE_COLS, build_features

DEFAULT_CONTAMINATION = 0.02
DEFAULT_SEED = 42

# sklearn limite max_samples à 256 par défaut. Avec ~28 000 profils, une
# attaque n'apparaît que dans ~1 arbre sur 100 et se retrouve classée ~200e.
# À 4096 elle entre dans ~15 % des arbres : les 3 attaques connues passent
# aux rangs 1, 6 et 8, pour 0,1 s de coût en plus.
DEFAULT_MAX_SAMPLES = 4096

# Split temporel (holdout). « Non supervisé » ne veut pas dire « pas de
# split » : sans y, pas de split stratifié, mais le modèle doit être évalué
# sur des lignes jamais vues — sinon on mesure la resubstitution, pas la
# capacité de détection. Le seuil de décision vient de `contamination`, que
# sklearn dérive de la distribution des scores du train : scorer les mêmes
# lignes calibrerait le seuil sur le jeu d'évaluation.
#
# Frontière : la veille de la 1re attaque. generate_logs.py produit 30 jours
# (DAY0=2026-10-10) et les attaques tombent aux jours 24-29, donc
#   jours 0-23 (10/10 → 02/11) : 100 % benignes -> entraînement
#   jours 24-29 (03/11 → 08/11) : les 4 attaques -> évaluation
# Le split est temporel et non aléatoire : un split aléatoire mettrait les
# mêmes motifs horaires des deux côtés (fuite).
DEFAULT_SPLIT_DATE = "2026-11-03"

def fit_and_score(
    logs_dir: Path | str,
    contamination: float = DEFAULT_CONTAMINATION,
    seed: int = DEFAULT_SEED,
    split_date: str | None = DEFAULT_SPLIT_DATE,
) -> pd.DataFrame:
    """Construit les features, ajuste l'Isolation Forest et retourne le tableau annoté.

    Le modèle est fit exclusivement sur les lignes ``hour < split_date`` puis
    évalue les lignes ``hour >= split_date`` — jamais servies à l'entraînement.

    Args:
        logs_dir: répertoire contenant auth.log, firewall.log, app.log
        contamination: part attendue d'anomalies (0.02 = 2 %)
        seed: graine aléatoire → résultats reproductibles
        split_date: frontière ``"YYYY-MM-DD"``. ``None`` = fit + score sur
            toutes les lignes (mode déploiement, marqué ``resubstitution``).

    Returns:
        DataFrame du jeu de test avec ``anomaly`` (bool) et ``anomaly_score``
        (0-100), trié par score décroissant. Métadonnées dans ``df.attrs``.
    """
    features = build_features(logs_dir)
    if features.empty:
        return features

    hours = features.index.get_level_values("hour")

    if split_date:
        train = features[hours < split_date]
        test = features[hours >= split_date]
        if train.empty or test.empty:
            msg = (
                f"split_date={split_date!r} invalide : "
                f"{len(train)} lignes d'entraînement, {len(test)} de test. "
                f"Bornes disponibles : {hours.min()} → {hours.max()}."
            )
            raise ValueError(msg)
    else:
        train = test = features

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples=DEFAULT_MAX_SAMPLES,
        random_state=seed,
        n_jobs=-1,
    )
    # fit() sur le jeu d'entraînement seul : c'est lui qui définit la
    # frontière du « normal » et le seuil dérivé de `contamination`.
    model.fit(train[FEATURE_COLS].to_numpy())

    # Évaluation sur des lignes jamais vues. decision_function applique le
    # seuil du train (score_samples donnerait le même classement, sans l'offset).
    X_test = test[FEATURE_COLS].to_numpy()
    raw = model.predict(X_test)
    raw_scores = model.decision_function(X_test)

    result = test.copy()
    result["anomaly"] = raw == -1

    # Scores ≤ 0 (log-densité) : remap sur le jeu de test, min -> 100.
    lo, hi = raw_scores.min(), raw_scores.max()
    span = (hi - lo) or 1.0
    result["anomaly_score"] = ((hi - raw_scores) / span * 100).round(1)

    result = result.sort_values("anomaly_score", ascending=False)

    if split_date:
        th, eh = train.index.get_level_values("hour"), test.index.get_level_values("hour")
        meta = {
            "mode": "holdout",
            "split_date": split_date,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_period": f"{th.min()} → {th.max()}",
            "test_period": f"{eh.min()} → {eh.max()}",
        }
    else:
        meta = {
            "mode": "resubstitution",
            "split_date": None,
            "train_rows": int(len(features)),
            "test_rows": int(len(features)),
            "train_period": f"{hours.min()} → {hours.max()}",
            "test_period": f"{hours.min()} → {hours.max()}",
            "warning": (
                "fit et score sur les MÊMES lignes : seuil calibré sur le jeu "
                "d'évaluation. Résultat non valide comme mesure de détection."
            ),
        }
    result.attrs["split"] = meta
    return result

def top_anomalies(table: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Filtre les lignes marquées anomalies, triées du plus au moins suspect."""
    if "anomaly" not in table.columns or table.empty:
        return table
    return table[table["anomaly"]].head(limit)

def format_report(table: pd.DataFrame, limit: int = 15) -> str:
    """Formatte les anomalies en JSON lisible pour que le LLM puisse les exploiter."""
    if table.empty:
        return json.dumps(
            {"anomalies": [], "message": "Aucune anomalie détectée."},
            ensure_ascii=False, indent=2,
        )

    flagged = top_anomalies(table, limit)
    entries = []
    for (ip, hour), row in flagged.iterrows():
        # Features non nulles seulement : le LLM voit l'évidence, pas le bruit.
        active = {
            col: (int(row[col]) if float(row[col]).is_integer() else round(float(row[col]), 2))
            for col in FEATURE_COLS
            if float(row[col]) != 0
        }
        entries.append({
            "ip": ip,
            "hour": hour,
            "anomaly_score": float(row["anomaly_score"]),
            "features": active,
        })

    payload = {
        "model": "IsolationForest(n_estimators=200, contamination="
                 f"{DEFAULT_CONTAMINATION}, max_samples={DEFAULT_MAX_SAMPLES}, "
                 f"random_state={DEFAULT_SEED})",
        "evaluation": table.attrs.get(
            "split",
            {"mode": "unknown", "note": "métadonnées de split absentes"},
        ),
        "total_hourly_profiles": int(table.shape[0]),
        "flagged": int(table["anomaly"].sum()),
        "shown": len(entries),
        "anomalies": entries,
        "interpretation": (
            "anomaly_score proche de 100 = très inhabituel. "
            "Une ligne listant plusieurs features simultanément anormales "
            "est plus suspecte qu'une seule feature isolée."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("workspace/logs")
    df = fit_and_score(logs)
    print(format_report(df))
