"""Prétraitement et découpage sans fuite de données (data leakage).

Les paramètres du prétraitement sont ajustés uniquement sur les données
d'entraînement, puis appliqués à la validation et au test :

    MAUVAIS :  fit(train + val + test) -> split     # le test a influencé le fit
    BON     :  split -> fit(train) -> transform(*)  # le test reste inconnu

Paramètres appris : vocabulaires de `proto` et `service` (hors vocabulaire ->
`__other__`), médianes d'imputation, moyennes / écarts-types de mise à
l'échelle. Seule exception : `log1p` sur `sbytes` est une transformation
fixe qui n'apprend aucun paramètre — appliquée à tous les jeux, ce n'est
pas une fuite.

Découpage stratifié sur `label` : les trois sous-jeux conservent la même
proportion normal/attaque (~55/45 chez UNSW-NB15).

Usage (lancé en module, sinon `agent` n'est pas trouvable) :
    python -m agent.preprocess        # démontre l'absence de fuite
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agent.flow_features import CATEGORICALS, NUMERICS

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# `sbytes` est à queue lourde (médiane 534, max 14 M chez UNSW) : log1p
# réduit la domination de la queue sans rien apprendre des données.
LOG_FEATURES = ["sbytes"]

# Vocabulaire plafonné par colonne catégorielle, retenu sur le train.
MAX_CATEGORIES = 20

OTHER = "__other__"

def split_stratified(
    df: pd.DataFrame,
    label_col: str = "label",
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe en train / validation / test en stratifiant sur le libellé.

    Sans stratification, un découpage aléatoire pourrait faire varier le taux
    d'attaque d'un jeu à l'autre et biaiser la comparaison des métriques.

    Returns:
        (train, val, test) — 70 % / 15 % / 15 %.
    """
    if label_col not in df.columns:
        raise KeyError(f"Colonne de libellé absente : {label_col}")

    train_val, test = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df[label_col],
        random_state=seed,
    )

    # 2e découpage sur les 85 % restants : test_size relatif
    # = 15 / (70 + 15) = 0.1765...
    rel_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val,
        test_size=rel_val,
        stratify=train_val[label_col],
        random_state=seed,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )

class Preprocessor:
    """Médianes, moyennes/écarts-types et vocabulaires — fit sur le train.

    Le même objet sert à transformer les trois sous-jeux ET nos journaux :
    c'est ce qui garantit qu'aucun jeu n'a pu influencer les paramètres.
    """

    def __init__(self) -> None:
        self.num_cols: list[str] = list(NUMERICS)
        self.cat_cols: list[str] = list(CATEGORICALS)
        self.vocab: dict[str, list[str]] = {}
        self.median: dict[str, float] = {}
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.fitted = False

    @staticmethod
    def _apply_log(work: pd.DataFrame) -> pd.DataFrame:
        """log1p sur les features à queue lourde — fonction fixe, sans paramètre."""
        for col in LOG_FEATURES:
            if col in work.columns:
                work[col] = np.log1p(work[col].clip(lower=0))
        return work

    def _num_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.num_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Colonnes numériques absentes : {missing}")
        work = df[self.num_cols].astype(float).copy()
        return self._apply_log(work)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Ajuste tous les paramètres sur `df` — qui doit être le train.

        Appeler avec val/test/relevé complet serait une fuite.
        """
        work = self._num_raw(df)
        for col in self.num_cols:
            self.median[col] = float(work[col].median())

        work = work.fillna(pd.Series(self.median))
        for col in self.num_cols:
            self.mean[col] = float(work[col].mean())
            std = float(work[col].std())
            self.std[col] = std if std > 1e-12 else 1.0

        for col in self.cat_cols:
            if col not in df.columns:
                raise KeyError(f"Colonne catégorielle absente : {col}")
            top = (
                df[col].astype(str)
                .value_counts()
                .head(MAX_CATEGORIES)
                .index.tolist()
            )
            self.vocab[col] = [OTHER] + [str(v) for v in top]

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique les paramètres du train. Ne réapprend rien ici."""
        if not self.fitted:
            raise RuntimeError("Preprocessor non ajusté : appeler fit(train) d'abord.")

        out = pd.DataFrame(index=df.index)

        work = self._num_raw(df)
        for col in self.num_cols:
            v = work[col].fillna(self.median[col])
            out[col] = (v - self.mean[col]) / self.std[col]

        # One-hot sur le vocabulaire du train, largeur fixe : hors vocabulaire
        # -> __other__ plutôt qu'une colonne fantôme.
        for col in self.cat_cols:
            s = df[col].astype(str)
            known = set(self.vocab[col])
            s = s.where(s.isin(known), OTHER)
            for cat in self.vocab[col]:
                out[f"{col}__{cat}"] = (s == cat).astype(float)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def to_dict(self) -> dict:
        return {
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "vocab": self.vocab,
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "log_features": LOG_FEATURES,
            "max_categories": MAX_CATEGORIES,
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Preprocessor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        pre = cls()
        pre.num_cols = data["num_cols"]
        pre.cat_cols = data["cat_cols"]
        pre.vocab = data["vocab"]
        pre.median = {k: float(v) for k, v in data["median"].items()}
        pre.mean = {k: float(v) for k, v in data["mean"].items()}
        pre.std = {k: float(v) for k, v in data["std"].items()}
        pre.fitted = True
        return pre

def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from agent.flow_features import load_unsw

    unsw_path = Path(r"D:\datasets\unsw_nb15\UNSW_NB15_testing-set.csv")
    df = load_unsw(unsw_path)
    print("=" * 78)
    print("DÉCOUPAGE + PRÉTRAITEMENT SANS FUITE")
    print("=" * 78)
    print(f"ensemble complet : {len(df):,} lignes")

    train, val, test = split_stratified(df)

    print("\n1. Proportion normal/attaque par sous-jeu")
    print(f"   {'sous-jeu':10s} {'n':>8s} {'normal':>8s} {'attaque':>8s} {'% attaque':>10s}")
    for name, part in (("train", train), ("val", val), ("test", test)):
        n = len(part)
        n_att = int((part["label"] == 1).sum())
        print(f"   {name:10s} {n:>8,} {n - n_att:>8,} {n_att:>8,} "
              f"{100 * n_att / n:>9.1f}%")

    print("\n2. fit() sur le train uniquement -> transform() partout")
    pre = Preprocessor().fit(train)
    X_train = pre.transform(train)
    X_val = pre.transform(val)
    X_test = pre.transform(test)
    print(f"   train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    print("\n3. Contrôle anti-fuite : le fit change-t-il si le test y entre ?")
    wrong = Preprocessor().fit(pd.concat([train, val, test], ignore_index=True))
    diffs = [
        col for col in pre.num_cols
        if abs(pre.mean[col] - wrong.mean[col]) > 1e-9
    ]
    print(f"   paramètres 'train-only'  : moyenne sbytes = {pre.mean['sbytes']:.6f}")
    print(f"   paramètres 'tous les jeux': moyenne sbytes = {wrong.mean['sbytes']:.6f}")
    print(f"   -> {'LES PARAMÈTRES DIFFÈRENT' if diffs else 'IDENTIQUES'} "
          f"({len(diffs)} colonnes numériques)")

    print("\n4. Le vocabulaire vient bien du train")
    for col in pre.cat_cols:
        print(f"   {col:8s} {len(pre.vocab[col]):>2d} entrées du train "
              f"-> {pre.vocab[col][:6]}{' ...' if len(pre.vocab[col]) > 6 else ''}")

    print("\n5. Largeur de la matrice (doit être identique pour nos journaux)")
    print(f"   {X_train.shape[1]} colonnes : "
          f"{len(pre.num_cols)} numériques + one-hot catégorielles")

if __name__ == "__main__":
    main()
