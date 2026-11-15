"""Entraînement du XGBoost multiclasse sur UNSW-NB15 — sans fuite de données.

Un modèle binaire ne répond qu'à « attaque ou normal ? » et ne peut donc pas
nommer l'attaque. En entraînant sur `attack_cat` (10 classes : Normal +
9 familles), un seul modèle fournit les deux :

    p_attack  = 1 - P(Normal)          -> signal de confirmation
    famille   = argmax des classes     -> « Reconnaissance », « DoS », ...

Ordre strict, sans fuite :
    1. chargement du dataset externe
    2. split stratifié sur `attack_cat` (préserve proportion normal/attaque
       ET proportion de chaque famille dans les 3 jeux)
    3. fit du prétraitement sur le train seul
    4. transform des trois jeux avec les paramètres du train
    5. entraînement, early stopping sur la val
    6. évaluation finale sur le test — jamais vu pendant les étapes 3 à 5

Écrit : models/xgboost.json, models/classes.json, models/preprocess.json,
        models/metrics.json

Usage :  python train_xgboost
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.flow_features import COMMON_FEATURES, load_unsw
from agent.preprocess import Preprocessor, split_stratified

UNSW_PATH = Path(r"D:\datasets\unsw_nb15\UNSW_NB15_testing-set.csv")
MODELS_DIR = Path("models")
SEP = "=" * 78
NORMAL_CLASS = "Normal"

# Modestes volontairement : on cherche à généraliser, pas à mémoriser.
# L'early stopping sur la VAL fixe la vraie profondeur.
PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

def report(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str],
           title: str) -> dict:
    """Affiche accuracy, F1 macro et rappel par classe."""
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    print(f"\n--- {title} ---")
    print(f"  accuracy         : {acc:.4f}")
    print(f"  balanced accuracy: {bacc:.4f}")
    print(f"  F1 macro         : {f1m:.4f}")
    print("  rappel par classe :")
    recalls = {}
    for i, name in enumerate(classes):
        support = cm[i].sum()
        if support == 0:
            continue
        r = cm[i, i] / support
        recalls[name] = float(r)
        print(f"      {name:16s} n={support:>6,}  rappel={r:.4f}")
    return {"accuracy": acc, "balanced_accuracy": bacc,
            "f1_macro": f1m, "per_class_recall": recalls}

def main() -> int:
    print(SEP)
    print("ENTRAÎNEMENT XGBoost MULTICLASSE — UNSW-NB15")
    print(SEP)

    # 1. chargement
    df = load_unsw(UNSW_PATH)
    print(f"\n1. chargement : {len(df):,} lignes, {len(COMMON_FEATURES)} features")

    # 2. split stratifié sur la famille (préserve aussi label)
    train, val, test = split_stratified(df, label_col="attack_cat")
    print(f"2. split stratifié sur attack_cat : train {len(train):,} / "
          f"val {len(val):,} / test {len(test):,}")

    # 3. encodage des classes (fit sur l'ensemble : simple re-labelage,
    #    aucun paramètre statistique appris — pas une fuite)
    le = LabelEncoder().fit(df["attack_cat"])
    classes: list[str] = [str(c) for c in le.classes_]
    n_classes = len(classes)
    print(f"   {n_classes} classes : {classes}")

    # 4. prétraitement fit sur le train seul
    pre = Preprocessor().fit(train)
    print("3. prétraitement ajusté sur le train uniquement")

    X_train, X_val, X_test = (pre.transform(train), pre.transform(val),
                              pre.transform(test))
    y_train = le.transform(train["attack_cat"])
    y_val = le.transform(val["attack_cat"])
    y_test = le.transform(test["attack_cat"])
    print(f"   matrices : {X_train.shape} / {X_val.shape} / {X_test.shape}")

    # 5. entraînement
    print(f"\n4. entraînement multiclasse (early stopping sur la val, "
          f"max {PARAMS['n_estimators']} arbres)...")
    model = XGBClassifier(**PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best = getattr(model, "best_iteration", None)
    print(f"   meilleur itération : {best if best is not None else 'n/a'}")

    # 6. référence naïve
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    d_acc = accuracy_score(y_test, dummy.predict(X_test))
    print(f"\n5. référence naïve (classe la plus fréquente) sur le test : {d_acc:.4f}")

    # 7. évaluation
    metrics_val = report(y_val, model.predict(X_val), classes,
                         "VALIDATION (vue par l'early stopping)")
    metrics_test = report(y_test, model.predict(X_test), classes,
                          "TEST (jamais vu)")

    # 8. persistance
    MODELS_DIR.mkdir(exist_ok=True)
    model.save_model(str(MODELS_DIR / "xgboost.json"))
    pre.save(MODELS_DIR / "preprocess.json")
    (MODELS_DIR / "classes.json").write_text(
        json.dumps({"classes": classes, "normal": NORMAL_CLASS}, indent=2),
        encoding="utf-8",
    )
    (MODELS_DIR / "metrics.json").write_text(json.dumps({
        "task": "multiclass",
        "classes": classes,
        "validation": metrics_val,
        "test": metrics_test,
        "naive_baseline_accuracy": float(d_acc),
        "best_iteration": int(best) if best is not None else None,
        "n_features": int(X_train.shape[1]),
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "attack_rate": {k: round(float(v), 4) for k, v in
                        zip(["train", "val", "test"],
                            [(train["attack_cat"] != NORMAL_CLASS).mean(),
                             (val["attack_cat"] != NORMAL_CLASS).mean(),
                             (test["attack_cat"] != NORMAL_CLASS).mean()])},
    }, indent=2), encoding="utf-8")

    print(SEP)
    print("SYNTHÈSE")
    print(SEP)
    print(f"  naïve (test)          : {d_acc:.4f}")
    print(f"  accuracy    (test)    : {metrics_test['accuracy']:.4f}")
    print(f"  F1 macro    (test)    : {metrics_test['f1_macro']:.4f}")
    print(f"  classes : {classes}")
    print(f"\n  écrit : models/xgboost.json, models/classes.json, "
          f"models/preprocess.json, models/metrics.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
