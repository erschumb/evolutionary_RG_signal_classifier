"""
Goal 3: train the final deployable model on ALL labeled data, persist it, and
score a new set of RG-motif regions.

Design: the final model is FOLD-STATIC only (no substitution_score) — that
feature was non-contributory, and excluding it means there is no per-fold
transformer to carry into deployment. So the pipeline is simply:
    static features -> correlation pruning -> median impute -> RandomForest

Persistence stores everything needed to score new data identically:
    the fitted RF, the imputer, the exact feature column order, the pruning
    decision, and the build-time feature config (so the new set is featurized
    the same way).
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.classifier.feature_groups import FEATURE_GROUPS, FOLDED_GROUPS
from src.classifier.nested_cv import columns_for_groups, _prune_correlated


def train_final_model(
    X_static, y, *,
    include_groups=None, n_trees=300, random_state=42, corr_threshold=0.95,
):
    """
    Fit the final fold-static model on ALL data. No CV (we're done measuring;
    this is the deployable artifact). Folded groups are excluded by design.

    Returns a dict bundle with everything needed to score new data:
        rf, imputer, feature_cols (post-pruning, in order), include_groups,
        corr_threshold, n_trees, random_state, train_auc_note
    """
    include_groups = list(include_groups or [g for g in FEATURE_GROUPS
                                             if g not in FOLDED_GROUPS])
    # safety: refuse folded groups in the deployable model
    folded_req = [g for g in include_groups if g in FOLDED_GROUPS]
    if folded_req:
        raise ValueError(f"final model is fold-static only; remove folded groups "
                         f"{folded_req} from include_groups")

    cols = columns_for_groups(include_groups, X_static.columns)
    Xs = X_static[cols].copy()
    Xs, dropped = _prune_correlated(Xs, corr_threshold)
    feature_cols = list(Xs.columns)

    imputer = SimpleImputer(strategy="median")
    Xi = imputer.fit_transform(Xs.values)

    rf = RandomForestClassifier(n_estimators=n_trees,
                                random_state=random_state, n_jobs=-1)
    rf.fit(Xi, y.reindex(X_static.index).values)

    return {
        "rf": rf,
        "imputer": imputer,
        "feature_cols": feature_cols,       # exact order the RF expects
        "include_groups": include_groups,
        "pruned_cols": dropped,
        "corr_threshold": corr_threshold,
        "n_trees": n_trees,
        "random_state": random_state,
        "n_train_regions": int(len(X_static)),
    }


def save_model(bundle, path):
    """Persist the bundle with joblib. Use a .joblib path."""
    joblib.dump(bundle, path)
    # also drop a small human-readable sidecar of the config
    meta = {k: bundle[k] for k in ("feature_cols", "include_groups",
                                   "pruned_cols", "corr_threshold",
                                   "n_trees", "random_state", "n_train_regions")}
    with open(str(path) + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved model -> {path}  (+ .meta.json sidecar)")


def load_model(path):
    return joblib.load(path)


def score_new_regions(bundle, X_static_new, *, return_proba=True):
    """
    Score a NEW set of regions with a trained bundle.

    X_static_new : static feature matrix for the new regions, built with the
                   SAME build_static_features config as training (same toggles,
                   same codon_source_aas, etc.), indexed by region_id.

    The new matrix is aligned to the model's exact training feature columns:
      - missing columns (a feature absent in the new set) are added as NaN
        (the trained imputer fills them with the TRAINING median)
      - extra columns are dropped
      - column order is matched to the RF's expectation
    The TRAINING imputer is reused (not refit), so new data is filled with
    training-derived medians — no leakage from the new set into itself.

    Returns a Series indexed by region_id: predicted P(functional) (or label).
    """
    cols = bundle["feature_cols"]
    X = X_static_new.copy()

    # add any missing training columns as NaN, drop extras, enforce order
    missing = [c for c in cols if c not in X.columns]
    for c in missing:
        X[c] = np.nan
    extra = [c for c in X.columns if c not in cols]
    X = X[cols]   # selects training cols in training order

    if missing:
        print(f"[score_new_regions] {len(missing)} training features absent in "
              f"new set -> imputed with training medians: {missing[:6]}"
              f"{'...' if len(missing) > 6 else ''}")
    if extra:
        print(f"[score_new_regions] {len(extra)} new-set columns not in model "
              f"-> ignored")

    Xi = bundle["imputer"].transform(X.values)   # TRAINING medians, not refit
    proba = bundle["rf"].predict_proba(Xi)[:, 1]
    out = pd.Series(proba, index=X_static_new.index, name="p_functional")
    if return_proba:
        return out
    return (out >= 0.5).astype(int).rename("pred_label")