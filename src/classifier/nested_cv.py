"""
Nested grouped-CV harness for the RG-motif classifier.

Design contract (the leakage-safe core):
  * OUTER StratifiedGroupKFold (grouped by accession) measures generalization.
  * Per outer-train fold: the FOLDED substitution feature is refit on that fold's
    training regions only, then concatenated with the static matrix sliced to the
    same regions; impute -> RF. Applied to the outer-test fold.
  * Correlation pruning is label-free, so it may run once up front (safe). RFECV
    uses labels and, if enabled, runs INSIDE each fold.
  * Every analysis (full / leave-one-group-out / group-in-isolation) is the SAME
    harness called with a different set of included groups, so the folded feature
    is always refit per fold and never leaks.

This module orchestrates; it imports the transformer and the group dict.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from src.classifier.feature_groups import FEATURE_GROUPS, FOLDED_GROUPS
from src.classifier.features_folded import SubstitutionScoreTransformer   # real import
# from _subscore_for_harness import SubstitutionScoreTransformer  # test stub


 
# ── group/column resolution ────────────────────────────────────────────────
def columns_for_groups(group_names, available_cols):
    """Resolve a list of group names to the columns present in `available_cols`.
    Lenient: a listed column missing from the matrix is skipped. Folded-group
    columns are NOT included here (they're produced per-fold by the transformer)."""
    cols = []
    for g in group_names:
        if g in FOLDED_GROUPS:
            continue  # produced per fold, not taken from the static matrix
        for c in FEATURE_GROUPS.get(g, []):
            if c in available_cols and c not in cols:
                cols.append(c)
    return cols
 
 
def report_ungrouped(available_cols):
    """Columns in the matrix that belong to no fine group (would never be tested)."""
    grouped = {c for cols in FEATURE_GROUPS.values() for c in cols}
    return [c for c in available_cols if c not in grouped]
 
 
# ── the harness ─────────────────────────────────────────────────────────────
def run_nested_cv(
    X_static: pd.DataFrame,          # static feature matrix, indexed by region_id
    y: pd.Series,                    # labels, indexed by region_id (0/1)
    groups: pd.Series,              # accession per region, indexed by region_id
    include_groups,                  # iterable of fine-group names to use
    *,
    folded_transformer_factory=None, # callable(**kw)->transformer, or None
    n_splits=5,
    n_trees=300,
    random_state=42,
    corr_threshold=0.95,
    return_oof=False,
):
    """
    Run outer grouped CV using only `include_groups`. If 'substitution_score'
    (or any FOLDED_GROUPS member) is in include_groups, the provided
    folded_transformer_factory is fit per outer-train fold and its output columns
    are concatenated to the static columns for that fold.
 
    Returns dict: per_fold_auc, mean_auc, std_auc, n_static_cols, used_folded,
    and (optionally) out-of-fold predictions.
    """
    include_groups = list(include_groups)
    static_cols = columns_for_groups(include_groups, X_static.columns)
    use_folded = any(g in FOLDED_GROUPS for g in include_groups)
    if use_folded and folded_transformer_factory is None:
        raise ValueError("folded group requested but no folded_transformer_factory given")
 
    # align everything on region_id
    region_ids = X_static.index
    y = y.reindex(region_ids)
    groups = groups.reindex(region_ids)
 
    Xs = X_static[static_cols].copy()
 
    # label-free correlation pruning (safe up front; on static cols only)
    if static_cols:
        Xs, _dropped = _prune_correlated(Xs, corr_threshold)
        static_cols = list(Xs.columns)
 
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)
    fold_aucs = []
    oof = pd.Series(np.nan, index=region_ids, dtype=float)
    oof_fold = pd.Series(np.nan, index=region_ids, dtype=float)
 
    # for tr_idx, te_idx in cv.split(Xs.values, y.values, groups.values):
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(Xs.values, y.values, groups.values)):  # CHANGED: enumerate
        tr_ids = region_ids[tr_idx]
        te_ids = region_ids[te_idx]
 
        X_tr = Xs.iloc[tr_idx].copy()
        X_te = Xs.iloc[te_idx].copy()
 
        # folded feature: fit on TRAIN regions only, transform both
        if use_folded:
            tf = folded_transformer_factory()
            tf.fit(pd.Series(tr_ids), y.loc[tr_ids])
            f_tr = tf.transform(pd.Series(tr_ids)).reindex(tr_ids)
            f_te = tf.transform(pd.Series(te_ids)).reindex(te_ids)
            if X_tr.shape[1] == 0:
                # folded-only case (e.g. isolating substitution_score): no static
                # columns at all — use the folded feature(s) directly.
                X_tr = f_tr.reset_index(drop=True)
                X_te = f_te.reset_index(drop=True)
            else:
                X_tr = pd.concat([X_tr.reset_index(drop=True),
                                  f_tr.reset_index(drop=True)], axis=1)
                X_te = pd.concat([X_te.reset_index(drop=True),
                                  f_te.reset_index(drop=True)], axis=1)
 
        # guard: a configuration with zero usable columns can't be modeled
        if X_tr.shape[1] == 0:
            raise ValueError(
                "configuration produced 0 feature columns (no static columns and "
                "no folded feature) — check include_groups / column names")
 
        # impute (fit on train only). all-NaN columns (e.g. folded score for a
        # fold where every region lacked variants) would survive as NaN; guard.
        imp = SimpleImputer(strategy="median")
        X_tr_i = imp.fit_transform(X_tr.values)
        X_te_i = imp.transform(X_te.values)
        # SimpleImputer drops all-NaN columns silently; if that left 0 cols, bail
        if X_tr_i.shape[1] == 0:
            raise ValueError(
                "all feature columns were all-NaN after imputation for this fold "
                "(folded-only config where training regions had no variants?)")
 
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    random_state=random_state, n_jobs=-1)
        rf.fit(X_tr_i, y.loc[tr_ids].values)
        p = rf.predict_proba(X_te_i)[:, 1]
        fold_aucs.append(roc_auc_score(y.loc[te_ids].values, p))
        oof.loc[te_ids] = p
        oof_fold.loc[te_ids] = fold_i
 
    out = {
        "include_groups": include_groups,
        "per_fold_auc": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "n_static_cols": len(static_cols),
        "used_folded": use_folded,
    }
    if return_oof:
        out["oof"] = oof
        out["oof_fold"] = oof_fold
    return out
 
 
def _prune_correlated(df, threshold):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        for partner in upper.index[upper[col] > threshold]:
            if partner in to_drop:
                continue
            loser = partner if df[col].var() >= df[partner].var() else col
            to_drop.add(loser)
    keep = [c for c in df.columns if c not in to_drop]
    return df[keep], sorted(to_drop)
 
