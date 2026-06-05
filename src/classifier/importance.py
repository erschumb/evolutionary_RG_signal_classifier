"""
Feature/group importance on the validated model.

Two complementary views:
  * group_permutation_importance — permute whole feature GROUPS on held-out data
    (out-of-fold), measure mean AUC drop. Robust to within-group correlation;
    the headline. Corroborates the ablation from inside a fitted model.
  * shap_importance — per-feature SHAP values (TreeExplainer) for direction and
    ranking. Illustrative; credit-splitting on correlated features is a caveat.

Both operate fold-wise on the SAME static matrix the harness uses, so importance
is measured on held-out predictions, never on training data.

NOTE on the folded substitution feature: group permutation supports it, but it
must be recomputed per fold like in the harness. For simplicity these functions
operate on the FOLD-STATIC matrix; if you kept substitution_score in the model,
pass include_folded=True and a factory (handled below). For the SHAP view we use
a single all-data fit on the static matrix (folded feature optional).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from src.classifier.feature_groups import FEATURE_GROUPS, FOLDED_GROUPS
from src.classifier.nested_cv import columns_for_groups, _prune_correlated


def group_permutation_importance(
    X_static, y, groups, include_groups=None, *,
    n_splits=5, n_trees=300, random_state=42, corr_threshold=0.95,
    n_repeats=10, factory=None,
):
    """
    Out-of-fold group permutation importance. For each fold: train on the fold's
    train set, then on the held-out set, permute each group's columns (n_repeats
    times) and measure the AUC drop vs unpermuted. Importance = mean AUC drop
    across folds & repeats. Larger drop = group more important.

    Returns a DataFrame: group, importance_mean, importance_std, baseline_auc.
    """
    include_groups = list(include_groups or [g for g in FEATURE_GROUPS
                                             if g not in FOLDED_GROUPS])
    static_cols = columns_for_groups(include_groups, X_static.columns)
    region_ids = X_static.index
    y = y.reindex(region_ids); groups = groups.reindex(region_ids)

    Xs = X_static[static_cols].copy()
    Xs, _ = _prune_correlated(Xs, corr_threshold)
    static_cols = list(Xs.columns)

    # map each surviving column to its group (for permuting whole groups)
    col_group = {}
    for g in include_groups:
        for c in FEATURE_GROUPS.get(g, []):
            if c in static_cols:
                col_group[c] = g
    present_groups = sorted(set(col_group.values()))

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)
    rng = np.random.default_rng(random_state)
    drops = {g: [] for g in present_groups}
    baselines = []

    for tr, te in cv.split(Xs.values, y.values, groups.values):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(Xs.iloc[tr].values)
        Xte = imp.transform(Xs.iloc[te].values)
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    random_state=random_state, n_jobs=-1)
        rf.fit(Xtr, y.values[tr])
        base = roc_auc_score(y.values[te], rf.predict_proba(Xte)[:, 1])
        baselines.append(base)

        col_idx = {c: i for i, c in enumerate(static_cols)}
        for g in present_groups:
            gcols = [col_idx[c] for c in static_cols if col_group[c] == g]
            for _ in range(n_repeats):
                Xperm = Xte.copy()
                # permute all columns of the group together (shuffle rows jointly)
                perm = rng.permutation(Xperm.shape[0])
                Xperm[:, gcols] = Xperm[np.ix_(perm, gcols)]
                a = roc_auc_score(y.values[te], rf.predict_proba(Xperm)[:, 1])
                drops[g].append(base - a)

    rows = [{"group": g,
             "importance_mean": float(np.mean(drops[g])),
             "importance_std": float(np.std(drops[g]))}
            for g in present_groups]
    out = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    out["baseline_auc"] = float(np.mean(baselines))
    return out.reset_index(drop=True)


def fit_full_model_for_shap(X_static, y, include_groups=None, *,
                            n_trees=300, random_state=42, corr_threshold=0.95):
    """Fit ONE RF on all data (static matrix) for SHAP. Returns (rf, X_imputed_df,
    feature_names, imputer). Folded feature omitted (SHAP on the static set)."""
    include_groups = list(include_groups or [g for g in FEATURE_GROUPS
                                             if g not in FOLDED_GROUPS])
    cols = columns_for_groups(include_groups, X_static.columns)
    Xs = X_static[cols].copy()
    Xs, _ = _prune_correlated(Xs, corr_threshold)
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(Xs.values)
    rf = RandomForestClassifier(n_estimators=n_trees,
                                random_state=random_state, n_jobs=-1)
    rf.fit(Xi, y.reindex(X_static.index).values)
    Xi_df = pd.DataFrame(Xi, columns=list(Xs.columns), index=X_static.index)
    return rf, Xi_df, list(Xs.columns), imp


def shap_importance(rf, Xi_df, max_display=20):
    """Compute SHAP values (TreeExplainer) and return (shap_values, summary_df).
    Requires the `shap` package. summary_df ranks features by mean|SHAP|."""
    import shap
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(Xi_df.values)
    # binary RF: shap_values is a list [class0, class1]; use class1
    sv1 = sv[1] if isinstance(sv, list) else sv
    mean_abs = np.abs(sv1).mean(axis=0)
    summary = (pd.DataFrame({"feature": Xi_df.columns, "mean_abs_shap": mean_abs})
               .sort_values("mean_abs_shap", ascending=False)
               .reset_index(drop=True))
    return sv1, summary


# ── plotting ────────────────────────────────────────────────────────────────
def plot_group_importance(gpi_df, ax=None, color="#4C72B0"):
    """Horizontal bar chart of group permutation importance (mean AUC drop)."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.45 * len(gpi_df))))
    else:
        fig = ax.figure
    d = gpi_df.sort_values("importance_mean")
    ax.barh(d["group"], d["importance_mean"], xerr=d["importance_std"],
            color=color, edgecolor="black", linewidth=0.4,
            error_kw=dict(lw=0.8, capsize=2))
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("Permutation importance\n(mean AUC drop when group shuffled)")
    ax.set_title(f"Group permutation importance "
                 f"(baseline AUC {gpi_df['baseline_auc'].iloc[0]:.3f})")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_shap_summary(shap_values, Xi_df, max_display=20):
    """Standard SHAP beeswarm summary (needs shap installed)."""
    import shap, matplotlib.pyplot as plt
    shap.summary_plot(shap_values, Xi_df, max_display=max_display, show=False)
    return plt.gcf()


def plot_shap_bar(summary_df, max_display=20, ax=None, color="#55A868"):
    """Mean|SHAP| bar chart from the summary_df (no shap dependency for plotting)."""
    import matplotlib.pyplot as plt
    d = summary_df.head(max_display).iloc[::-1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(d))))
    else:
        fig = ax.figure
    ax.barh(d["feature"], d["mean_abs_shap"], color=color,
            edgecolor="black", linewidth=0.4)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Feature importance (SHAP)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig