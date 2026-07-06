"""
Goal 2: systematic per-group comparison to test "no single group drives it."

Three analyses, all through the validated nested-CV harness:
  * FULL                — all groups together (the headline)
  * leave-one-group-out — drop each group; AUC drop = NECESSITY of that group
  * group-in-isolation  — each group alone; AUC = SUFFICIENCY of that group

Plus the ROC plot (one pooled out-of-fold curve per configuration).

A group is NECESSARY if removing it drops AUC a lot.
A group is SUFFICIENT if it alone gets high AUC.
"no single group drives it" = no group is both highly necessary AND highly
sufficient; the full model beats every group alone.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from src.classifier.feature_groups import FEATURE_GROUPS, FOLDED_GROUPS
from src.classifier.nested_cv import run_nested_cv


def _run(X, y, groups, include, factory, harness_kw):
    return run_nested_cv(
        X, y, groups, include_groups=include,
        folded_transformer_factory=factory, return_oof=True, **harness_kw)


def run_group_comparison(X_static, y, groups, factory=None,
                         group_names=None, do_logo=True, do_isolation=True,
                         **harness_kw):
    """
    Returns (results, table):
      results : dict {label -> harness result dict (with 'oof')} for ROC plotting
      table   : tidy DataFrame summarizing every configuration
    """
    group_names = list(group_names or FEATURE_GROUPS.keys())
    results = {}
    rows = []

    # FULL
    full = _run(X_static, y, groups, group_names, factory, harness_kw)
    results["FULL (all groups)"] = full
    full_auc = full["mean_auc"]
    rows.append({"config": "FULL (all groups)", "type": "full",
                 "group": "—", "n_cols": full["n_static_cols"],
                 "auc": full_auc, "auc_std": full["std_auc"],
                 "delta_vs_full": 0.0})
    print(f"{'FULL (all groups)':<34} AUC {full_auc:.3f} ± {full['std_auc']:.3f}")

    # leave-one-group-out (necessity)
    if do_logo:
        for g in group_names:
            inc = [x for x in group_names if x != g]
            res = _run(X_static, y, groups, inc, factory, harness_kw)
            label = f"− {g}"
            results[label] = res
            rows.append({"config": label, "type": "logo", "group": g,
                         "n_cols": res["n_static_cols"], "auc": res["mean_auc"],
                         "auc_std": res["std_auc"],
                         "delta_vs_full": res["mean_auc"] - full_auc})
            print(f"{label:<34} AUC {res['mean_auc']:.3f} ± {res['std_auc']:.3f} "
                  f"(Δ {res['mean_auc']-full_auc:+.3f})")

    # group-in-isolation (sufficiency)
    if do_isolation:
        for g in group_names:
            # print(g)
            res = _run(X_static, y, groups, [g], factory, harness_kw)
            label = f"{g} only"
            results[label] = res
            rows.append({"config": label, "type": "isolation", "group": g,
                         "n_cols": res["n_static_cols"], "auc": res["mean_auc"],
                         "auc_std": res["std_auc"],
                         "delta_vs_full": res["mean_auc"] - full_auc})
            print(f"{label:<34} AUC {res['mean_auc']:.3f} ± {res['std_auc']:.3f}")

    table = pd.DataFrame(rows)
    return results, table


def plot_necessity_sufficiency(table, ax=None, group_colors=None):
    iso = table[table.type == "isolation"].set_index("group")["auc"]
    logo = table[table.type == "logo"].set_index("group")["delta_vs_full"]
    common = [g for g in iso.index if g in logo.index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    full_auc = float(table[table.type == "full"]["auc"].iloc[0])

    for g in common:
        color = group_colors.get(g, "#333333") if group_colors else None
        ax.scatter(iso[g], -logo[g], s=60, zorder=3, color=color)
        ax.annotate(g, (iso[g], -logo[g]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axvline(full_auc, ls="--", color="#999", lw=1)
    ax.text(full_auc, ax.get_ylim()[1], "full-model AUC", fontsize=7,
            color="#999", ha="center", va="bottom")
    ax.set_xlabel("AUC of group ALONE  (→ more sufficient)")
    ax.set_ylabel("AUC drop when group REMOVED  (↑ more necessary)")
    ax.set_title("Group necessity vs sufficiency")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


# def plot_group_rocs(results, y, configs_to_show=None,
#                     title="Out-of-fold ROC by feature group", ax=None,
#                     sort_by_auc=True):
#     """One pooled out-of-fold ROC curve per configuration in results.
#     configs_to_show: optional list of labels to restrict the plot (else all)."""
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6.5, 6.5))
#     else:
#         fig = ax.figure

#     items = [(k, v) for k, v in results.items()
#              if configs_to_show is None or k in configs_to_show]
#     if sort_by_auc:
#         items.sort(key=lambda kv: kv[1]["mean_auc"], reverse=True)

#     cmap = plt.cm.viridis(np.linspace(0, 0.85, len(items)))
#     for (label, res), color in zip(items, cmap):
#         oof = res["oof"].dropna()
#         yt = y.reindex(oof.index).values
#         pooled_auc = roc_auc_score(yt, oof.values)
#         fpr, tpr, _ = roc_curve(yt, oof.values)
#         ax.plot(fpr, tpr, color=color, lw=1.8,
#                 label=f"{label}  (AUC {pooled_auc:.3f})")
#     ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, zorder=0)
#     ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
#     ax.set_title(title); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
#     ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=7, frameon=False)
#     for s in ("top", "right"):
#         ax.spines[s].set_visible(False)
#     return fig


from matplotlib.lines import Line2D

# def plot_group_rocs(results, y, configs_to_show=None,
#                     title="Out-of-fold ROC by feature group", ax=None,
#                     sort_by_auc=True, group_colors=None, legend=True):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6.5, 6.5))
#     else:
#         fig = ax.figure

#     items = [(k, v) for k, v in results.items()
#              if configs_to_show is None or k in configs_to_show]
#     if sort_by_auc:
#         items.sort(key=lambda kv: kv[1]["mean_auc"], reverse=True)

#     def _group_from_label(label):
#         if label.endswith(" only"):
#             return label[: -len(" only")]
#         if label.startswith("− "):
#             return label[2:]
#         return label  # e.g. "FULL (all groups)"

#     if group_colors is None:
#         cmap = plt.cm.viridis(np.linspace(0, 0.85, len(items)))
#         colors = {k: c for (k, _), c in zip(items, cmap)}
#     else:
#         colors = {k: group_colors.get(_group_from_label(k), "#333333")
#                   for k, _ in items}

#     handles = []
#     for label, res in items:
#         oof = res["oof"].dropna()
#         yt = y.reindex(oof.index).values
#         pooled_auc = roc_auc_score(yt, oof.values)
#         fpr, tpr, _ = roc_curve(yt, oof.values)
#         color = colors[label]
#         ax.plot(fpr, tpr, color=color, lw=1.8)
#         handles.append(Line2D([0], [0], color=color, lw=1.8,
#                               label=f"{_group_from_label(label)} ({pooled_auc:.3f})"))

#     ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, zorder=0)
#     ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
#     ax.set_title(title); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
#     ax.set_aspect("equal")
#     if legend:
#         ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
#     for s in ("top", "right"):
#         ax.spines[s].set_visible(False)
#     return fig

def _per_fold_roc(res, y, n_points=100):
    """Interpolate each fold's ROC onto a common FPR grid.
    Returns (fpr_grid, tpr_mean, tpr_std)."""
    oof, fold_id = res["oof"], res["oof_fold"]
    fpr_grid = np.linspace(0, 1, n_points)
    tprs = []
    for f in sorted(fold_id.dropna().unique()):
        idx = fold_id[fold_id == f].index
        idx = idx[oof.loc[idx].notna()]
        yt = y.reindex(idx).values
        yp = oof.loc[idx].values
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        tpr_i = np.interp(fpr_grid, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
    tprs = np.array(tprs)
    return fpr_grid, tprs.mean(axis=0), tprs.std(axis=0)


def plot_group_rocs(results, y, configs_to_show=None,
                    title="Out-of-fold ROC by feature group", ax=None,
                    sort_by_auc=True, group_colors=None, legend=True,
                    full_label="FULL (all groups)"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig = ax.figure

    items = [(k, v) for k, v in results.items()
             if configs_to_show is None or k in configs_to_show]
    if sort_by_auc:
        items.sort(key=lambda kv: kv[1]["mean_auc"], reverse=True)

    def _group_from_label(label):
        if label.endswith(" only"):
            return label[: -len(" only")]
        if label.startswith("− "):
            return label[2:]
        return label

    non_full = [(k, v) for k, v in items if k != full_label]
    if group_colors is None:
        cmap = plt.cm.viridis(np.linspace(0, 0.85, len(non_full)))
        colors = {k: c for (k, _), c in zip(non_full, cmap)}
    else:
        colors = {k: group_colors.get(_group_from_label(k), "#333333")
                  for k, _ in non_full}

    handles = []

    # FULL: black line + std band from per-fold ROC curves
    full_res = dict(items).get(full_label)
    if full_res is not None:
        fpr_grid, tpr_mean, tpr_std = _per_fold_roc(full_res, y)
        ax.fill_between(fpr_grid,
                        np.clip(tpr_mean - tpr_std, 0, 1),
                        np.clip(tpr_mean + tpr_std, 0, 1),
                        color="black", alpha=0.15, lw=0, zorder=1)
        ax.plot(fpr_grid, tpr_mean, color="black", lw=2.2, zorder=4)
        handles.append(Line2D([0], [0], color="black", lw=2.2,
                              label=f"{full_label} "
                                    f"({full_res['mean_auc']:.3f} ± {full_res['std_auc']:.3f})"))

    # everything else: pooled oof curve, colored by group
    for label, res in non_full:
        oof = res["oof"].dropna()
        yt = y.reindex(oof.index).values
        pooled_auc = roc_auc_score(yt, oof.values)
        fpr, tpr, _ = roc_curve(yt, oof.values)
        color = colors[label]
        ax.plot(fpr, tpr, color=color, lw=1.8, zorder=3)
        handles.append(Line2D([0], [0], color=color, lw=1.8,
                              label=f"{_group_from_label(label)} ({pooled_auc:.3f})"))

    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, zorder=0)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    if legend:
        ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig