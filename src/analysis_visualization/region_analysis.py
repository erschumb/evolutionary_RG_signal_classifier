"""
General per-region analyses.

Dataset-agnostic: same functions work on gnomAD variants and homolog
substitutions. For RG-motif-specific analyses, see rg_analysis.py.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.analysis_visualization.plot_config import (
    FIGSIZE_SINGLE, FIGSIZE_DOUBLE, GROUP_COLORS,
    save_figure, significance_stars,
)


def compute_variant_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    Per-region variant counts and density (variants per residue).

    Returns one row per region with:
        region_id, group, region_length, n_variants, density
    """
    region_len = (
        df.groupby("region_id")[["region_start_aa", "region_end_aa"]].first()
    )
    # Region length in residues — the +1 treats both ends as inclusive
    # (consistent with the JSON convention after our earlier patch).
    region_len["region_length"] = (
        region_len["region_end_aa"] - region_len["region_start_aa"] + 1
    )

    counts = df.groupby(["region_id", "group"]).size().reset_index(name="n_variants")

    out = counts.merge(region_len["region_length"], on="region_id")
    out["density"] = out["n_variants"] / out["region_length"]
    return out


def compare_variant_density(df: pd.DataFrame, dataset: str = "gnomad") -> dict:
    """
    [Dataset-agnostic]
    Mann-Whitney test comparing per-region variant density between pos and neg.
    """
    density_df = compute_variant_density(df)
    pos = density_df.loc[density_df["group"] == "pos", "density"]
    neg = density_df.loc[density_df["group"] == "neg", "density"]

    u_stat, u_p = stats.mannwhitneyu(pos, neg, alternative="two-sided")

    results = {
        "dataset": dataset,
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "median_pos": float(pos.median()),
        "median_neg": float(neg.median()),
        "mean_pos": float(pos.mean()),
        "mean_neg": float(neg.mean()),
        "mannwhitney_U": float(u_stat),
        "mannwhitney_p": float(u_p),
        "mannwhitney_sig": significance_stars(u_p),
    }

    print(f"\n── Variant density per region ({dataset}) ──")
    print(f"  n_pos = {results['n_pos']}, n_neg = {results['n_neg']}")
    print(f"  Median density: pos = {results['median_pos']:.3f}, "
          f"neg = {results['median_neg']:.3f} variants/residue")
    print(f"  Mann-Whitney p = {u_p:.2e} {results['mannwhitney_sig']}")

    return results

# ── VARIANCE DENSITY PLOT ─────────────────────────────────────────────

def plot_variant_density(
    df: pd.DataFrame,
    dataset: str = "gnomad",
    save: bool = True,
    y_max: float | None = None,
) -> tuple[plt.Figure, dict]:
    """
    [Dataset-agnostic]
    Box + strip plot of per-region variant density, pos vs neg.
    """
    density_df = compute_variant_density(df)
    results = compare_variant_density(df, dataset=dataset)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    sns.boxplot(
        data=density_df, x="group", y="density",
        order=["neg", "pos"],
        palette=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
        width=0.5, fliersize=0, linewidth=0.6, ax=ax,
    )
    sns.stripplot(
        data=density_df, x="group", y="density",
        order=["neg", "pos"], color="black", size=1.5, alpha=0.4,
        jitter=0.15, ax=ax,
    )

    # Significance annotation
    ymax = density_df["density"].quantile(0.99)
    y_bar = ymax * 1.05
    ax.plot([0, 1], [y_bar, y_bar], color="black", lw=0.6)
    ax.text(0.5, y_bar * 1.02, results["mannwhitney_sig"],
            ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Variants per residue")
    ax.set_xlabel("")
    if y_max is not None:
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(0, y_bar * 1.15)
    ax.set_title(f"Variant density per region ({dataset})")

    stats_text = (
        f"p = {results['mannwhitney_p']:.1e}\n"
        f"n_pos = {results['n_pos']}\n"
        f"n_neg = {results['n_neg']}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=6.5, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2))

    sns.despine()
    plt.tight_layout()

    if save:
        save_figure(fig, "variant_density", dataset=dataset)

    return fig, results


# ── Consequence-class utilities ─────────────────────────────────────────────

CONSEQUENCE_GROUPS = {
    "synonymous":      {"synonymous_variant"},
    "missense":        {"missense_variant"},
    "LoF":             {"stop_gained", "frameshift_variant",
                        "splice_donor_variant", "splice_acceptor_variant",
                        "start_lost"},
    "inframe_indel":   {"inframe_insertion", "inframe_deletion"},
    # "stop_lost":       {"stop_lost"},
    "other":           set(),   # fallback
}

# Flipped mapping: individual VEP term → group name
_TERM_TO_GROUP = {
    term: group
    for group, terms in CONSEQUENCE_GROUPS.items()
    for term in terms
}

CONSEQUENCE_ORDER = ["synonymous", "missense", "inframe_indel",
                    "LoF", "other"]


def collapse_consequence(consequence: str | None) -> str:
    """
    Collapse VEP Consequence (possibly compound like 'missense_variant&...')
    to one of the broader categories defined in CONSEQUENCE_GROUPS.
    """
    if consequence is None or pd.isna(consequence):
        return "other"
    for term in str(consequence).split("&"):
        if term in _TERM_TO_GROUP:
            return _TERM_TO_GROUP[term]
    return "other"


def compute_consequence_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    Returns long-form df: group, consequence_class, count, proportion.
    """
    df = df.copy()
    df["consequence_class"] = df["Consequence"].apply(collapse_consequence)

    counts = (
        df.groupby(["group", "consequence_class"])
          .size()
          .reset_index(name="count")
    )
    counts["proportion"] = counts.groupby("group")["count"].transform(lambda x: x / x.sum())
    # Put in consistent order
    counts["consequence_class"] = pd.Categorical(
        counts["consequence_class"], categories=CONSEQUENCE_ORDER, ordered=True
    )
    return counts.sort_values(["group", "consequence_class"])


def compare_consequence_distributions(
    df: pd.DataFrame,
    dataset: str = "gnomad",
) -> dict:
    """
    [Dataset-agnostic]
    Overall chi² + per-category Fisher's exact (with Bonferroni correction).
    """
    df = df.copy()
    df["consequence_class"] = df["Consequence"].apply(collapse_consequence)
    contingency = pd.crosstab(df["group"], df["consequence_class"])

    # Overall chi²
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

    # Per-category Fisher: is this category enriched in pos vs neg?
    per_cat = []
    n_tests = len(contingency.columns)
    for cat in contingency.columns:
        pos_in  = contingency.loc["pos", cat]
        pos_out = contingency.loc["pos"].sum() - pos_in
        neg_in  = contingency.loc["neg", cat]
        neg_out = contingency.loc["neg"].sum() - neg_in
        odds, p = stats.fisher_exact([[pos_in, pos_out], [neg_in, neg_out]])
        per_cat.append({
            "consequence_class": cat,
            "pos_count": int(pos_in),
            "neg_count": int(neg_in),
            "pos_prop": pos_in / contingency.loc["pos"].sum(),
            "neg_prop": neg_in / contingency.loc["neg"].sum(),
            "log2_fc": float(np.log2((pos_in / contingency.loc["pos"].sum()) /
                                      (neg_in / contingency.loc["neg"].sum()))
                             if neg_in > 0 and pos_in > 0 else np.nan),
            "odds_ratio": float(odds),
            "p_raw": float(p),
            "p_bonf": min(float(p) * n_tests, 1.0),
            "sig": significance_stars(min(p * n_tests, 1.0)),
        })
    per_cat_df = pd.DataFrame(per_cat)

    results = {
        "dataset": dataset,
        "chi2": float(chi2),
        "chi2_p": float(chi_p),
        "chi2_dof": int(dof),
        "chi2_sig": significance_stars(chi_p),
        "per_category": per_cat_df,
    }

    print(f"\n── Consequence distribution comparison ({dataset}) ──")
    print(f"  Overall chi² = {chi2:.1f}, df = {dof}, p = {chi_p:.2e} {results['chi2_sig']}")
    print(f"  Per-category (Bonferroni-corrected):")
    print(per_cat_df[["consequence_class", "pos_count", "neg_count",
                      "pos_prop", "neg_prop", "log2_fc", "p_bonf", "sig"]]
          .to_string(index=False))
    return results


def plot_consequence_distributions(
    df: pd.DataFrame,
    dataset: str = "gnomad",
    save: bool = True,
) -> tuple[plt.Figure, dict]:
    """
    [Dataset-agnostic]
    Two-panel: (left) proportions per group as grouped bars;
               (right) log2 fold-change enrichment, pos vs neg.
    """
    results = compare_consequence_distributions(df, dataset=dataset)
    props = compute_consequence_proportions(df)
    per_cat = results["per_category"].copy()
    per_cat = per_cat.set_index("consequence_class").reindex(CONSEQUENCE_ORDER).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE,
                              gridspec_kw={"width_ratios": [1.3, 1]})

    # Left: grouped proportions
    ax = axes[0]
    sns.barplot(
        data=props, x="consequence_class", y="proportion", hue="group",
        order=CONSEQUENCE_ORDER, hue_order=["neg", "pos"],
        palette=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
        edgecolor="black", linewidth=0.4, ax=ax,
    )
    ax.set_ylabel("Proportion of variants")
    ax.set_xlabel("")
    ax.set_title("Consequence distribution")
    ax.legend(title=None, frameon=False, loc="upper right")
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    # Annotate overall chi² p
    ax.text(0.02, 0.98,
            f"χ² p = {results['chi2_p']:.1e} {results['chi2_sig']}",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2))

    # Right: log2 fold-change enrichment
    ax = axes[1]
    per_cat = per_cat.dropna(subset=["log2_fc"])
    bar_colors = [
        GROUP_COLORS["pos"] if x > 0 else GROUP_COLORS["neg"]
        for x in per_cat["log2_fc"]
    ]
    bars = ax.barh(
        per_cat["consequence_class"], per_cat["log2_fc"],
        color=bar_colors, edgecolor="black", linewidth=0.4,
    )
    # Significance stars next to each bar
    for bar, sig_mark in zip(bars, per_cat["sig"]):
        w = bar.get_width()
        ax.text(
            w + (0.02 if w >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            sig_mark,
            va="center", ha="left" if w >= 0 else "right", fontsize=7,
        )
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("log₂(pos / neg)")
    ax.set_ylabel("")
    ax.set_title("Enrichment")
    ax.invert_yaxis()

    sns.despine(fig=fig)
    plt.tight_layout()

    if save:
        save_figure(fig, "consequence_distribution", dataset=dataset)

    return fig, results


def plot_median_alphamissense(
    df: pd.DataFrame,
    dataset: str = "gnomad",
    save: bool = True,
) -> tuple[plt.Figure, dict]:
    """
    [gnomAD-specific]
    For each region, compute the median AlphaMissense pathogenicity score
    across its missense variants. Compare pos vs neg.
    """
    mask = (
        df["Consequence"].fillna("").str.contains("missense_variant") &
        df["am_pathogenicity"].notna()
    )
    sub = df[mask]

    per_region = (
        sub.groupby(["region_id", "group"])["am_pathogenicity"]
           .median()
           .reset_index(name="median_am")
    )

    pos_vals = per_region.loc[per_region["group"] == "pos", "median_am"]
    neg_vals = per_region.loc[per_region["group"] == "neg", "median_am"]
    _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    sig = significance_stars(p)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    sns.boxplot(
        data=per_region, x="group", y="median_am",
        order=["neg", "pos"],
        palette=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
        width=0.5, fliersize=0, linewidth=0.6, ax=ax,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 4},
    )
    sns.stripplot(
        data=per_region, x="group", y="median_am",
        order=["neg", "pos"], color="black", size=1.5,
        alpha=0.4, jitter=0.15, ax=ax,
    )

    ymax = per_region["median_am"].quantile(0.98)
    y_bar = ymax * 1.05
    ax.plot([0, 1], [y_bar, y_bar], color="black", lw=0.6)
    ax.text(0.5, y_bar * 1.02, sig, ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, y_bar * 1.15)
    ax.set_title("Median AlphaMissense pathogenicity (per region)")
    ax.set_ylabel("Median AlphaMissense score")
    ax.set_xlabel("")

    stats_text = (
        f"p = {p:.1e} {sig}\n"
        f"n_pos = {len(pos_vals)}\n"
        f"n_neg = {len(neg_vals)}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=6.5, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2))

    sns.despine()
    plt.tight_layout()

    if save:
        save_figure(fig, "region_median_alphamissense", dataset=dataset)

    results = {
        "p": float(p), "sig": sig,
        "median_pos": float(pos_vals.median()),
        "median_neg": float(neg_vals.median()),
        "mean_pos": float(pos_vals.mean()),
        "mean_neg": float(neg_vals.mean()),
        "n_pos": int(len(pos_vals)),
        "n_neg": int(len(neg_vals)),
    }

    print(f"\n── Median AlphaMissense per region ({dataset}) ──")
    print(f"  pos: median={results['median_pos']:.3f}, mean={results['mean_pos']:.3f}, n={results['n_pos']}")
    print(f"  neg: median={results['median_neg']:.3f}, mean={results['mean_neg']:.3f}, n={results['n_neg']}")
    print(f"  Mann-Whitney p = {p:.2e} {sig}")

    return fig, results

