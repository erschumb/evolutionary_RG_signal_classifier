"""
Save as: src/analysis_visualization/homolog_recruitment.py

Phase 1 homolog analyses:
  - Homolog abundance per motif (n_homologs per query-motif pair)
  - Evolutionary persistence (n_unique_species per query-motif pair)
  - Zero-vs-any substitutions (fraction of homolog hits with any AA change)

All analyses operate at the motif level: one row per (UniqueID, orig_motif_index).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.analysis_visualization.plot_config import (
    FIGSIZE_LARGE, GROUP_COLORS, save_figure, significance_stars,
)


# ════════════════════════════════════════════════════════════════════════════
# Data preparation
# ════════════════════════════════════════════════════════════════════════════

def harmonize_group_labels(df: pd.DataFrame, group_col: str = "group") -> pd.DataFrame:
    """
    Harmonize homolog group labels (positive/negative) to match gnomAD (pos/neg).
    """
    df = df.copy()
    df[group_col] = df[group_col].map({
        "positive": "pos", "negative": "neg",
        "pos": "pos", "neg": "neg",
    }).fillna(df[group_col])
    return df


def add_motif_uid(
    df: pd.DataFrame,
    query_col: str = "UniqueID",
    motif_col: str = "orig_motif_index",
) -> pd.DataFrame:
    """
    Add a motif-level unique ID combining query accession and motif index.
    Format: "UniqueID_m<orig_motif_index>" to distinguish from gnomAD region_ids.
    """
    df = df.copy()
    df["motif_uid"] = df[query_col].astype(str) + "_m" + df[motif_col].astype(str)
    return df


def exclude_self_hits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude hits where qseq_rg_region == hseq_rg_region AND species is the query
    species. Self-hits from other species (perfectly conserved) are kept — they
    carry genuine conservation information.

    NOTE: currently this excludes ALL rows where qseq == hseq. Refine based on
    diagnostic (are self-hits mostly same-species?).
    """
    raise NotImplementedError(
        "Need to decide how to treat self-hits — see diagnostic in notebook."
    )


# ════════════════════════════════════════════════════════════════════════════
# Per-motif aggregations
# ════════════════════════════════════════════════════════════════════════════

def compute_recruitment_stats(
    df: pd.DataFrame,
    motif_col: str = "motif_uid",
    group_col: str = "group",
    species_col: str = "species",
    qseq_col: str = "qseq_rg_region",
    hseq_col: str = "hseq_rg_region",
) -> pd.DataFrame:
    """
    [Homolog-specific]
    Per-motif recruitment stats:
        n_homologs               — total BLAST hits
        n_unique_species         — unique species breadth (evolutionary persistence)
        n_homologs_per_species   — redundancy per species
        n_with_substitution      — number of hits with at least one AA change
        fraction_with_sub        — fraction of hits with any AA change
    """
    rows = []
    for motif_uid, sub in df.groupby(motif_col):
        n_total = len(sub)
        n_species = sub[species_col].nunique()
        # Count hits where query seq != homolog seq (any AA change)
        has_sub = sub[qseq_col] != sub[hseq_col]
        n_with_sub = int(has_sub.sum())

        rows.append({
            "motif_uid": motif_uid,
            "group": sub[group_col].iloc[0],
            "n_homologs": int(n_total),
            "n_unique_species": int(n_species),
            "n_homologs_per_species": float(n_total / n_species) if n_species > 0 else np.nan,
            "n_with_substitution": n_with_sub,
            "fraction_with_sub": float(n_with_sub / n_total) if n_total > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

def _box_and_strip(
    ax, data_pos, data_neg, ylabel, title, log_y=False,
):
    """
    Box-and-strip plot for a single per-motif metric.
    Consistent styling: pos green, neg red, violin + embedded box + jittered strip.
    """
    plot_data = [data_neg.values, data_pos.values]
    labels = ["neg", "pos"]
    colors = [GROUP_COLORS["neg"], GROUP_COLORS["pos"]]

    parts = ax.violinplot(
        plot_data, positions=[0, 1], widths=0.75,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.35)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.4)

    # Box plot on top
    bp = ax.boxplot(
        plot_data, positions=[0, 1], widths=0.2, showfliers=False,
        patch_artist=True, zorder=3,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    # Strip plot
    for x_pos, vals, color in zip([0, 1], plot_data, colors):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.full(len(vals), x_pos) + jitter, vals,
            color=color, alpha=0.5, s=6, edgecolor="none", zorder=2,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    if log_y:
        ax.set_yscale("log")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Statistical test
    u, p = stats.mannwhitneyu(data_pos, data_neg, alternative="two-sided")
    sig = significance_stars(p)
    y_top = ax.get_ylim()[1]
    ax.text(
        0.5, 0.96, f"p = {p:.2e} {sig}",
        transform=ax.transAxes, ha="center", va="top", fontsize=7,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
    )
    return {"u": float(u), "p": float(p), "sig": sig}


def plot_recruitment_stats(
    stats_df: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    Three-panel figure:
      1. Homologs per motif (log y)
      2. Unique species per motif
      3. Homologs-per-species ratio (redundancy)
    """
    pos = stats_df[stats_df["group"] == "pos"]
    neg = stats_df[stats_df["group"] == "neg"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    test_results = {}
    test_results["n_homologs"] = _box_and_strip(
        axes[0],
        pos["n_homologs"], neg["n_homologs"],
        ylabel="Homologs per motif",
        title="Homolog abundance",
        log_y=True,
    )
    test_results["n_unique_species"] = _box_and_strip(
        axes[1],
        pos["n_unique_species"], neg["n_unique_species"],
        ylabel="Unique species per motif",
        title="Evolutionary persistence",
        log_y=False,
    )
    test_results["n_homologs_per_species"] = _box_and_strip(
        axes[2],
        pos["n_homologs_per_species"], neg["n_homologs_per_species"],
        ylabel="Homologs / species",
        title="Recruitment density per species",
        log_y=False,
    )

    fig.suptitle(
        f"Homolog recruitment ({dataset})",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "homolog_recruitment", dataset=dataset)

    # Printed summary
    print(f"\n── Recruitment stats ({dataset}) ──")
    print(f"  Pos motifs: {len(pos)}, Neg motifs: {len(neg)}")
    for metric, res in test_results.items():
        print(f"\n  {metric}:")
        print(f"    pos median = {pos[metric].median():.2f}, "
              f"neg median = {neg[metric].median():.2f}")
        print(f"    Mann-Whitney p = {res['p']:.2e} {res['sig']}")

    return test_results


def plot_zero_vs_any_sub(
    stats_df: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    Fraction-of-motif-hits-with-any-AA-change per motif, pos vs neg.
    A motif fully conserved across all homologs → fraction_with_sub = 0.
    A motif where every homolog differs → fraction_with_sub = 1.
    """
    pos = stats_df[stats_df["group"] == "pos"]
    neg = stats_df[stats_df["group"] == "neg"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: box-and-strip of fraction_with_sub
    test_result = _box_and_strip(
        axes[0],
        pos["fraction_with_sub"], neg["fraction_with_sub"],
        ylabel="Fraction of homologs\nwith any AA change in motif",
        title="Homolog-level conservation",
        log_y=False,
    )
    axes[0].set_ylim(-0.02, 1.05)

    # Right: fraction of motifs that are FULLY conserved (fraction_with_sub == 0)
    frac_fully_conserved_pos = float((pos["fraction_with_sub"] == 0).mean())
    frac_fully_conserved_neg = float((neg["fraction_with_sub"] == 0).mean())

    n_fc_pos = int((pos["fraction_with_sub"] == 0).sum())
    n_fc_neg = int((neg["fraction_with_sub"] == 0).sum())

    # Fisher's exact on 2x2 contingency
    table = np.array([
        [n_fc_pos, len(pos) - n_fc_pos],
        [n_fc_neg, len(neg) - n_fc_neg],
    ])
    odds, pval_fisher = stats.fisher_exact(table)
    sig = significance_stars(pval_fisher)

    ax = axes[1]
    x = [0, 1]
    fracs = [frac_fully_conserved_neg, frac_fully_conserved_pos]
    ax.bar(
        x, fracs,
        color=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
        edgecolor="black", linewidth=0.6,
    )
    for xi, (n_fc, total) in enumerate([(n_fc_neg, len(neg)), (n_fc_pos, len(pos))]):
        ax.text(
            xi, fracs[xi] + 0.005,
            f"{n_fc}/{total}",
            ha="center", va="bottom", fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["neg", "pos"])
    ax.set_ylabel("Fraction of motifs\nfully conserved in all homologs")
    ax.set_title("Proportion with zero substitutions", fontsize=9)
    ax.set_ylim(0, max(fracs) * 1.3 + 0.02)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(
        0.5, 0.96, f"Fisher p = {pval_fisher:.2e} {sig}",
        transform=ax.transAxes, ha="center", va="top", fontsize=7,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
    )

    fig.suptitle(
        f"Homolog sequence conservation ({dataset})",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "homolog_zero_vs_any_sub", dataset=dataset)

    print(f"\n── Zero-vs-any substitution ({dataset}) ──")
    print(f"  fraction_with_sub: pos median={pos['fraction_with_sub'].median():.3f}, "
          f"neg median={neg['fraction_with_sub'].median():.3f}, "
          f"Mann-Whitney p={test_result['p']:.2e} {test_result['sig']}")
    print(f"  fully-conserved motifs: pos {n_fc_pos}/{len(pos)} "
          f"({frac_fully_conserved_pos:.1%}), "
          f"neg {n_fc_neg}/{len(neg)} ({frac_fully_conserved_neg:.1%}), "
          f"Fisher p={pval_fisher:.2e} {sig}")

    return {
        "fraction_with_sub_test": test_result,
        "fully_conserved_fisher_p": pval_fisher,
        "fully_conserved_sig": sig,
    }


# ════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_phase1_homolog_analyses(
    df_combined: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    df = harmonize_group_labels(df_combined)
    df = add_motif_uid(df)
    df = compute_substitution_counts_per_hit(df)

    stats_df = compute_recruitment_stats(df)
    per_motif_sub_df = aggregate_substitution_counts_per_motif(df)

    recruitment_results = plot_recruitment_stats(stats_df, dataset=dataset, save=save)
    substitution_results = plot_zero_vs_any_sub(stats_df, dataset=dataset, save=save)
    sub_count_results = plot_substitution_count_breakdown(
        df, per_motif_sub_df, dataset=dataset, save=save,
    )
    stacked_results = plot_substitution_stacked_fade(df, dataset=dataset, save=save)

    return {
        "df_with_subs": df,
        "stats_df": stats_df,
        "per_motif_sub_df": per_motif_sub_df,
        "recruitment_results": recruitment_results,
        "substitution_results": substitution_results,
        "sub_count_results": sub_count_results,
        "stacked_results": stacked_results,
    }

def compute_substitution_counts_per_hit(
    df: pd.DataFrame,
    qseq_col: str = "qseq_rg_region",
    hseq_col: str = "hseq_rg_region",
) -> pd.DataFrame:
    """
    [Homolog-specific]
    For each BLAST hit row, count the number of positions where qseq differs
    from hseq within the motif region. Also excludes aligned positions where
    either side has a gap (since gaps aren't substitutions).
    Returns the dataframe with a new column `n_substitutions`.
    """
    df = df.copy()

    def _count_sub(row):
        q = row[qseq_col]
        h = row[hseq_col]
        if pd.isna(q) or pd.isna(h) or len(q) != len(h):
            return np.nan
        # Count positions where q != h AND neither is a gap
        return sum(
            1 for qc, hc in zip(q, h)
            if qc != hc and qc != "-" and hc != "-"
        )

    df["n_substitutions"] = df.apply(_count_sub, axis=1)
    return df


def aggregate_substitution_counts_per_motif(
    df_with_subs: pd.DataFrame,
    motif_col: str = "motif_uid",
    group_col: str = "group",
) -> pd.DataFrame:
    """
    [Homolog-specific]
    Per-motif aggregation of substitution counts across homolog hits.
    Returns dataframe with one row per motif and columns:
        n_sub_mean, n_sub_median, n_sub_max, n_sub_std
    """
    rows = []
    for motif_uid, sub in df_with_subs.groupby(motif_col):
        counts = sub["n_substitutions"].dropna()
        if len(counts) == 0:
            continue
        motif_length = len(sub["qseq_rg_region"].iloc[0])
        rows.append({
            "motif_uid": motif_uid,
            "group": sub[group_col].iloc[0],
            "n_sub_mean": float(counts.mean()),
            "n_sub_median": float(counts.median()),
            "n_sub_max": int(counts.max()),
            "n_sub_std": float(counts.std()) if len(counts) > 1 else 0.0,
            "motif_length": motif_length,
            # Normalized version (per-position): sub rate
            "mean_sub_rate": float(counts.mean() / motif_length) if motif_length > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def plot_substitution_count_breakdown(
    df_with_subs: pd.DataFrame,
    per_motif_df: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    Three-panel breakdown:
      1. Histogram of n_substitutions per BLAST hit, overlaid pos vs neg
      2. Box-and-strip of per-motif mean n_substitutions
      3. Box-and-strip of per-motif mean substitution RATE (normalized by motif length)
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # ── Panel 1: raw hit-level histogram (overlaid) ─────────────────────
    ax = axes[0]
    pos_hits = df_with_subs.loc[
        df_with_subs["group"] == "pos", "n_substitutions"
    ].dropna()
    neg_hits = df_with_subs.loc[
        df_with_subs["group"] == "neg", "n_substitutions"
    ].dropna()

    max_val = int(max(pos_hits.max(), neg_hits.max()))
    bins = np.arange(0, max_val + 2) - 0.5  # bin edges at integers

    ax.hist(
        neg_hits, bins=bins, density=True,
        color=GROUP_COLORS["neg"], alpha=0.45, edgecolor="black",
        linewidth=0.4, label=f"neg (n={len(neg_hits):,})",
    )
    ax.hist(
        pos_hits, bins=bins, density=True,
        color=GROUP_COLORS["pos"], alpha=0.45, edgecolor="black",
        linewidth=0.4, label=f"pos (n={len(pos_hits):,})",
    )

    # Mann-Whitney on raw hit-level substitution counts
    u, p = stats.mannwhitneyu(pos_hits, neg_hits, alternative="two-sided")
    sig = significance_stars(p)

    ax.set_xlabel("N substitutions (per homolog hit)")
    ax.set_ylabel("Density")
    ax.set_title("Per-hit substitution count", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.text(
        0.98, 0.70, f"Mann-Whitney\np = {p:.2e} {sig}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # ── Panel 2: per-motif mean n_substitutions ─────────────────────────
    pos_motif = per_motif_df[per_motif_df["group"] == "pos"]
    neg_motif = per_motif_df[per_motif_df["group"] == "neg"]

    test2 = _box_and_strip(
        axes[1],
        pos_motif["n_sub_mean"], neg_motif["n_sub_mean"],
        ylabel="Mean substitutions per motif\n(averaged across homologs)",
        title="Per-motif substitution burden",
        log_y=False,
    )

    # ── Panel 3: per-motif mean RATE (normalized) ───────────────────────
    test3 = _box_and_strip(
        axes[2],
        pos_motif["mean_sub_rate"], neg_motif["mean_sub_rate"],
        ylabel="Mean substitution rate\n(per motif position)",
        title="Length-normalized substitution rate",
        log_y=False,
    )

    fig.suptitle(
        f"Homolog substitution count breakdown ({dataset})",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "homolog_substitution_count_breakdown", dataset=dataset)

    print(f"\n── Substitution count breakdown ({dataset}) ──")
    print(f"  Per-hit median n_subs: pos = {pos_hits.median():.1f}, "
          f"neg = {neg_hits.median():.1f}, Mann-Whitney p = {p:.2e} {sig}")
    print(f"  Per-motif median mean n_subs: pos = {pos_motif['n_sub_mean'].median():.2f}, "
          f"neg = {neg_motif['n_sub_mean'].median():.2f}, p = {test2['p']:.2e} {test2['sig']}")
    print(f"  Per-motif median sub rate: pos = {pos_motif['mean_sub_rate'].median():.3f}, "
          f"neg = {neg_motif['mean_sub_rate'].median():.3f}, p = {test3['p']:.2e} {test3['sig']}")

    return {
        "hit_level_test": {"u": float(u), "p": float(p), "sig": sig},
        "motif_mean_test": test2,
        "motif_rate_test": test3,
    }


def plot_substitution_stacked_fade(
    df_with_subs: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
    categories: list = None,
) -> dict:
    """
    [Homolog-specific]
    Stacked bar plot with fading alpha showing the fraction of homolog hits
    in each substitution-count category. Uses group-specific base colors.

    Two panels:
      1. Binary: 0 vs ≥1 substitutions
      2. Detailed: 0, 1, 2, 3, 4, 5, >5 substitutions
    """
    import matplotlib.patches as mpatches

    if categories is None:
        categories = ["0", "1", "2", "3", "4", ">4"]

    # Convert hex colors to RGB tuples
    def _hex_to_rgb(hex_str):
        hex_str = hex_str.lstrip("#")
        return tuple(int(hex_str[i:i+2], 16) / 255 for i in (0, 2, 4))

    base_colors_rgb = {
        "pos": _hex_to_rgb(GROUP_COLORS["pos"]),
        "neg": _hex_to_rgb(GROUP_COLORS["neg"]),
    }

    def _categorize(n):
        if pd.isna(n):
            return None
        n = int(n)
        if n > 4:
            return ">4"
        return str(n)

    df = df_with_subs.copy()
    df["sub_cat"] = df["n_substitutions"].apply(_categorize)

    # ── Compute fractions per category per group ──────────────────────
    fracs_binary = {}
    fracs_detailed = {}
    for group in ["neg", "pos"]:
        sub = df[df["group"] == group]
        n_total = len(sub)
        if n_total == 0:
            fracs_binary[group] = {"0": 0, "≥1": 0}
            fracs_detailed[group] = {c: 0 for c in categories}
            continue
        fracs_binary[group] = {
            "0": float((sub["n_substitutions"] == 0).mean()),
            "≥1": float((sub["n_substitutions"] >= 1).mean()),
        }
        fracs_detailed[group] = {
            cat: float((sub["sub_cat"] == cat).mean()) for cat in categories
        }

    # ── Plot helper ───────────────────────────────────────────────────
    def _plot_stacked_fade(ax, cats, fracs_dict, title):
        groups = ["neg", "pos"]
        alphas = np.linspace(1.0, 0.2, len(cats))
        x = np.arange(len(groups))
        bottoms = np.zeros(len(groups))

        for i, cat in enumerate(cats):
            for j, grp in enumerate(groups):
                r, g, b = base_colors_rgb[grp]
                a = alphas[i]
                val = fracs_dict[grp][cat]
                ax.bar(
                    x[j], val, bottom=bottoms[j],
                    color=(r, g, b, a), width=0.55,
                    edgecolor="black", linewidth=0.3,
                )
            bottoms += np.array([fracs_dict[grp][cat] for grp in groups])

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("Fraction of alignments")
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=9)

        # Grey legend patches showing fade → category
        grey_patches = [
            mpatches.Patch(facecolor=(0.4, 0.4, 0.4, a), label=cat)
            for cat, a in zip(cats, alphas)
        ]
        ax.legend(
            handles=grey_patches, title="N subs",
            bbox_to_anchor=(1.01, 1), loc="upper left",
            fontsize=7, title_fontsize=7, frameon=False,
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _plot_stacked_fade(
        axes[0], ["0", "≥1"], fracs_binary, "Zero vs. any substitutions",
    )
    _plot_stacked_fade(
        axes[1], categories, fracs_detailed, "Substitution count breakdown",
    )

    fig.suptitle(
        f"Homolog substitution categories ({dataset})",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "homolog_substitution_stacked", dataset=dataset)

    # Printed summary
    print(f"\n── Substitution category breakdown ({dataset}) ──")
    print(f"  Binary (zero vs any):")
    for grp in ["neg", "pos"]:
        for cat in ["0", "≥1"]:
            print(f"    {grp} {cat}: {fracs_binary[grp][cat]:.3f}")
    print(f"\n  Detailed:")
    header = "                  " + "  ".join(f"{c:>6}" for c in categories)
    print(header)
    for grp in ["neg", "pos"]:
        vals = "  ".join(f"{fracs_detailed[grp][c]:.3f}" for c in categories)
        print(f"    {grp:<13}: {vals}")

    return {
        "fracs_binary": fracs_binary,
        "fracs_detailed": fracs_detailed,
    }