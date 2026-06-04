
"""
Save as: src/analysis_visualization/af_spectrum.py

Allele frequency spectrum comparison between pos and neg groups.

Panel per consequence class (synonymous / missense / LoF), each showing log10(AF)
distributions for pos and neg as overlaid KDE curves with a rug plot of raw
values and Mann-Whitney U test annotated.

[gnomAD-specific] — requires AF_joint column.
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
from src.analysis_visualization.region_analysis import collapse_consequence


# ════════════════════════════════════════════════════════════════════════════
# Which consequence classes to show — now includes inframe_indel
# ════════════════════════════════════════════════════════════════════════════
 
AF_SPECTRUM_CONSEQUENCES = ["synonymous", "missense", "inframe_indel", "LoF"]
 
 
# ════════════════════════════════════════════════════════════════════════════
# AF bins for the side panel
# ════════════════════════════════════════════════════════════════════════════
 
AF_BINS = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1.0]
AF_BIN_LABELS = [
    "< 1e-5",
    "1e-5\nto\n1e-4",
    "1e-4\nto\n1e-3",
    "1e-3\nto\n1e-2",
    "≥ 1e-2",
]
 
 
# ════════════════════════════════════════════════════════════════════════════
# Stats
# ════════════════════════════════════════════════════════════════════════════
 
def _compute_af_spectrum_stats(
    df: pd.DataFrame,
    consequence_class: str,
    af_col: str,
) -> dict:
    sub = df[
        (df["consequence_class"] == consequence_class) &
        df[af_col].notna() &
        (df[af_col] > 0)
    ]
    pos_af = sub.loc[sub["group"] == "pos", af_col]
    neg_af = sub.loc[sub["group"] == "neg", af_col]
 
    if len(pos_af) < 10 or len(neg_af) < 10:
        return {
            "consequence_class": consequence_class,
            "n_pos": int(len(pos_af)), "n_neg": int(len(neg_af)),
            "median_pos_log10": np.nan, "median_neg_log10": np.nan,
            "u_stat": np.nan, "p": np.nan, "sig": "n.s.",
            "bins_pos": None, "bins_neg": None,
        }
 
    u, p = stats.mannwhitneyu(pos_af, neg_af, alternative="two-sided")
 
    # AF bin counts and fractions
    bins_pos = pd.cut(pos_af, bins=AF_BINS, labels=AF_BIN_LABELS, right=False).value_counts()
    bins_neg = pd.cut(neg_af, bins=AF_BINS, labels=AF_BIN_LABELS, right=False).value_counts()
    bins_pos = bins_pos.reindex(AF_BIN_LABELS, fill_value=0)
    bins_neg = bins_neg.reindex(AF_BIN_LABELS, fill_value=0)
 
    return {
        "consequence_class": consequence_class,
        "n_pos": int(len(pos_af)),
        "n_neg": int(len(neg_af)),
        "median_pos_log10": float(np.log10(pos_af.median())),
        "median_neg_log10": float(np.log10(neg_af.median())),
        "u_stat": float(u),
        "p": float(p),
        "sig": significance_stars(p),
        "bins_pos": bins_pos,
        "bins_neg": bins_neg,
        "pos_af_vals": pos_af,
        "neg_af_vals": neg_af,
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# Plot
# ════════════════════════════════════════════════════════════════════════════
 
def plot_af_spectrum_cdf(
    df: pd.DataFrame,
    af_col: str = "AF_joint",
    dataset: str = "gnomad",
    save: bool = True,
    consequence_classes: list[str] = None,
    af_range: tuple[float, float] = (-7, 0),
) -> dict:
    """
    [gnomAD-specific]
    CDF plot of log10(AF) per consequence class, pos vs neg. Below each CDF,
    a grouped bar plot of AF-bin fractions shows the quantitative shift.
    """
    if consequence_classes is None:
        consequence_classes = AF_SPECTRUM_CONSEQUENCES
 
    df = df.copy()
    df["consequence_class"] = df["Consequence"].apply(collapse_consequence)
    df = df[df[af_col].notna() & (df[af_col] > 0)]
    df["log10_af"] = np.log10(df[af_col])
 
    n_panels = len(consequence_classes)
    # Two rows: CDFs on top, AF-bin bars on bottom
    fig, axes = plt.subplots(
        2, n_panels,
        figsize=(n_panels * 3.0, 5.5),
        gridspec_kw={"height_ratios": [1.7, 1.0]},
        sharex=False,
    )
    if n_panels == 1:
        axes = axes.reshape(2, 1)
 
    all_stats = {}
 
    for col_idx, cclass in enumerate(consequence_classes):
        ax_cdf = axes[0, col_idx]
        ax_bar = axes[1, col_idx]
 
        sub = df[df["consequence_class"] == cclass]
        if len(sub) == 0:
            ax_cdf.set_visible(False)
            ax_bar.set_visible(False)
            continue
 
        stats_dict = _compute_af_spectrum_stats(df, cclass, af_col)
        all_stats[cclass] = stats_dict
 
        # ── CDF panel ──────────────────────────────────────────────────
        for group in ["neg", "pos"]:
            group_sub = sub[sub["group"] == group]
            if len(group_sub) < 10:
                continue
            sorted_vals = np.sort(group_sub["log10_af"].values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax_cdf.step(
                sorted_vals, cdf, where="post",
                color=GROUP_COLORS[group], linewidth=1.4,
                label=f"{group} (n={len(group_sub):,})",
            )
 
        # Median markers
        if pd.notna(stats_dict["median_pos_log10"]):
            ax_cdf.axvline(
                stats_dict["median_pos_log10"],
                color=GROUP_COLORS["pos"], linestyle="--",
                linewidth=0.6, alpha=0.5,
            )
        if pd.notna(stats_dict["median_neg_log10"]):
            ax_cdf.axvline(
                stats_dict["median_neg_log10"],
                color=GROUP_COLORS["neg"], linestyle="--",
                linewidth=0.6, alpha=0.5,
            )
 
        p_text = (
            f"p = {stats_dict['p']:.1e} {stats_dict['sig']}\n"
            f"median log₁₀AF:\n"
            f"  pos = {stats_dict['median_pos_log10']:.2f}\n"
            f"  neg = {stats_dict['median_neg_log10']:.2f}"
        )
        ax_cdf.text(
            0.02, 0.98, p_text,
            transform=ax_cdf.transAxes, fontsize=6.5,
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.9,
                      edgecolor="none", pad=2),
        )
 
        ax_cdf.set_title(cclass)
        ax_cdf.set_xlim(af_range)
        ax_cdf.set_ylim(0, 1.02)
        ax_cdf.set_xlabel("log₁₀(AF)")
        if col_idx == 0:
            ax_cdf.set_ylabel("Cumulative fraction")
        else:
            ax_cdf.set_ylabel("")
        ax_cdf.legend(loc="lower right", frameon=False, fontsize=7)
        ax_cdf.grid(alpha=0.3, linestyle=":", linewidth=0.4)
 
        # ── AF-bin side panel ──────────────────────────────────────────
        if stats_dict["bins_pos"] is not None:
            bin_df = pd.DataFrame({
                "neg": stats_dict["bins_neg"] / stats_dict["bins_neg"].sum()
                        if stats_dict["bins_neg"].sum() > 0 else stats_dict["bins_neg"],
                "pos": stats_dict["bins_pos"] / stats_dict["bins_pos"].sum()
                        if stats_dict["bins_pos"].sum() > 0 else stats_dict["bins_pos"],
            })
 
            x = np.arange(len(AF_BIN_LABELS))
            width = 0.38
            ax_bar.bar(
                x - width / 2, bin_df["neg"].values, width,
                color=GROUP_COLORS["neg"], edgecolor="black",
                linewidth=0.4, label="neg",
            )
            ax_bar.bar(
                x + width / 2, bin_df["pos"].values, width,
                color=GROUP_COLORS["pos"], edgecolor="black",
                linewidth=0.4, label="pos",
            )
 
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(AF_BIN_LABELS, fontsize=6)
            ax_bar.set_ylim(0, max(bin_df.max().max() * 1.2, 0.05))
            if col_idx == 0:
                ax_bar.set_ylabel("Fraction")
            else:
                ax_bar.set_ylabel("")
            ax_bar.tick_params(axis="y", labelsize=6)
            ax_bar.legend(loc="upper right", frameon=False, fontsize=6)
 
    sns.despine(fig=fig)
    fig.suptitle(
        f"AF spectrum — CDF and bin distribution ({dataset})",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
 
    if save:
        save_figure(fig, "af_spectrum_cdf", dataset=dataset)
 
    # Printed summary
    print(f"\n── AF spectrum (CDF) [{dataset}] ──")
    for cclass, s in all_stats.items():
        print(f"\n  {cclass}:")
        print(f"    n_pos = {s['n_pos']:,}, n_neg = {s['n_neg']:,}")
        print(f"    Median log10(AF): pos = {s['median_pos_log10']:.3f}, "
              f"neg = {s['median_neg_log10']:.3f}")
        print(f"    Mann-Whitney p = {s['p']:.2e} {s['sig']}")
        if s["bins_pos"] is not None:
            print(f"    AF bin fractions:")
            for bin_label in AF_BIN_LABELS:
                p_frac = s["bins_pos"][bin_label] / s["bins_pos"].sum() \
                          if s["bins_pos"].sum() > 0 else 0
                n_frac = s["bins_neg"][bin_label] / s["bins_neg"].sum() \
                          if s["bins_neg"].sum() > 0 else 0
                print(f"      {bin_label.replace(chr(10), ' '):20s}  "
                      f"pos={p_frac:.3f}  neg={n_frac:.3f}")
 
    return all_stats
 
# ════════════════════════════════════════════════════════════════════════════
# Subset definitions
# ════════════════════════════════════════════════════════════════════════════
 
AROMATIC_AAS = {"F", "Y", "W"}
CHARGED_AAS = {"R", "K", "D", "E"}
 
 
def _subset_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    """
    Boolean mask for the requested variant subset.
    """
    if subset == "all":
        return pd.Series(True, index=df.index)
    if subset == "rg_disrupting":
        # Requires is_rg_disrupting column (added by compute_rg_disruption_columns)
        if "is_rg_disrupting" not in df.columns:
            raise KeyError(
                "Column 'is_rg_disrupting' not found. Run "
                "compute_rg_disruption_columns first."
            )
        return df["is_rg_disrupting"].fillna(False)
    if subset == "aromatic_hit":
        return df["before_aa"].isin(AROMATIC_AAS)
    if subset == "charged_hit":
        return df["before_aa"].isin(CHARGED_AAS)
    raise ValueError(f"Unknown subset: {subset}")
 
 
SUBSET_DISPLAY_NAMES = {
    "all":            "All variants",
    "rg_disrupting":  "RG-disrupting",
    "aromatic_hit":   "Aromatic residue hit (F/Y/W)",
    "charged_hit":    "Charged residue hit (R/K/D/E)",
}
 
AF_BINS = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1.0]
AF_BIN_LABELS = [
    "< 1e-5",
    "1e-5\nto\n1e-4",
    "1e-4\nto\n1e-3",
    "1e-3\nto\n1e-2",
    "≥ 1e-2",
]
 
 
# ════════════════════════════════════════════════════════════════════════════
# Stats per subset
# ════════════════════════════════════════════════════════════════════════════
 
def _subset_af_stats(
    df: pd.DataFrame,
    subset: str,
    af_col: str,
) -> dict:
    """Mann-Whitney on pos vs neg AF within a given variant subset."""
    mask = _subset_mask(df, subset)
    sub = df[mask & df[af_col].notna() & (df[af_col] > 0)]
 
    pos_af = sub.loc[sub["group"] == "pos", af_col]
    neg_af = sub.loc[sub["group"] == "neg", af_col]
 
    if len(pos_af) < 10 or len(neg_af) < 10:
        return {
            "subset": subset,
            "n_pos": int(len(pos_af)), "n_neg": int(len(neg_af)),
            "median_pos_log10": np.nan, "median_neg_log10": np.nan,
            "u_stat": np.nan, "p": np.nan, "sig": "n.s.",
            "bins_pos": None, "bins_neg": None,
            "pos_af_vals": pos_af, "neg_af_vals": neg_af,
        }
 
    u, p = stats.mannwhitneyu(pos_af, neg_af, alternative="two-sided")
 
    bins_pos = pd.cut(pos_af, bins=AF_BINS, labels=AF_BIN_LABELS, right=False).value_counts()
    bins_neg = pd.cut(neg_af, bins=AF_BINS, labels=AF_BIN_LABELS, right=False).value_counts()
    bins_pos = bins_pos.reindex(AF_BIN_LABELS, fill_value=0)
    bins_neg = bins_neg.reindex(AF_BIN_LABELS, fill_value=0)
 
    return {
        "subset": subset,
        "n_pos": int(len(pos_af)),
        "n_neg": int(len(neg_af)),
        "median_pos_log10": float(np.log10(pos_af.median())),
        "median_neg_log10": float(np.log10(neg_af.median())),
        "max_pos_af": float(pos_af.max()),
        "max_neg_af": float(neg_af.max()),
        "u_stat": float(u),
        "p": float(p),
        "sig": significance_stars(p),
        "bins_pos": bins_pos,
        "bins_neg": bins_neg,
        "pos_af_vals": pos_af,
        "neg_af_vals": neg_af,
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# Plot
# ════════════════════════════════════════════════════════════════════════════
 
def plot_af_spectrum_by_subset(
    df: pd.DataFrame,
    af_col: str = "AF_joint",
    dataset: str = "gnomad",
    save: bool = True,
    subsets: list[str] = None,
    af_range: tuple[float, float] = (-7, 0),
) -> dict:
    """
    [gnomAD-specific]
    CDF + AF-bin panels for each variant subset, pos vs neg.
    """
    if subsets is None:
        subsets = ["all", "rg_disrupting", "aromatic_hit", "charged_hit"]
 
    df = df[df[af_col].notna() & (df[af_col] > 0)].copy()
    df["log10_af"] = np.log10(df[af_col])
 
    n_panels = len(subsets)
    fig, axes = plt.subplots(
        2, n_panels,
        figsize=(n_panels * 3.0, 5.5),
        gridspec_kw={"height_ratios": [1.7, 1.0]},
        sharex=False,
    )
    if n_panels == 1:
        axes = axes.reshape(2, 1)
 
    all_stats = {}
 
    for col_idx, subset in enumerate(subsets):
        ax_cdf = axes[0, col_idx]
        ax_bar = axes[1, col_idx]
 
        mask = _subset_mask(df, subset)
        sub = df[mask]
 
        if len(sub) == 0:
            ax_cdf.set_visible(False)
            ax_bar.set_visible(False)
            continue
 
        stats_dict = _subset_af_stats(df, subset, af_col)
        all_stats[subset] = stats_dict
 
        # ── CDF panel ──────────────────────────────────────────────────
        for group in ["neg", "pos"]:
            group_sub = sub[sub["group"] == group]
            if len(group_sub) < 10:
                continue
            sorted_vals = np.sort(group_sub["log10_af"].values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax_cdf.step(
                sorted_vals, cdf, where="post",
                color=GROUP_COLORS[group], linewidth=1.4,
                label=f"{group} (n={len(group_sub):,})",
            )
 
        if pd.notna(stats_dict["median_pos_log10"]):
            ax_cdf.axvline(
                stats_dict["median_pos_log10"],
                color=GROUP_COLORS["pos"], linestyle="--",
                linewidth=0.6, alpha=0.5,
            )
        if pd.notna(stats_dict["median_neg_log10"]):
            ax_cdf.axvline(
                stats_dict["median_neg_log10"],
                color=GROUP_COLORS["neg"], linestyle="--",
                linewidth=0.6, alpha=0.5,
            )
 
        # Stats text
        max_text = ""
        if "max_pos_af" in stats_dict and pd.notna(stats_dict.get("max_pos_af")):
            max_text = (f"\nmax AF:\n"
                        f"  pos = {stats_dict['max_pos_af']:.2e}\n"
                        f"  neg = {stats_dict['max_neg_af']:.2e}")
        p_text = (
            f"p = {stats_dict['p']:.1e} {stats_dict['sig']}\n"
            f"median log₁₀AF:\n"
            f"  pos = {stats_dict['median_pos_log10']:.2f}\n"
            f"  neg = {stats_dict['median_neg_log10']:.2f}"
            f"{max_text}"
        )
        ax_cdf.text(
            0.02, 0.98, p_text,
            transform=ax_cdf.transAxes, fontsize=6,
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.9,
                      edgecolor="none", pad=2),
        )
 
        ax_cdf.set_title(SUBSET_DISPLAY_NAMES.get(subset, subset), fontsize=9)
        ax_cdf.set_xlim(af_range)
        ax_cdf.set_ylim(0, 1.02)
        ax_cdf.set_xlabel("log₁₀(AF)")
        if col_idx == 0:
            ax_cdf.set_ylabel("Cumulative fraction")
        else:
            ax_cdf.set_ylabel("")
        ax_cdf.legend(loc="lower right", frameon=False, fontsize=7)
        ax_cdf.grid(alpha=0.3, linestyle=":", linewidth=0.4)
 
        # ── AF-bin bars ────────────────────────────────────────────────
        if stats_dict["bins_pos"] is not None:
            pos_total = stats_dict["bins_pos"].sum()
            neg_total = stats_dict["bins_neg"].sum()
            pos_frac = (stats_dict["bins_pos"] / pos_total) if pos_total > 0 else stats_dict["bins_pos"]
            neg_frac = (stats_dict["bins_neg"] / neg_total) if neg_total > 0 else stats_dict["bins_neg"]
 
            x = np.arange(len(AF_BIN_LABELS))
            width = 0.38
            ax_bar.bar(
                x - width / 2, neg_frac.values, width,
                color=GROUP_COLORS["neg"], edgecolor="black",
                linewidth=0.4, label="neg",
            )
            ax_bar.bar(
                x + width / 2, pos_frac.values, width,
                color=GROUP_COLORS["pos"], edgecolor="black",
                linewidth=0.4, label="pos",
            )
 
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(AF_BIN_LABELS, fontsize=6)
            ax_bar.set_ylim(0, max(max(pos_frac.max(), neg_frac.max()) * 1.2, 0.05))
            if col_idx == 0:
                ax_bar.set_ylabel("Fraction")
            else:
                ax_bar.set_ylabel("")
            ax_bar.tick_params(axis="y", labelsize=6)
            ax_bar.legend(loc="upper right", frameon=False, fontsize=6)
 
    sns.despine(fig=fig)
    fig.suptitle(
        f"AF spectrum by variant subset ({dataset}) — all consequence types",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
 
    if save:
        save_figure(fig, "af_spectrum_by_subset", dataset=dataset)
 
    # Printed summary
    print(f"\n── AF spectrum by subset ({dataset}) ──")
    for subset, s in all_stats.items():
        print(f"\n  {SUBSET_DISPLAY_NAMES.get(subset, subset)}:")
        print(f"    n_pos = {s['n_pos']:,}, n_neg = {s['n_neg']:,}")
        print(f"    Median log10(AF): pos = {s['median_pos_log10']:.3f}, "
              f"neg = {s['median_neg_log10']:.3f}")
        if "max_pos_af" in s and pd.notna(s.get("max_pos_af")):
            print(f"    Max AF: pos = {s['max_pos_af']:.2e}, "
                  f"neg = {s['max_neg_af']:.2e}")
        print(f"    Mann-Whitney p = {s['p']:.2e} {s['sig']}")
 
    return all_stats
 
"""
Save as: src/classifier_features/af_features.py

Per-region AF-derived features for the false-negative RG motif classifier.

Aggregates variant-level gnomAD AF data into region_id-level features spanning:
  - variant counts per consequence class
  - AF distribution shape (median, singleton/common fractions)
  - selection-proxy ratios (mis/syn, lof/syn, within-region MAPS-like deltas)
  - AlphaMissense × AF interactions
  - RG-disrupting subset features

Missing values (regions with too few variants for a given feature) are returned
as NaN. Imputation, if desired, is handled downstream by build_classifier_features
or by an sklearn Pipeline at fit time — this keeps the function consistent with
the rest of the per-region functions (compute_alphamissense_per_region, etc.)
which also return NaN for unobserved cases.

[gnomAD-specific] — requires AF_joint and AlphaMissense columns.
"""

# from __future__ import annotations
# import numpy as np
# import pandas as pd

from src.analysis_visualization.region_analysis import collapse_consequence


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

SINGLETON_AF_THRESHOLD = 1e-5
COMMON_AF_THRESHOLD    = 1e-3
RARE_AF_THRESHOLD      = 1e-4   # for MAPS-like delta and AM-rare counts
COUNT_PSEUDOCOUNT      = 0.5    # for ratio features (avoid div-by-zero, Haldane-style)

CONSEQUENCE_CLASSES = ["synonymous", "missense", "inframe_indel", "LoF"]
CONSEQUENCE_SHORT   = {
    "synonymous":    "syn",
    "missense":      "mis",
    "inframe_indel": "indel",
    "LoF":           "lof",
}

AM_LIKELY_PATHOGENIC_LABEL = "likely_pathogenic"


# ════════════════════════════════════════════════════════════════════════════
# Small helpers
# ════════════════════════════════════════════════════════════════════════════

def _safe_median_log10(s: pd.Series, min_n: int = 1) -> float:
    """Median of log10(AF) for a series of AF values. NaN if fewer than min_n
    positive values."""
    vals = s[s > 0]
    if len(vals) < min_n:
        return np.nan
    return float(np.log10(vals.median()))


def _safe_fraction(numerator_mask: pd.Series, denom_series: pd.Series) -> float:
    """Fraction of denom_series matching numerator_mask. NaN if denom is empty."""
    if len(denom_series) == 0:
        return np.nan
    return float(numerator_mask.sum() / len(denom_series))


def _log_ratio_with_pseudo(num: float, den: float, pseudo: float = COUNT_PSEUDOCOUNT) -> float:
    """log2((num + pseudo) / (den + pseudo)). Always defined."""
    return float(np.log2((num + pseudo) / (den + pseudo)))


# ════════════════════════════════════════════════════════════════════════════
# Per-region feature extractors (one function per feature family)
# ════════════════════════════════════════════════════════════════════════════

def _count_features(region_df: pd.DataFrame) -> dict:
    """Variant counts per consequence class. Always defined; zero is valid."""
    out = {}
    for cclass in CONSEQUENCE_CLASSES:
        short = CONSEQUENCE_SHORT[cclass]
        out[f"af_n_{short}"] = int((region_df["consequence_class"] == cclass).sum())
    return out


def _shape_features(region_df: pd.DataFrame, af_col: str) -> dict:
    """Median log10(AF), singleton fraction, common fraction per consequence."""
    out = {}
    for cclass in CONSEQUENCE_CLASSES:
        short = CONSEQUENCE_SHORT[cclass]
        sub = region_df[region_df["consequence_class"] == cclass]
        af = sub[af_col]
        af_pos = af[af > 0]

        out[f"af_median_log10_{short}"] = _safe_median_log10(af_pos)

        if len(af_pos) == 0:
            out[f"af_frac_singleton_{short}"] = np.nan
            out[f"af_frac_common_{short}"]    = np.nan
        else:
            out[f"af_frac_singleton_{short}"] = _safe_fraction(
                af_pos < SINGLETON_AF_THRESHOLD, af_pos
            )
            out[f"af_frac_common_{short}"] = _safe_fraction(
                af_pos >= COMMON_AF_THRESHOLD, af_pos
            )
    return out


def _selection_ratio_features(region_df: pd.DataFrame, af_col: str) -> dict:
    """mis/syn and lof/syn count ratios; median log-AF deltas; MAPS-like delta."""
    syn = region_df[region_df["consequence_class"] == "synonymous"]
    mis = region_df[region_df["consequence_class"] == "missense"]
    lof = region_df[region_df["consequence_class"] == "LoF"]

    out = {
        "af_log2ratio_mis_syn_count": _log_ratio_with_pseudo(len(mis), len(syn)),
        "af_log2ratio_lof_syn_count": _log_ratio_with_pseudo(len(lof), len(syn)),
    }

    syn_af = syn.loc[syn[af_col] > 0, af_col]
    mis_af = mis.loc[mis[af_col] > 0, af_col]

    # Median log10 AF delta (missense − synonymous). NaN if either side empty.
    if len(syn_af) >= 1 and len(mis_af) >= 1:
        out["af_median_log10_delta_mis_syn"] = float(
            np.log10(mis_af.median()) - np.log10(syn_af.median())
        )
    else:
        out["af_median_log10_delta_mis_syn"] = np.nan

    # MAPS-like delta: fraction-rare in missense minus fraction-rare in synonymous.
    # Positive = missense enriched in rare variants relative to synonymous
    # (i.e. consistent with purifying selection acting on missense).
    if len(syn_af) >= 1 and len(mis_af) >= 1:
        frac_rare_mis = float((mis_af < RARE_AF_THRESHOLD).sum() / len(mis_af))
        frac_rare_syn = float((syn_af < RARE_AF_THRESHOLD).sum() / len(syn_af))
        out["af_frac_rare_delta_mis_syn"] = frac_rare_mis - frac_rare_syn
    else:
        out["af_frac_rare_delta_mis_syn"] = np.nan

    return out


def _am_interaction_features(
    region_df: pd.DataFrame,
    af_col: str,
    am_score_col: str,
    am_class_col: str,
) -> dict:
    """AlphaMissense × AF interactions on missense variants only."""
    mis = region_df[
        (region_df["consequence_class"] == "missense") &
        region_df[am_score_col].notna() &
        (region_df[af_col] > 0)
    ]

    out = {}

    if len(mis) == 0:
        out["af_am_score_weighted_by_rarity"] = np.nan
    else:
        # Mean of AM_score × −log10(AF). Higher = deleterious-predicted
        # variants tend to be rare (consistent with purifying selection).
        weights = -np.log10(mis[af_col].values)
        out["af_am_score_weighted_by_rarity"] = float(
            (mis[am_score_col].values * weights).mean()
        )

    # Count features are always defined (zero is meaningful).
    rare_path = region_df[
        (region_df["consequence_class"] == "missense") &
        (region_df[am_class_col] == AM_LIKELY_PATHOGENIC_LABEL) &
        (region_df[af_col] > 0) &
        (region_df[af_col] < RARE_AF_THRESHOLD)
    ]
    out["af_n_likely_path_rare"] = int(len(rare_path))

    return out


def _rg_subset_features(region_df: pd.DataFrame, af_col: str) -> dict:
    """All-of-the-above logic restricted to is_rg_disrupting variants,
       plus within-region RG-vs-non-RG AF ratio (on missense)."""
    if "is_rg_disrupting" not in region_df.columns:
        raise KeyError(
            "Column 'is_rg_disrupting' not found. Run "
            "compute_rg_disruption_columns first."
        )

    rg_mask = region_df["is_rg_disrupting"].fillna(False)
    rg = region_df[rg_mask]
    rg_af = rg.loc[rg[af_col] > 0, af_col]

    out = {
        "af_rg_n_variants":     int(len(rg)),
        "af_rg_median_log10":   _safe_median_log10(rg_af),
        "af_rg_frac_singleton": (
            _safe_fraction(rg_af < SINGLETON_AF_THRESHOLD, rg_af)
            if len(rg_af) > 0 else np.nan
        ),
    }

    # Within-region log-AF delta between RG-disrupting and non-RG-disrupting
    # missense. Restricted to missense for interpretability.
    mis = region_df[region_df["consequence_class"] == "missense"]
    mis_rg     = mis.loc[mis["is_rg_disrupting"].fillna(False) & (mis[af_col] > 0), af_col]
    mis_non_rg = mis.loc[(~mis["is_rg_disrupting"].fillna(False)) & (mis[af_col] > 0), af_col]
    if len(mis_rg) >= 1 and len(mis_non_rg) >= 1:
        out["af_rg_vs_nonrg_log10_delta_mis"] = float(
            np.log10(mis_rg.median()) - np.log10(mis_non_rg.median())
        )
    else:
        out["af_rg_vs_nonrg_log10_delta_mis"] = np.nan

    return out


# ════════════════════════════════════════════════════════════════════════════
# Feature name registry
# ════════════════════════════════════════════════════════════════════════════

def get_feature_names() -> list[str]:
    """Canonical ordered list of feature column names."""
    names: list[str] = []

    # Counts
    for cclass in CONSEQUENCE_CLASSES:
        names.append(f"af_n_{CONSEQUENCE_SHORT[cclass]}")

    # Shape
    for cclass in CONSEQUENCE_CLASSES:
        s = CONSEQUENCE_SHORT[cclass]
        names += [f"af_median_log10_{s}", f"af_frac_singleton_{s}", f"af_frac_common_{s}"]

    # Selection ratios
    names += [
        "af_log2ratio_mis_syn_count",
        "af_log2ratio_lof_syn_count",
        "af_median_log10_delta_mis_syn",
        "af_frac_rare_delta_mis_syn",
    ]

    # AM interactions
    names += [
        "af_am_score_weighted_by_rarity",
        "af_n_likely_path_rare",
    ]

    # RG subset
    names += [
        "af_rg_n_variants",
        "af_rg_median_log10",
        "af_rg_frac_singleton",
        "af_rg_vs_nonrg_log10_delta_mis",
    ]

    return names


# ════════════════════════════════════════════════════════════════════════════
# Top-level entry point — matches the per-region function signature pattern
# used elsewhere in the pipeline (compute_alphamissense_per_region, etc.)
# ════════════════════════════════════════════════════════════════════════════

def compute_af_features_per_region(
    df: pd.DataFrame,
    af_col: str = "AF_joint",
    am_score_col: str = "am_pathogenicity",
    am_class_col: str = "am_class",
) -> pd.DataFrame:
    """
    Per-region AF-derived features for the RG-motif classifier.

    Returns one row per region with columns:
        ["region_id", "group", *get_feature_names()]

    Matches the shape of compute_alphamissense_per_region etc. so it can be
    merged into build_classifier_features with a single merge() call on
    ["region_id", "group"]. NaN values are left for downstream imputation.

    Parameters
    ----------
    df : variant-level DataFrame. Must have columns:
            "region_id", "group", af_col, am_score_col, am_class_col,
            "Consequence", "is_rg_disrupting".
    """
    df = df.copy()
    df["consequence_class"] = df["Consequence"].apply(collapse_consequence)
    # ── Consistency fix: a variant with no allele-frequency data is not
    # evidence for any feature, so drop NaN-AF rows ONCE here. This guarantees
    # the count features and the frequency-spectrum features describe the
    # SAME set of variants (previously counts included NaN-AF rows that the
    # spectra silently excluded).
    n_before = len(df)
    df = df[df[af_col].notna()]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [af_features] dropped {n_dropped} variants with missing {af_col} "
              f"({100*n_dropped/n_before:.1f}%) for feature-set consistency")
    feature_rows = []
    for (region_id, group), region_df in df.groupby(["region_id", "group"], sort=False):
        row = {"region_id": region_id, "group": group}
        row.update(_count_features(region_df))
        row.update(_shape_features(region_df, af_col))
        row.update(_selection_ratio_features(region_df, af_col))
        row.update(_am_interaction_features(region_df, af_col, am_score_col, am_class_col))
        row.update(_rg_subset_features(region_df, af_col))
        feature_rows.append(row)

    feat_df = pd.DataFrame(feature_rows)

    # Canonical column order: region_id, group, then features in registry order
    feat_df = feat_df[["region_id", "group"] + get_feature_names()]
    return feat_df