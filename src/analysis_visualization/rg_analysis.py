from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

from src.analysis_visualization.plot_config import (
    FIGSIZE_SINGLE, FIGSIZE_DOUBLE, GROUP_COLORS,
    save_figure, significance_stars,
)

from src.analysis_visualization.region_analysis import collapse_consequence


# ════════════════════════════════════════════════════════════════════════════
# Per-region RG descriptors (from WT sequence only)
# ════════════════════════════════════════════════════════════════════════════

def count_rg_positions(seq: str | None) -> list[int]:
    """Return all start positions of 'RG' motifs in a protein sequence (0-based)."""
    if seq is None or pd.isna(seq):
        return []
    return [m.start() for m in re.finditer("RG", seq)]

def compute_region_rg_stats(region_by_id: dict) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    For each region, compute:
        region_length    — residues in the region (from prot_seq)
        n_rg_motifs      — count of RG motifs in prot_seq
        rg_fraction      — fraction of residues that are part of an RG motif
                           (each RG covers 2 residues, hence 2*count/length)
        group            — from the JSON
    """
    rows = []
    for rid, r in region_by_id.items():
        seq = r["prot_seq"]
        if not seq or len(seq) < 2:
            continue
        n_rg = len(count_rg_positions(seq))
        rows.append({
            "region_id": rid,
            "group": r["group"],
            "region_length": len(seq),
            "n_rg_motifs": n_rg,
            "rg_fraction": (2 * n_rg) / len(seq),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Per-RG variant burden
# ════════════════════════════════════════════════════════════════════════════

VARIANT_TYPES = ["missense", "synonymous", "inframe_indel", "LoF"]


def compute_per_rg_burden(
    df_rg: pd.DataFrame,
    region_by_id: dict,
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    For each (region, RG motif, variant type), count how many variants hit
    the R or G of that motif.

    Returns one row per (region, RG motif) with columns for each variant type.
    """
    df = df_rg[df_rg["hits_rg"]].copy()
    df["variant_type"] = df["Consequence"].apply(collapse_consequence)

    # Keep only the variant types we care about
    df = df[df["variant_type"].isin(VARIANT_TYPES)]

    # Count hits per (region, rg_motif_pos, variant_type)
    counts = (
        df.groupby(["region_id", "rg_motif_pos", "variant_type"])
          .size()
          .unstack("variant_type", fill_value=0)
          .reset_index()
    )

    # Ensure all variant type columns exist
    for vt in VARIANT_TYPES:
        if vt not in counts.columns:
            counts[vt] = 0

    # Enumerate ALL RG motifs from region_by_id, including those with zero hits
    all_rgs = []
    for rid, r in region_by_id.items():
        for rg_pos in count_rg_positions(r["prot_seq"]):
            all_rgs.append({"region_id": rid, "rg_motif_pos": rg_pos,
                            "group": r["group"]})
    all_rgs_df = pd.DataFrame(all_rgs)

    merged = all_rgs_df.merge(
        counts, on=["region_id", "rg_motif_pos"], how="left"
    )
    for vt in VARIANT_TYPES:
        merged[vt] = merged[vt].fillna(0).astype(int)
    merged["total_hits"] = merged[VARIANT_TYPES].sum(axis=1)

    return merged


def compute_per_region_burden_stats(
    per_rg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    Aggregate per-RG burden to per-region stats needed for panels D and E.

    For each region × variant type:
        fraction_rgs_hit   — fraction of RGs in region hit by this variant type
        mean_burden_on_hit — mean variants per RG, among RGs that were hit
                             (NaN if no RGs were hit)
    """
    rows = []
    for (rid, group), sub in per_rg_df.groupby(["region_id", "group"]):
        n_rgs = len(sub)
        for vt in VARIANT_TYPES:
            hit_mask = sub[vt] > 0
            n_hit = int(hit_mask.sum())
            rows.append({
                "region_id": rid,
                "group": group,
                "variant_type": vt,
                "n_rgs": n_rgs,
                "n_rgs_hit": n_hit,
                "fraction_rgs_hit": n_hit / n_rgs if n_rgs > 0 else np.nan,
                "mean_burden_on_hit": (
                    sub.loc[hit_mask, vt].mean() if n_hit > 0 else np.nan
                ),
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Statistical testing helper — reused for every subplot
# ════════════════════════════════════════════════════════════════════════════

def _mw_test(pos_vals, neg_vals) -> tuple[float, str]:
    """Mann-Whitney U, returns (p-value, significance-stars)."""
    pos_vals = pd.Series(pos_vals).dropna()
    neg_vals = pd.Series(neg_vals).dropna()
    if len(pos_vals) < 3 or len(neg_vals) < 3:
        return (np.nan, "n.s.")
    _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    return (p, significance_stars(p))


def _boxplot(ax, data, x, y, pvalue_sig):
    """Small helper for consistent box+strip plot + significance annotation."""
    sns.boxplot(
        data=data, x=x, y=y, order=["neg", "pos"],
        palette=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
        width=0.5, fliersize=0, linewidth=0.6, ax=ax,
    )
    sns.stripplot(
        data=data, x=x, y=y, order=["neg", "pos"],
        color="black", size=1.2, alpha=0.4, jitter=0.15, ax=ax,
    )
    # Significance bar
    vals = data[y].dropna()
    if len(vals) > 0:
        ymax = vals.quantile(0.98)
        y_bar = ymax * 1.05 if ymax > 0 else 0.05
        ax.plot([0, 1], [y_bar, y_bar], color="black", lw=0.5)
        ax.text(0.5, y_bar * 1.02, pvalue_sig, ha="center", va="bottom", fontsize=7)
        ax.set_ylim(top=y_bar * 1.2)


# ════════════════════════════════════════════════════════════════════════════
# Small shared helpers
# ════════════════════════════════════════════════════════════════════════════
 
def _mw_test(pos_vals, neg_vals) -> tuple[float, str]:
    """Mann-Whitney U → (p-value, significance stars)."""
    pos_vals = pd.Series(pos_vals).dropna()
    neg_vals = pd.Series(neg_vals).dropna()
    if len(pos_vals) < 3 or len(neg_vals) < 3:
        return (np.nan, "n.s.")
    _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    return (p, significance_stars(p))
 
 
def _single_boxplot(data, x, y, order, title, ylabel,
                    dataset, filename, save, showmeans=True,
                    ylim_max=None):
    """
    Shared routine for the three "per-region single-metric" box plots
    (A, B, C). Returns (fig, results_dict).
    """
    pos_vals = data.loc[data[x] == "pos", y]
    neg_vals = data.loc[data[x] == "neg", y]
    p, sig = _mw_test(pos_vals, neg_vals)
 
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    sns.boxplot(
        data=data, x=x, y=y, order=order,
        palette=[GROUP_COLORS[g] for g in order],
        width=0.5, fliersize=0, linewidth=0.6, ax=ax,
        showmeans=showmeans,
        meanprops={
            "marker": "D",            # diamond
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
        },
    )
    sns.stripplot(
        data=data, x=x, y=y, order=order,
        color="black", size=1.5, alpha=0.4, jitter=0.15, ax=ax,
    )
 
    # Significance bar between boxes
    vals = data[y].dropna()
    if len(vals) > 0:
        ymax = vals.quantile(0.98)
        y_bar = ymax * 1.05 if ymax > 0 else 0.05
        ax.plot([0, 1], [y_bar, y_bar], color="black", lw=0.6)
        ax.text(0.5, y_bar * 1.02, sig, ha="center", va="bottom", fontsize=8)
        upper = ylim_max if ylim_max is not None else y_bar * 1.2
        ax.set_ylim(top=upper)
 
    ax.set_title(title)
    ax.set_ylabel(ylabel)
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
        save_figure(fig, filename, dataset=dataset)
 
    results = {
        "p": p, "sig": sig,
        "median_pos": float(pos_vals.median()),
        "median_neg": float(neg_vals.median()),
        "n_pos": int(len(pos_vals)),
        "n_neg": int(len(neg_vals)),
    }
 
    print(f"\n── {title} ({dataset}) ──")
    print(f"  pos median = {results['median_pos']:.4g}, neg median = {results['median_neg']:.4g}")
    print(f"  p = {p:.2e} {sig}  (n_pos = {results['n_pos']}, n_neg = {results['n_neg']})")
 
    return fig, results
 
 
def _grouped_boxplot_by_variant_type(
    data, value_col, variant_types,
    title, ylabel, dataset, filename, save, showmeans=True,
    ylim_bottom=None,
):
    """
    Grouped box plot for panels D and E: per-region metric across variant types,
    split by pos/neg. Returns (fig, pd.DataFrame of per-type stats).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
 
    sns.boxplot(
        data=data, x="variant_type", y=value_col, hue="group",
        order=variant_types, hue_order=["neg", "pos"],
        palette=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
        width=0.6, fliersize=0, linewidth=0.5, ax=ax,
        showmeans=showmeans,
        meanprops={
            "marker": "D",            # diamond
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
            },
        )
 
    # Per-type Mann-Whitney tests + significance annotation
    rows = []
    for vt in variant_types:
        sub = data[data["variant_type"] == vt]
        pos_vals = sub.loc[sub["group"] == "pos", value_col]
        neg_vals = sub.loc[sub["group"] == "neg", value_col]
        p, sig = _mw_test(pos_vals, neg_vals)
        rows.append({
            "variant_type": vt, "p": p, "sig": sig,
            "median_pos": pos_vals.median(),
            "median_neg": neg_vals.median(),
            "n_pos": len(pos_vals.dropna()),
            "n_neg": len(neg_vals.dropna()),
        })
    per_type_stats = pd.DataFrame(rows)
 
    # Stars above each x-category
    vals = data[value_col].dropna()
    if len(vals) > 0:
        ymax = vals.quantile(0.98)
        for i, row in per_type_stats.iterrows():
            ax.text(i, ymax * 1.05, row["sig"],
                    ha="center", va="bottom", fontsize=7)
        ax.set_ylim(top=ymax * 1.15)
 
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
 
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(title=None, frameon=False, loc="upper right")
 
    sns.despine()
    plt.tight_layout()
    if save:
        save_figure(fig, filename, dataset=dataset)
 
    print(f"\n── {title} ({dataset}) ──")
    print(per_type_stats.to_string(index=False))
 
    return fig, per_type_stats
 
 
# ════════════════════════════════════════════════════════════════════════════
# The five plots
# ════════════════════════════════════════════════════════════════════════════
 
def plot_region_length(region_by_id, dataset="gnomad", save=True):
    """A. Region length distribution per group."""
    region_stats = compute_region_rg_stats(region_by_id)
    return _single_boxplot(
        data=region_stats, x="group", y="region_length",
        order=["neg", "pos"],
        title="Region length", ylabel="Residues",
        dataset=dataset, filename="rg_region_length", save=save,
    )
 
 
def plot_n_rg_motifs(region_by_id, dataset="gnomad", save=True):
    """B. Raw count of RG motifs per region."""
    region_stats = compute_region_rg_stats(region_by_id)
    return _single_boxplot(
        data=region_stats, x="group", y="n_rg_motifs",
        order=["neg", "pos"],
        title="RG motifs per region", ylabel="Count",
        dataset=dataset, filename="rg_n_motifs", save=save,
    )
 
 
def plot_rg_density(region_by_id, dataset="gnomad", save=True):
    """C. Fraction of residues in RG motifs per region."""
    region_stats = compute_region_rg_stats(region_by_id)
    return _single_boxplot(
        data=region_stats, x="group", y="rg_fraction",
        order=["neg", "pos"],
        title="RG density", ylabel="Fraction of residues in RG",
        dataset=dataset, filename="rg_density", save=save,
    )
 
 
# def plot_fraction_rgs_hit(df_rg, region_by_id, dataset="gnomad", save=True):
#     """D. Fraction of RGs hit by each variant type, per region."""
#     per_rg = compute_per_rg_burden(df_rg, region_by_id)
#     per_region = compute_per_region_burden_stats(per_rg)
#     return _grouped_boxplot_by_variant_type(
#         data=per_region, value_col="fraction_rgs_hit",
#         variant_types=VARIANT_TYPES,
#         title="Fraction of RGs hit, by variant type (per region)",
#         ylabel="Fraction of RGs hit",
#         dataset=dataset, filename="rg_fraction_hit_by_type", save=save,
#     )
 
def plot_variants_per_rg_by_type(
    df_rg, region_by_id, dataset="gnomad", save=True,
):
    """
    D1 (replacement).
    For each region × variant type, compute:
        variants_per_rg = n_variants_of_type_on_RGs / n_rgs_in_region

    Not saturating (unlike fraction-of-RGs-hit); reflects density of hits.
    """
    per_rg = compute_per_rg_burden(df_rg, region_by_id)

    rows = []
    for (rid, group), sub in per_rg.groupby(["region_id", "group"]):
        n_rgs = len(sub)
        for vt in VARIANT_TYPES:
            total_hits = sub[vt].sum()
            rows.append({
                "region_id": rid,
                "group": group,
                "variant_type": vt,
                "variants_per_rg": total_hits / n_rgs if n_rgs > 0 else np.nan,
            })
    per_region = pd.DataFrame(rows)

    return _grouped_boxplot_by_variant_type(
        data=per_region, value_col="variants_per_rg",
        variant_types=VARIANT_TYPES,
        title="Variants per RG motif, by variant type (per region)",
        ylabel="Variants per RG",
        dataset=dataset, filename="rg_variants_per_rg_by_type", save=save,
    )


def plot_median_alphamissense_on_rgs(
    df_rg, dataset="gnomad", save=True,
):
    """
    D2.
    For each region, median AlphaMissense pathogenicity score across
    missense variants that hit an RG residue. Compares pos vs neg.

    [gnomAD-specific] — relies on am_pathogenicity.
    """
    # Restrict to missense variants that hit an RG
    mask = (
        df_rg["hits_rg"] &
        df_rg["Consequence"].fillna("").str.contains("missense_variant") &
        df_rg["am_pathogenicity"].notna()
    )
    sub = df_rg[mask]

    per_region = (
        sub.groupby(["region_id", "group"])["am_pathogenicity"]
           .median()
           .reset_index(name="median_am_on_rg")
    )

    return _single_boxplot(
        data=per_region, x="group", y="median_am_on_rg",
        order=["neg", "pos"],
        title="Median AlphaMissense (RG-hitting missense only)",
        ylabel="Median AlphaMissense pathogenicity",
        dataset=dataset, filename="rg_median_alphamissense_on_rg", save=save,
    )


# def plot_mean_burden_on_hit(df_rg, region_by_id, dataset="gnomad", save=True):
#     """E. Mean variants per hit RG, by variant type, per region."""
#     per_rg = compute_per_rg_burden(df_rg, region_by_id)
#     per_region = compute_per_region_burden_stats(per_rg)
#     return _grouped_boxplot_by_variant_type(
#         data=per_region, value_col="mean_burden_on_hit",
#         variant_types=VARIANT_TYPES,
#         title="Mean variants per hit RG, by variant type (per region)",
#         ylabel="Mean variants per hit RG",
#         dataset=dataset, filename="rg_mean_burden_on_hit", save=save,
#         ylim_bottom=1,
#     )
 

# ════════════════════════════════════════════════════════════════════════════
# Per-variant RG hit classification
# (adds hits_rg, is_rg_disrupting, rg_role, rg_motif_pos, aa_from, aa_to,
#  disruption_type, aa_transition columns to a variant dataframe)
# ════════════════════════════════════════════════════════════════════════════

def find_rg_motifs(seq: str) -> list[tuple[int, int]]:
    """Return (R_position, G_position) tuples for every RG in seq (0-based)."""
    if not isinstance(seq, str):
        return []
    return [(m.start(), m.start() + 1) for m in re.finditer("RG", seq)]


# VEP Consequence -> simplified category used for disruption classification
_VEP_TO_CATEGORY = {
    "synonymous_variant":       "silent",
    "missense_variant":         "missense",
    "stop_gained":              "nonsense",
    "stop_lost":                "missense",
    "start_lost":               "missense",
    "inframe_insertion":        "inframe_insertion",
    "inframe_deletion":         "inframe_deletion",
    "frameshift_variant":       "frameshift",
    "protein_altering_variant": "complex",
}


def _normalize_consequence(consequence):
    if consequence is None or pd.isna(consequence):
        return None
    for term in str(consequence).split("&"):
        if term in _VEP_TO_CATEGORY:
            return _VEP_TO_CATEGORY[term]
    return None


def _classify_rg_hit(region_seq, variant_pos_in_region, before_aa, after_aa, category):
    """
    Single-variant classifier. Returns a dict of RG hit / disruption attributes.
    """
    out = {
        "hits_rg": False, "is_rg_disrupting": False,
        "rg_role": None, "rg_motif_pos": None,
        "aa_from": None, "aa_to": None, "disruption_type": None,
    }
    if (variant_pos_in_region is None or region_seq is None
            or not isinstance(region_seq, str)
            or variant_pos_in_region < 0
            or variant_pos_in_region >= len(region_seq)):
        return out

    rgs = find_rg_motifs(region_seq)
    if not rgs:
        return out

    hit_pos, hit_role = None, None
    for r_pos, g_pos in rgs:
        if variant_pos_in_region == r_pos:
            hit_pos, hit_role = r_pos, "R"
            break
        if variant_pos_in_region == g_pos:
            hit_pos, hit_role = r_pos, "G"
            break

    if hit_pos is None:
        return out

    out["hits_rg"] = True
    out["rg_role"] = hit_role
    out["rg_motif_pos"] = hit_pos
    out["aa_from"] = "R" if hit_role == "R" else "G"

    if category == "silent":
        out["aa_to"] = out["aa_from"]
    elif category == "missense":
        out["aa_to"] = after_aa or None
        out["is_rg_disrupting"] = (after_aa != out["aa_from"])
        if out["is_rg_disrupting"]:
            out["disruption_type"] = "substitution"
    elif category == "nonsense":
        out["aa_to"] = "*"
        out["is_rg_disrupting"] = True
        out["disruption_type"] = "stop"
    elif category == "inframe_deletion":
        out["aa_to"] = "-"
        out["is_rg_disrupting"] = True
        out["disruption_type"] = "deletion"
    elif category == "inframe_insertion":
        out["aa_to"] = after_aa
        out["is_rg_disrupting"] = True
        out["disruption_type"] = "insertion"
    elif category == "frameshift":
        out["is_rg_disrupting"] = True
        out["disruption_type"] = "frameshift"
    else:
        out["aa_to"] = after_aa
        out["is_rg_disrupting"] = bool(after_aa and after_aa != out["aa_from"])
        if out["is_rg_disrupting"]:
            out["disruption_type"] = "complex"
    return out


def compute_rg_disruption_columns(df: pd.DataFrame, region_by_id: dict) -> pd.DataFrame:
    """
    Add RG hit / disruption columns to a variant dataframe.

    Expects: region_id, protein_position_int, region_start_aa,
             before_aa, after_aa, Consequence.
    """
    df = df.copy()

    def _row(row):
        cons = _normalize_consequence(row["Consequence"])
        region = region_by_id.get(row["region_id"])
        if region is None or pd.isna(row["protein_position_int"]):
            return _classify_rg_hit(None, None, None, None, cons)
        pos_in_region = int(row["protein_position_int"]) - int(row["region_start_aa"])
        return _classify_rg_hit(
            region_seq=region["prot_seq"],
            variant_pos_in_region=pos_in_region,
            before_aa=row["before_aa"],
            after_aa=row["after_aa"],
            category=cons,
        )

    results = df.apply(_row, axis=1)
    results_df = pd.DataFrame(list(results), index=df.index)
    results_df["aa_transition"] = results_df.apply(
        lambda r: f"{r['aa_from']}→{r['aa_to']}"
                   if r["hits_rg"] and r["aa_from"] and r["aa_to"] else None,
        axis=1,
    )
    return pd.concat([df, results_df], axis=1)



def plot_rg_role_asymmetry(
    df_rg: pd.DataFrame,
    dataset: str = "gnomad",
    save: bool = True,
) -> tuple[plt.Figure, dict]:
    """
    [Dataset-agnostic]
    For each region, compute the fraction of RG-disrupting missense variants
    that hit the R residue (vs the G). Compare pos vs neg.

    Scope: missense + RG-disrupting only, per earlier discussion. Synonymous
    and non-missense consequences are excluded.

    Per-region metric:
        r_fraction = n_R_hits / (n_R_hits + n_G_hits)
    Interpretation: 0.5 = balanced, >0.5 = R-biased, <0.5 = G-biased.
    Regions with zero disrupting missense hits contribute NaN and are dropped.
    """
    mask = (
        df_rg["is_rg_disrupting"] &
        df_rg["Consequence"].fillna("").str.contains("missense_variant") &
        df_rg["rg_role"].isin(["R", "G"])
    )
    sub = df_rg[mask]

    # Per-region counts
    per_region = (
        sub.groupby(["region_id", "group", "rg_role"])
           .size()
           .unstack("rg_role", fill_value=0)
           .reset_index()
    )
    # Ensure both R and G columns exist
    for col in ("R", "G"):
        if col not in per_region.columns:
            per_region[col] = 0

    per_region["n_total"] = per_region["R"] + per_region["G"]
    per_region = per_region[per_region["n_total"] > 0].copy()
    per_region["r_fraction"] = per_region["R"] / per_region["n_total"]

    # Mann-Whitney test
    pos_vals = per_region.loc[per_region["group"] == "pos", "r_fraction"]
    neg_vals = per_region.loc[per_region["group"] == "neg", "r_fraction"]
    _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    sig = significance_stars(p)

    # Aggregate counts (over all variants, not per region) — useful context
    total_r_pos = int(per_region.loc[per_region["group"] == "pos", "R"].sum())
    total_g_pos = int(per_region.loc[per_region["group"] == "pos", "G"].sum())
    total_r_neg = int(per_region.loc[per_region["group"] == "neg", "R"].sum())
    total_g_neg = int(per_region.loc[per_region["group"] == "neg", "G"].sum())
    agg_r_frac_pos = total_r_pos / (total_r_pos + total_g_pos)
    agg_r_frac_neg = total_r_neg / (total_r_neg + total_g_neg)

    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    sns.boxplot(
        data=per_region, x="group", y="r_fraction",
        order=["neg", "pos"],
        palette=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
        width=0.5, fliersize=0, linewidth=0.6, ax=ax,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 4},
    )
    sns.stripplot(
        data=per_region, x="group", y="r_fraction",
        order=["neg", "pos"], color="black", size=1.5,
        alpha=0.4, jitter=0.15, ax=ax,
    )

    # Reference line at 0.5 = balanced
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    # Significance bar
    y_bar = 1.05
    ax.plot([0, 1], [y_bar, y_bar], color="black", lw=0.6)
    ax.text(0.5, y_bar * 1.02, sig, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(-0.05, 1.2)

    ax.set_title("R vs G hit asymmetry (RG-disrupting missense)")
    ax.set_ylabel("Fraction of hits on R (R / (R + G))")
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
        save_figure(fig, "rg_r_vs_g_asymmetry", dataset=dataset)

    results = {
        "p": float(p), "sig": sig,
        "median_r_fraction_pos": float(pos_vals.median()),
        "median_r_fraction_neg": float(neg_vals.median()),
        "mean_r_fraction_pos": float(pos_vals.mean()),
        "mean_r_fraction_neg": float(neg_vals.mean()),
        "n_pos": int(len(pos_vals)),
        "n_neg": int(len(neg_vals)),
        "aggregate_r_frac_pos": float(agg_r_frac_pos),
        "aggregate_r_frac_neg": float(agg_r_frac_neg),
        "total_R_pos": total_r_pos, "total_G_pos": total_g_pos,
        "total_R_neg": total_r_neg, "total_G_neg": total_g_neg,
    }

    print(f"\n── R-vs-G hit asymmetry ({dataset}) ──")
    print(f"  Per-region R fraction:")
    print(f"    pos: median = {results['median_r_fraction_pos']:.3f}, "
          f"mean = {results['mean_r_fraction_pos']:.3f}, n = {results['n_pos']}")
    print(f"    neg: median = {results['median_r_fraction_neg']:.3f}, "
          f"mean = {results['mean_r_fraction_neg']:.3f}, n = {results['n_neg']}")
    print(f"    Mann-Whitney p = {p:.2e} {sig}")
    print(f"  Aggregate (all variants pooled):")
    print(f"    pos: R={total_r_pos}, G={total_g_pos}, R-fraction = {agg_r_frac_pos:.3f}")
    print(f"    neg: R={total_r_neg}, G={total_g_neg}, R-fraction = {agg_r_frac_neg:.3f}")

    return fig, results


def classify_rg_change_event(rg_before: list[int], rg_after: list[int]) -> str:
    """
    Classify what happened to RG motifs by comparing position lists.

    Returns one of:
        'no_change'  — rg_before == rg_after (same length, same positions)
        'gain'       — len(rg_after) > len(rg_before)
        'loss'       — len(rg_after) < len(rg_before)
        'movement'   — same length, different positions
    """
    if rg_before == rg_after:
        return "no_change"
    if len(rg_after) > len(rg_before):
        return "gain"
    if len(rg_after) < len(rg_before):
        return "loss"
    return "movement"  # same length, different positions


def compute_rg_change_events(
    df_rg: pd.DataFrame,
    region_by_id: dict,
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    For each missense variant, apply the single-residue substitution to the
    region sequence, recount RG motifs, and label the event type.

    Adds columns:
        rg_before_positions, rg_after_positions  (lists of 0-based R positions)
        n_rg_before, n_rg_after
        rg_change_event  ('no_change', 'gain', 'loss', 'movement')
    """
    df = df_rg[
        df_rg["Consequence"].fillna("").str.contains("missense_variant")
    ].copy()

    rg_before_list = []
    rg_after_list = []
    event_list = []

    for row in df.itertuples(index=False):
        region = region_by_id.get(row.region_id)
        if region is None or pd.isna(row.protein_position_int) or row.after_aa is None:
            rg_before_list.append(None)
            rg_after_list.append(None)
            event_list.append(None)
            continue

        seq_before = region["prot_seq"]
        pos_in_region = int(row.protein_position_int) - int(row.region_start_aa)

        if pos_in_region < 0 or pos_in_region >= len(seq_before):
            rg_before_list.append(None)
            rg_after_list.append(None)
            event_list.append(None)
            continue

        seq_after = seq_before[:pos_in_region] + row.after_aa + seq_before[pos_in_region + 1:]

        rg_before = count_rg_positions(seq_before)
        rg_after = count_rg_positions(seq_after)

        rg_before_list.append(rg_before)
        rg_after_list.append(rg_after)
        event_list.append(classify_rg_change_event(rg_before, rg_after))

    df["rg_before_positions"] = rg_before_list
    df["rg_after_positions"] = rg_after_list
    df["n_rg_before"] = [len(x) if x is not None else None for x in rg_before_list]
    df["n_rg_after"] = [len(x) if x is not None else None for x in rg_after_list]
    df["rg_change_event"] = event_list

    return df


RG_EVENT_TYPES = ["no_change", "loss", "gain", "movement"]


# def plot_rg_change_events(
#     df_rg: pd.DataFrame,
#     region_by_id: dict,
#     dataset: str = "gnomad",
#     save: bool = True,
# ) -> dict:
#     """
#     [Dataset-agnostic]
#     Four separate box plots: for each event type, fraction of the region's
#     missense variants that fall into that category. Compare pos vs neg.
#     """
#     df_events = compute_rg_change_events(df_rg, region_by_id)
#     df_events = df_events[df_events["rg_change_event"].notna()]

#     # Per-region: fraction of missense variants in each event category
#     total_per_region = df_events.groupby(["region_id", "group"]).size().reset_index(name="total")
#     counts_per_region = (
#         df_events.groupby(["region_id", "group", "rg_change_event"])
#                  .size()
#                  .reset_index(name="n")
#     )
#     merged = counts_per_region.merge(total_per_region, on=["region_id", "group"])
#     merged["fraction"] = merged["n"] / merged["total"]

#     # Fill in zero-count combinations so every region has every event category
#     all_regions = merged[["region_id", "group"]].drop_duplicates()
#     full_index = all_regions.assign(key=1).merge(
#         pd.DataFrame({"rg_change_event": RG_EVENT_TYPES}).assign(key=1),
#         on="key",
#     ).drop(columns="key")
#     full = full_index.merge(
#         merged[["region_id", "group", "rg_change_event", "fraction"]],
#         on=["region_id", "group", "rg_change_event"], how="left",
#     )
#     full["fraction"] = full["fraction"].fillna(0)

#     results = {}
#     for event in RG_EVENT_TYPES:
#         sub = full[full["rg_change_event"] == event]
#         fig, res = _single_boxplot(
#             data=sub, x="group", y="fraction", order=["neg", "pos"],
#             title=f"RG change event: {event}",
#             ylabel=f"Fraction of missense in region",
#             dataset=dataset,
#             filename=f"rg_change_event_{event}",
#             save=save,
#         )
#         results[event] = res

#     # Overall chi² on event distribution
#     contingency = pd.crosstab(df_events["group"], df_events["rg_change_event"])
#     chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)
#     print(f"\n── Overall χ² on event distribution ({dataset}) ──")
#     print(contingency)
#     print(f"  χ² = {chi2:.2f}, df = {dof}, p = {chi_p:.2e} "
#           f"{significance_stars(chi_p)}")

#     results["overall_chi2"] = {
#         "chi2": float(chi2), "dof": int(dof), "p": float(chi_p),
#         "sig": significance_stars(chi_p),
#         "contingency": contingency,
#     }
#     return results

def plot_rg_change_events_stacked(
    df_rg: pd.DataFrame,
    region_by_id: dict,
    dataset: str = "gnomad",
    save: bool = True,
) -> dict:
    """
    [Dataset-agnostic]
    Compact stacked bar summary of RG change event distribution, pos vs neg.
    Shows the overall proportion of each event category within each group,
    with the chi² test result for the whole distribution.
    """
    df_events = compute_rg_change_events(df_rg, region_by_id)
    df_events = df_events[df_events["rg_change_event"].notna()]

    # Aggregate counts + proportions per group
    contingency = pd.crosstab(df_events["group"], df_events["rg_change_event"])
    # Ensure consistent event order even if some categories are zero
    for ev in RG_EVENT_TYPES:
        if ev not in contingency.columns:
            contingency[ev] = 0
    contingency = contingency[RG_EVENT_TYPES]
    proportions = contingency.div(contingency.sum(axis=1), axis=0)

    # Overall chi²
    chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)
    sig = significance_stars(chi_p)

    # Plot
    fig, ax = plt.subplots(figsize=(FIGSIZE_SINGLE[0], FIGSIZE_SINGLE[1] + 0.3))

    # Colors for the 4 event categories — sequential scheme (cool → warm)

    event_colors = {
        "no_change": "#C7C7C7",   # neutral gray
        "loss":      "#D55E00",   # strong — loss is the most interesting
        "gain":      "#009E73",   # moderate
        "movement":  "#56B4E9",   # rare but notable
    }

    bottoms = np.zeros(len(proportions))
    groups = list(proportions.index)  # ['neg', 'pos'] alphabetically
    x = np.arange(len(groups))

    for event in RG_EVENT_TYPES:
        vals = proportions[event].values
        ax.bar(
            x, vals, bottom=bottoms,
            color=event_colors[event], edgecolor="black", linewidth=0.4,
            label=event, width=0.6,
        )
        # Annotate percentage inside each segment if > 3%
        for i, (val, bot) in enumerate(zip(vals, bottoms)):
            if val > 0.03:
                ax.text(
                    x[i], bot + val / 2, f"{val * 100:.1f}%",
                    ha="center", va="center", fontsize=6.5,
                    color="black" if event == "no_change" else "white",
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of missense variants")
    ax.set_xlabel("")
    ax.set_title("RG change event distribution")

    # Extend y-range slightly for the n labels above the bars
    ax.set_ylim(0, 1.12)

    # n per group labels — above each bar
    for i, g in enumerate(groups):
        n = int(contingency.loc[g].sum())
        ax.text(i, 1.02, f"n = {n:,}", ha="center", va="bottom", fontsize=7)

    # χ² annotation — sits in the legend column, above the legend itself
    ax.text(
        1.2, 0.92,
        f"χ² p = {chi_p:.1e} {sig}",
        transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray",
                pad=2, linewidth=0.4),
    )

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 0.85),
        frameon=False, title="Event",
    )
    sns.despine()
    plt.tight_layout()

    if save:
        save_figure(fig, "rg_change_events_stacked", dataset=dataset)

    print(f"\n── RG change event distribution ({dataset}) ──")
    print("Counts:")
    print(contingency)
    print("\nProportions:")
    print(proportions.round(4))
    print(f"\nχ² = {chi2:.2f}, df = {dof}, p = {chi_p:.2e} {sig}")

    return {
        "contingency": contingency,
        "proportions": proportions,
        "chi2": float(chi2),
        "chi2_p": float(chi_p),
        "chi2_sig": sig,
        "dof": int(dof),
    }

# ════════════════════════════════════════════════════════════════════════════
# Analysis 1: Loss transitions heatmap
# ════════════════════════════════════════════════════════════════════════════
 
def plot_rg_loss_transitions(
    df_events: pd.DataFrame,
    dataset: str = "gnomad",
    save: bool = True,
) -> dict:
    """
    [Dataset-agnostic]
    Heatmap of transitions (source → target) for missense variants classified
    as 'loss' events. Per-cell Fisher's exact comparing pos vs neg with
    Bonferroni correction.
 
    Expects df_events to have:
        rg_change_event (must include 'loss')
        rg_role          ('R' or 'G')
        aa_from, aa_to   (single amino acid letters)
        group            ('pos' or 'neg')
    """
    # Restrict to loss events where we know R/G was hit and what replaced it
    sub = df_events[
        (df_events["rg_change_event"] == "loss") &
        df_events["rg_role"].isin(["R", "G"]) &
        df_events["aa_to"].notna() &
        (df_events["aa_to"].str.len() == 1) &
        (df_events["aa_to"] != "*")
    ].copy()
 
    all_aa = list("ACDEFGHIKLMNPQRSTVWY")
    sources = ["R", "G"]
 
    # Build 4-row matrix: R→? pos, R→? neg, G→? pos, G→? neg
    row_labels = []
    counts_rows = []
    enrichment_rows = []
    for source in sources:
        for group in ["pos", "neg"]:
            row_labels.append(f"{source}→? ({group})")
            group_sub = sub[(sub["aa_from"] == source) & (sub["group"] == group)]
            counts = group_sub["aa_to"].value_counts().reindex(all_aa, fill_value=0)
            counts_rows.append(counts.values)
            # Enrichment vs uniform expectation over the 19 non-source targets
            non_source_total = counts.drop(source, errors="ignore").sum()
            expected = non_source_total / 19 if non_source_total > 0 else 0
            enrichment = np.where(
                expected > 0,
                counts.values / expected,
                np.nan,
            )
            enrichment_rows.append(enrichment)
 
    counts_matrix = np.vstack(counts_rows).astype(int)
    enrichment_matrix = np.vstack(enrichment_rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        # pseudo = 1e-6
        # log2_enrichment = np.log2(enrichment_matrix + pseudo)
        log2_enrichment = np.log2(enrichment_matrix)
        log2_enrichment = np.nan_to_num(log2_enrichment, neginf=np.nan)
 
    # Per-cell Fisher's exact pos-vs-neg for each (source, target)
    sig_matrix = np.full_like(counts_matrix, "", dtype=object)
    p_records = []
    n_tests = 0
    for s_idx, source in enumerate(sources):
        pos_row = counts_matrix[s_idx * 2]
        neg_row = counts_matrix[s_idx * 2 + 1]
        pos_total = pos_row.sum()
        neg_total = neg_row.sum()
        for t_idx, target in enumerate(all_aa):
            if target == source:
                continue
            pos_in = pos_row[t_idx]
            neg_in = neg_row[t_idx]
            if pos_in + neg_in < 3:
                continue  # skip too-rare cells
            pos_out = pos_total - pos_in
            neg_out = neg_total - neg_in
            _, p_raw = stats.fisher_exact(
                [[pos_in, pos_out], [neg_in, neg_out]]
            )
            p_records.append({
                "source": source, "target": target,
                "pos_count": int(pos_in), "neg_count": int(neg_in),
                "p_raw": float(p_raw),
            })
            n_tests += 1
 
    # Bonferroni correction
    p_df = pd.DataFrame(p_records)
    if len(p_df) > 0:
        p_df["p_bonf"] = (p_df["p_raw"] * n_tests).clip(upper=1.0)
        p_df["sig"] = p_df["p_bonf"].apply(significance_stars)
        # Mark significant cells on BOTH pos and neg rows for each (source, target)
        for _, r in p_df.iterrows():
            if r["sig"] == "n.s.":
                continue
            s_idx = sources.index(r["source"])
            t_idx = all_aa.index(r["target"])
            sig_matrix[s_idx * 2, t_idx] = r["sig"]
            sig_matrix[s_idx * 2 + 1, t_idx] = r["sig"]
 
    # Build annotation matrix: count + significance stars
    annot = np.empty_like(counts_matrix, dtype=object)
    for i in range(counts_matrix.shape[0]):
        for j in range(counts_matrix.shape[1]):
            count = counts_matrix[i, j]
            stars = sig_matrix[i, j]
            annot[i, j] = f"{count}\n{stars}" if stars else f"{count}"
 
    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    vmax = np.nanmax(np.abs(log2_enrichment))
    mask = np.isnan(log2_enrichment)
    sns.heatmap(
        log2_enrichment,
        xticklabels=all_aa, yticklabels=row_labels,
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        annot=annot, fmt="", annot_kws={"size": 6},
        cbar_kws={"label": "log₂(observed / uniform expected)", "shrink": 0.7},
        linewidths=0.3, linecolor="white", ax=ax,
    )
    ax.set_xlabel("Target amino acid")
    ax.set_ylabel("")
    ax.set_title(f"RG-loss transitions ({dataset}, missense only; * = Bonferroni p<0.05)")
 
    plt.tight_layout()
    if save:
        save_figure(fig, "rg_loss_transitions", dataset=dataset)
 
    print(f"\n── RG-loss transitions ({dataset}) ──")
    print(f"  Total loss events: pos = {counts_matrix[::2].sum()}, "
          f"neg = {counts_matrix[1::2].sum()}")
    print(f"  Fisher tests performed: {n_tests}")
    if len(p_df) > 0:
        sig_hits = p_df[p_df["sig"] != "n.s."]
        # sig_hits = p_df
        if len(sig_hits) > 0:
            print(f"  Significant transitions (Bonferroni):")
            print(sig_hits.to_string(index=False))
        else:
            print(f"  No transitions significant after Bonferroni correction.")
 
    return {
        "counts_matrix": counts_matrix,
        "log2_enrichment": log2_enrichment,
        "fisher_results": p_df,
        "n_tests": n_tests,
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# Analysis 2: Isolated vs clustered RG losses
# ════════════════════════════════════════════════════════════════════════════
 
def _classify_rg_cluster_status(
    rg_positions: list[int],
    target_rg_pos: int,
    window: int,
) -> str:
    """
    For a given RG motif at target_rg_pos in a sequence, decide if it's
    'isolated' (no other RG within ±window) or 'clustered' (at least one
    other RG within ±window).
 
    Distance is measured between RG start positions (i.e., the R position).
    The 2-residue motif length is implicit; the window is inclusive.
    """
    for other_pos in rg_positions:
        if other_pos == target_rg_pos:
            continue
        if abs(other_pos - target_rg_pos) <= window:
            return "clustered"
    return "isolated"
 
 
def plot_isolated_vs_clustered_loss(
    df_events: pd.DataFrame,
    region_by_id: dict,
    window_sizes: list[int] = [3, 5, 10],
    dataset: str = "gnomad",
    save: bool = True,
) -> dict:
    """
    [Dataset-agnostic]
    For each window size, classify each 'loss' event by whether the disrupted
    RG was isolated or clustered (≥1 other RG within ±window residues).
 
    Plots one subplot per window — grouped bar plot showing fraction of losses
    that hit isolated vs clustered RGs, for pos and neg.
 
    Per-window statistics: Fisher's exact on the 2x2 table
    (isolated/clustered × pos/neg).
    """
    # Keep only loss events where we know which RG was hit
    sub = df_events[
        (df_events["rg_change_event"] == "loss") &
        df_events["rg_motif_pos"].notna()
    ].copy()
 
    # Precompute RG positions per region (from WT sequence) — used for cluster status
    from src.analysis_visualization.rg_analysis import count_rg_positions
    region_rg_positions = {
        rid: count_rg_positions(r["prot_seq"]) for rid, r in region_by_id.items()
    }
 
    # For each loss event, classify the disrupted RG at each window size
    results_per_window = {}
    for window in window_sizes:
        statuses = []
        for row in sub.itertuples(index=False):
            rg_positions = region_rg_positions.get(row.region_id, [])
            if not rg_positions:
                statuses.append(None)
                continue
            status = _classify_rg_cluster_status(
                rg_positions, int(row.rg_motif_pos), window
            )
            statuses.append(status)
        col = f"cluster_status_w{window}"
        sub[col] = statuses
        results_per_window[window] = col
 
    # Build counts per (group, status) for each window
    fig, axes = plt.subplots(
        1, len(window_sizes),
        figsize=(FIGSIZE_SINGLE[0] * len(window_sizes), FIGSIZE_SINGLE[1]),
        sharey=True,
    )
    if len(window_sizes) == 1:
        axes = [axes]
 
    all_stats = {}
 
    for ax, window in zip(axes, window_sizes):
        col = results_per_window[window]
        valid = sub[sub[col].notna()]
        contingency = pd.crosstab(valid["group"], valid[col]).reindex(
            index=["neg", "pos"], columns=["isolated", "clustered"], fill_value=0
        )
 
        # Convert to fractions per group for the bars
        fractions = contingency.div(contingency.sum(axis=1), axis=0)
 
        # Fisher's exact on the 2x2 table
        if contingency.shape == (2, 2) and contingency.values.sum() > 0:
            odds, p = stats.fisher_exact(contingency.values)
            sig = significance_stars(p)
        else:
            odds, p, sig = np.nan, np.nan, "n.s."
 
        all_stats[window] = {
            "contingency": contingency,
            "odds_ratio": float(odds) if not np.isnan(odds) else np.nan,
            "p": float(p) if not np.isnan(p) else np.nan,
            "sig": sig,
        }
 
        # Grouped bar plot: x = cluster status, hue = group
        plot_df = fractions.reset_index().melt(
            id_vars="group", value_name="fraction", var_name="cluster_status"
        )
        plot_df["cluster_status"] = pd.Categorical(
            plot_df["cluster_status"], categories=["isolated", "clustered"], ordered=True
        )
        sns.barplot(
            data=plot_df, x="cluster_status", y="fraction", hue="group",
            order=["isolated", "clustered"], hue_order=["neg", "pos"],
            palette=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
            edgecolor="black", linewidth=0.4, ax=ax,
        )
 
        # Title + p-value annotation
        ax.set_title(f"window = ±{window}")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.1)
 
        ax.text(
            0.5, 0.98,
            f"Fisher p = {p:.2e} {sig}\nOR = {odds:.2f}",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="center",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2),
        )
 
        # Sample sizes annotation
        for i, status in enumerate(["isolated", "clustered"]):
            n_neg = int(contingency.loc["neg", status])
            n_pos = int(contingency.loc["pos", status])
            ax.text(
                i, 1.03,
                f"n_neg={n_neg}\nn_pos={n_pos}",
                ha="center", va="bottom", fontsize=5.5,
            )
 
        if ax is axes[0]:
            ax.set_ylabel("Fraction of losses")
        else:
            ax.set_ylabel("")
            ax.get_legend().remove() if ax.get_legend() else None
 
        # Keep legend only on the last subplot
        if ax is not axes[-1] and ax.get_legend():
            ax.get_legend().remove()
 
    # Final legend on the last axis
    axes[-1].legend(title=None, frameon=False, loc="upper right")
 
    fig.suptitle(f"RG losses in isolated vs clustered contexts ({dataset})", y=1.02)
    sns.despine(fig=fig)
    plt.tight_layout()
 
    if save:
        save_figure(fig, "rg_isolated_vs_clustered_loss", dataset=dataset)
 
    print(f"\n── Isolated vs clustered RG losses ({dataset}) ──")
    for window, r in all_stats.items():
        print(f"\n  Window ±{window}:")
        print(r["contingency"])
        print(f"  Odds ratio = {r['odds_ratio']:.2f}, Fisher p = {r['p']:.2e} {r['sig']}")
 
    return all_stats
 

def plot_rg_gain_transitions(
    df_events: pd.DataFrame,
    dataset: str = "gnomad",
    save: bool = True,
) -> dict:
    """
    [Dataset-agnostic]
    Heatmap of transitions (source → target) for missense variants classified
    as 'gain' events (i.e., mutations that create R or G).

    Expects df_events to have:
        rg_change_event (must include 'gain')
        rg_role          ('R' or 'G')  # the residue gained
        aa_from, aa_to   (single amino acid letters)
        group            ('pos' or 'neg')
    """

    # Restrict to gain events where R/G is the TARGET
    sub = df_events[
        (df_events["rg_change_event"] == "gain") &
        # df_events["rg_role"].isin(["R", "G"]) &
        df_events["before_aa"].notna() &
        (df_events["before_aa"].str.len() == 1) &
        (df_events["before_aa"] != "*")
    ].copy()
    # print(sub)

    all_aa = list("ACDEFGHIKLMNPQRSTVWY")
    targets = ["R", "G"]

    # Build 4-row matrix: ?→R pos, ?→R neg, ?→G pos, ?→G neg
    row_labels = []
    counts_rows = []
    enrichment_rows = []

    for target in targets:
        for group in ["pos", "neg"]:
            row_labels.append(f"?→{target} ({group})")

            group_sub = sub[
                (sub["after_aa"] == target) &
                (sub["group"] == group)
            ]

            counts = group_sub["before_aa"].value_counts().reindex(all_aa, fill_value=0)
            print(counts)
            counts_rows.append(counts.values)

            # Enrichment vs uniform expectation over 19 non-target residues
            non_target_total = counts.drop(target, errors="ignore").sum()
            print(non_target_total)
            expected = non_target_total / 19 if non_target_total > 0 else 0
            print(expected)

            enrichment = np.where(
                expected > 0,
                counts.values / expected,
                np.nan,
            )
            # print(enrichment)
            print("____")
            enrichment_rows.append(enrichment)
    # print(enrichment_rows)
    counts_matrix = np.vstack(counts_rows).astype(int)
    enrichment_matrix = np.vstack(enrichment_rows)

    with np.errstate(divide="ignore", invalid="ignore"):
        log2_enrichment = np.log2(enrichment_matrix)
        log2_enrichment = np.nan_to_num(log2_enrichment, neginf=np.nan)
        # pseudo = 1e-6
        # log2_enrichment = np.log2(enrichment_matrix + pseudo)
    # print(log2_enrichment)
    # Per-cell Fisher's exact (pos vs neg) for each (source → target)
    sig_matrix = np.full_like(counts_matrix, "", dtype=object)
    p_records = []
    n_tests = 0

    for t_idx, target in enumerate(targets):
        pos_row = counts_matrix[t_idx * 2]
        neg_row = counts_matrix[t_idx * 2 + 1]

        pos_total = pos_row.sum()
        neg_total = neg_row.sum()

        for s_idx, source in enumerate(all_aa):
            if source == target:
                continue

            pos_in = pos_row[s_idx]
            neg_in = neg_row[s_idx]

            if pos_in + neg_in < 3:
                continue

            pos_out = pos_total - pos_in
            neg_out = neg_total - neg_in

            _, p_raw = stats.fisher_exact(
                [[pos_in, pos_out], [neg_in, neg_out]]
            )

            p_records.append({
                "source": source,
                "target": target,
                "pos_count": int(pos_in),
                "neg_count": int(neg_in),
                "p_raw": float(p_raw),
            })
            n_tests += 1

    # Bonferroni correction
    p_df = pd.DataFrame(p_records)
    n_tests = 1
    if len(p_df) > 0:
        p_df["p_bonf"] = (p_df["p_raw"] * n_tests).clip(upper=1.0)
        p_df["sig"] = p_df["p_bonf"].apply(significance_stars)

        for _, r in p_df.iterrows():
            if r["sig"] == "n.s.":
                continue
            t_idx = targets.index(r["target"])
            s_idx = all_aa.index(r["source"])

            sig_matrix[t_idx * 2, s_idx] = r["sig"]
            sig_matrix[t_idx * 2 + 1, s_idx] = r["sig"]

    # Annotation matrix
    annot = np.empty_like(counts_matrix, dtype=object)
    for i in range(counts_matrix.shape[0]):
        for j in range(counts_matrix.shape[1]):
            count = counts_matrix[i, j]
            stars = sig_matrix[i, j]
            annot[i, j] = f"{count}\n{stars}" if stars else f"{count}"

    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    vmax = np.nanmax(np.abs(log2_enrichment))
    mask = np.isnan(log2_enrichment)
    sns.heatmap(
        log2_enrichment,
        mask=mask,
        xticklabels=all_aa, yticklabels=row_labels,
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        annot=annot, fmt="", annot_kws={"size": 6},
        cbar_kws={"label": "log₂(observed / uniform expected)", "shrink": 0.7},
        linewidths=0.3, linecolor="white", ax=ax,
    )

    ax.set_xlabel("Source amino acid")
    ax.set_ylabel("")
    ax.set_title(f"RG-gain transitions ({dataset}, missense only; * = Bonferroni p<0.05)")

    plt.tight_layout()
    if save:
        save_figure(fig, "rg_gain_transitions", dataset=dataset)

    print(f"\n── RG-gain transitions ({dataset}) ──")
    print(f"  Total gain events: pos = {counts_matrix[::2].sum()}, "
          f"neg = {counts_matrix[1::2].sum()}")
    print(f"  Fisher tests performed: {n_tests}")

    if len(p_df) > 0:
        sig_hits = p_df
        if len(sig_hits) > 0:
            print(f"  Significant transitions (Bonferroni):")
            print(sig_hits.to_string(index=False))
        else:
            print(f"  No transitions significant after Bonferroni correction.")

    return {
        "counts_matrix": counts_matrix,
        "log2_enrichment": log2_enrichment,
        "fisher_results": p_df,
        "n_tests": n_tests,
    }


_CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K',
    'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W',
    'CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
 
 
def _translate(dna: str) -> str:
    """Translate DNA to protein (length must be a multiple of 3)."""
    return "".join(
        _CODON_TABLE.get(dna[i:i + 3].upper(), "X")
        for i in range(0, len(dna) - 2, 3)
    )
 
 
# ════════════════════════════════════════════════════════════════════════════
# Per-region enumeration of all possible single-nt substitutions
# ════════════════════════════════════════════════════════════════════════════
 
_NUCLEOTIDES = ["A", "C", "G", "T"]
 
 
def enumerate_single_nt_substitutions(
    dna: str,
    protein: str,
) -> pd.DataFrame:
    """
    For a given DNA sequence (coding, must be multiple of 3), enumerate every
    possible single-nucleotide substitution. For each one, compute:
 
      - consequence: 'synonymous', 'missense', 'nonsense', 'other'
      - rg_event:    'no_change', 'loss', 'gain', 'movement'  (for missense only)
      - aa_from, aa_to: for missense only
 
    The frameshift and in-frame indel categories do not occur — we only
    substitute single bases.
 
    Returns one row per (dna_position × alternative_base) combination.
    """
    dna = dna.upper()
    n = len(dna)
 
    # Precompute RG positions in reference protein
    ref_rg = _rg_positions(protein)
 
    rows = []
    for i in range(n):
        ref_base = dna[i]
        for alt_base in _NUCLEOTIDES:
            if alt_base == ref_base:
                continue
            mutated_dna = dna[:i] + alt_base + dna[i + 1:]
            mutated_protein = _translate(mutated_dna)
 
            # Position in protein affected by this nt substitution
            aa_pos = i // 3
            if aa_pos >= len(protein) or aa_pos >= len(mutated_protein):
                continue
 
            ref_aa = protein[aa_pos]
            alt_aa = mutated_protein[aa_pos]
 
            # Consequence classification
            if alt_aa == ref_aa:
                cons = "synonymous"
            elif alt_aa == "*":
                cons = "nonsense"
            elif ref_aa == "*":
                cons = "stop_lost"
            else:
                cons = "missense"
 
            # RG event (only meaningful for missense)
            if cons == "missense":
                alt_rg = _rg_positions(mutated_protein)
                rg_event = _classify_rg_event(ref_rg, alt_rg)
            else:
                rg_event = None
 
            rows.append({
                "dna_pos": i,
                "ref_base": ref_base,
                "alt_base": alt_base,
                "aa_pos": aa_pos,
                "aa_from": ref_aa,
                "aa_to": alt_aa,
                "consequence": cons,
                "rg_event": rg_event,
            })
 
    return pd.DataFrame(rows)
 
 
def _rg_positions(seq: str) -> list[int]:
    """R positions of RG motifs."""
    return [i for i in range(len(seq) - 1) if seq[i] == "R" and seq[i + 1] == "G"]
 
 
def _classify_rg_event(rg_before: list[int], rg_after: list[int]) -> str:
    """Same classifier used for observed events — duplicated to keep modules independent."""
    if rg_before == rg_after:
        return "no_change"
    if len(rg_after) > len(rg_before):
        return "gain"
    if len(rg_after) < len(rg_before):
        return "loss"
    return "movement"
 
 
# ════════════════════════════════════════════════════════════════════════════
# Build null distributions across regions
# ════════════════════════════════════════════════════════════════════════════
 
def build_enumeration_null(
    region_by_id: dict,
    df_observed: pd.DataFrame,
) -> dict:
    """
    For each region, enumerate all possible single-nt substitutions and
    aggregate into per-group null counts, weighted by the observed missense
    count per region (so regions with more observed variants contribute more).
 
    Returns a dict with:
        rg_events_null      — DataFrame of expected counts per (group, rg_event)
        consequences_null   — DataFrame of expected counts per (group, consequence)
        per_region_enumerations — dict[region_id] -> enumeration DataFrame
    """
    # Observed missense count per region (used for weighting the null)
    obs_missense_per_region = (
        df_observed[
            df_observed["Consequence"].fillna("").str.contains("missense_variant")
        ]
        .groupby("region_id")
        .size()
    )
 
    # Observed total variants per region (for weighting the consequence null)
    obs_total_per_region = df_observed.groupby("region_id").size()
 
    rg_rows = []
    cons_rows = []
    per_region_enum = {}
 
    for rid, region in region_by_id.items():
        dna = region.get("dna")
        protein = region.get("prot_seq")
        group = region.get("group")
        if not dna or not protein or len(dna) != 3 * len(protein):
            continue
 
        enum_df = enumerate_single_nt_substitutions(dna, protein)
        per_region_enum[rid] = enum_df
 
        # ── RG-event null: weighted by this region's observed missense count ──
        missense_enum = enum_df[enum_df["consequence"] == "missense"]
        n_possible_missense = len(missense_enum)
        n_obs_missense = int(obs_missense_per_region.get(rid, 0))
        if n_possible_missense > 0 and n_obs_missense > 0:
            event_props = (
                missense_enum["rg_event"].value_counts(normalize=True)
            )
            for event, prop in event_props.items():
                rg_rows.append({
                    "region_id": rid,
                    "group": group,
                    "rg_event": event,
                    "expected_count": prop * n_obs_missense,
                })
 
        # ── Consequence null: weighted by this region's observed total ──
        n_possible = len(enum_df)
        n_obs_total = int(obs_total_per_region.get(rid, 0))
        if n_possible > 0 and n_obs_total > 0:
            cons_props = enum_df["consequence"].value_counts(normalize=True)
            for cons, prop in cons_props.items():
                cons_rows.append({
                    "region_id": rid,
                    "group": group,
                    "consequence": cons,
                    "expected_count": prop * n_obs_total,
                })
 
    rg_events_null = (
        pd.DataFrame(rg_rows)
          .groupby(["group", "rg_event"])["expected_count"]
          .sum()
          .reset_index()
    )
    consequences_null = (
        pd.DataFrame(cons_rows)
          .groupby(["group", "consequence"])["expected_count"]
          .sum()
          .reset_index()
    )
 
    return {
        "rg_events_null": rg_events_null,
        "consequences_null": consequences_null,
        "per_region_enumerations": per_region_enum,
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# Plot: observed vs expected RG-event distribution
# ════════════════════════════════════════════════════════════════════════════
 
RG_EVENT_TYPES_ORDERED = ["no_change", "loss", "gain", "movement"]
 
_EVENT_COLORS = {
    "no_change": "#C7C7C7",
    "loss":      "#D55E00",
    "gain":      "#009E73",
    "movement":  "#56B4E9",
}
 
 
def plot_rg_events_observed_vs_expected(
    df_events: pd.DataFrame,
    null_results: dict,
    dataset: str = "gnomad",
    save: bool = True,
) -> dict:
    """
    Stacked-bar plot with FOUR bars: neg observed, neg expected, pos observed,
    pos expected. Chi² goodness-of-fit test per group (observed vs expected).
    """
    df_events = df_events[df_events["rg_change_event"].notna()].copy()
 
    # Observed counts
    obs = pd.crosstab(df_events["group"], df_events["rg_change_event"])
    for ev in RG_EVENT_TYPES_ORDERED:
        if ev not in obs.columns:
            obs[ev] = 0
    obs = obs[RG_EVENT_TYPES_ORDERED]
 
    # Expected counts
    null_df = null_results["rg_events_null"]
    exp = null_df.pivot(index="group", columns="rg_event", values="expected_count")
    for ev in RG_EVENT_TYPES_ORDERED:
        if ev not in exp.columns:
            exp[ev] = 0
    exp = exp.reindex(index=obs.index)[RG_EVENT_TYPES_ORDERED].fillna(0)
 
    # Convert both to proportions
    obs_prop = obs.div(obs.sum(axis=1), axis=0)
    exp_prop = exp.div(exp.sum(axis=1), axis=0)
 
    # χ² goodness-of-fit per group (observed vs expected)
    group_stats = {}
    for group in obs.index:
        obs_counts = obs.loc[group].values.astype(float)
        exp_counts = exp.loc[group].values.astype(float)
        # Normalize expected to match observed total
        exp_scaled = exp_counts * (obs_counts.sum() / exp_counts.sum())
        # Drop cells where expected == 0 to avoid divide-by-zero in chisquare
        mask = exp_scaled > 0
        chi2, p = stats.chisquare(obs_counts[mask], f_exp=exp_scaled[mask])
        group_stats[group] = {
            "chi2": float(chi2),
            "p": float(p),
            "sig": significance_stars(p),
            "dof": int(mask.sum() - 1),
        }
 
    # ── Plot: 4 bars ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIGSIZE_SINGLE[0] + 1.0, FIGSIZE_SINGLE[1] + 0.3))
 
    groups = list(obs.index)  # ['neg', 'pos']
    bar_labels = []
    bar_data = []
    for group in groups:
        bar_labels.append(f"{group}\nobserved")
        bar_data.append(obs_prop.loc[group].values)
        bar_labels.append(f"{group}\nexpected")
        bar_data.append(exp_prop.loc[group].values)
 
    x = np.arange(len(bar_labels))
    bottoms = np.zeros(len(bar_labels))
 
    for i, event in enumerate(RG_EVENT_TYPES_ORDERED):
        vals = np.array([row[i] for row in bar_data])
        ax.bar(
            x, vals, bottom=bottoms,
            color=_EVENT_COLORS[event], edgecolor="black",
            linewidth=0.4, width=0.7, label=event,
        )
        for j, (val, bot) in enumerate(zip(vals, bottoms)):
            if val > 0.03:
                ax.text(
                    x[j], bot + val / 2, f"{val * 100:.1f}%",
                    ha="center", va="center", fontsize=6,
                    color="black" if event == "no_change" else "white",
                )
        bottoms += vals
 
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of missense variants")
    ax.set_xlabel("")
    ax.set_title("RG change events: observed vs expected (enumeration null)")
 
    # n above each observed bar
    for idx, group in enumerate(groups):
        n_obs = int(obs.loc[group].sum())
        ax.text(idx * 2, 1.02, f"n = {n_obs:,}",
                ha="center", va="bottom", fontsize=6.5)
 
    # Annotations for χ² per group (positioned above the pair of bars)
    for idx, group in enumerate(groups):
        s = group_stats[group]
        x_center = idx * 2 + 0.5
        ax.text(
            x_center, 1.08,
            f"χ² p = {s['p']:.2e} {s['sig']}",
            ha="center", va="bottom", fontsize=6.5,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray",
                      pad=2, linewidth=0.4),
        )
 
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="Event")
 
    sns.despine()
    plt.tight_layout()
    if save:
        save_figure(fig, "rg_events_observed_vs_expected", dataset=dataset)
 
    print(f"\n── RG change events: observed vs expected ({dataset}) ──")
    print("\nObserved proportions:")
    print(obs_prop.round(4))
    print("\nExpected proportions (enumeration null):")
    print(exp_prop.round(4))
    print("\nχ² goodness-of-fit per group (observed vs expected):")
    for g, s in group_stats.items():
        print(f"  {g}: χ² = {s['chi2']:.2f}, df = {s['dof']}, "
              f"p = {s['p']:.2e} {s['sig']}")
 
    return {
        "observed": obs,
        "expected": exp,
        "observed_prop": obs_prop,
        "expected_prop": exp_prop,
        "group_stats": group_stats,
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# Plot: observed vs expected consequence distribution
# ════════════════════════════════════════════════════════════════════════════
 
def plot_consequences_observed_vs_expected(
    df_observed: pd.DataFrame,
    null_results: dict,
    dataset: str = "gnomad",
    save: bool = True,
    consequence_order: list[str] = None,
) -> dict:
    """
    Same logic as plot_rg_events_observed_vs_expected, but for the consequence
    distribution. The enumeration null only knows these categories:
    synonymous, missense, nonsense, stop_lost.
 
    Observed uses your existing collapse_consequence mapping; we restrict to
    categories the null can produce. LoF (frameshift/splice) and inframe_indel
    cannot arise from single-nt substitutions and are excluded from this
    comparison.
    """
    from src.analysis_visualization.region_analysis import collapse_consequence
 
    # Observed side
    df = df_observed.copy()
    df["consequence_class"] = df["Consequence"].apply(collapse_consequence)
 
    # Keep only categories the null could generate
    # collapse_consequence maps nonsense/stop_lost into "LoF" — so we
    # need a finer version here: keep missense, synonymous separately
    # and call the rest "nonsense" (includes stop_gained and stop_lost).
    def _refine(row):
        cons = str(row["Consequence"]) if pd.notna(row["Consequence"]) else ""
        if "synonymous_variant" in cons: return "synonymous"
        if "missense_variant" in cons:   return "missense"
        if "stop_gained" in cons:        return "nonsense"
        if "stop_lost" in cons:          return "stop_lost"
        return "other"
    df["cons_refined"] = df.apply(_refine, axis=1)
    df = df[df["cons_refined"].isin(["synonymous", "missense", "nonsense", "stop_lost"])]
 
    cons_order = consequence_order or ["synonymous", "missense", "nonsense", "stop_lost"]
 
    obs = pd.crosstab(df["group"], df["cons_refined"])
    for c in cons_order:
        if c not in obs.columns:
            obs[c] = 0
    obs = obs[cons_order]
 
    null_df = null_results["consequences_null"]
    exp = null_df.pivot(index="group", columns="consequence", values="expected_count")
    for c in cons_order:
        if c not in exp.columns:
            exp[c] = 0
    exp = exp.reindex(index=obs.index)[cons_order].fillna(0)
 
    obs_prop = obs.div(obs.sum(axis=1), axis=0)
    exp_prop = exp.div(exp.sum(axis=1), axis=0)
 
    # Per-group χ² goodness-of-fit
    group_stats = {}
    for group in obs.index:
        obs_counts = obs.loc[group].values.astype(float)
        exp_counts = exp.loc[group].values.astype(float)
        exp_scaled = exp_counts * (obs_counts.sum() / exp_counts.sum())
        mask = exp_scaled > 0
        chi2, p = stats.chisquare(obs_counts[mask], f_exp=exp_scaled[mask])
        group_stats[group] = {
            "chi2": float(chi2), "p": float(p),
            "sig": significance_stars(p),
            "dof": int(mask.sum() - 1),
        }
 
    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIGSIZE_SINGLE[0] + 1.0, FIGSIZE_SINGLE[1] + 0.3))
 
    cons_colors = {
        "synonymous": "#C7C7C7",
        "missense":   "#E69F00",
        "nonsense":   "#D55E00",
        "stop_lost":  "#56B4E9",
    }
 
    groups = list(obs.index)
    bar_labels = []
    bar_data = []
    for group in groups:
        bar_labels.append(f"{group}\nobserved")
        bar_data.append(obs_prop.loc[group].values)
        bar_labels.append(f"{group}\nexpected")
        bar_data.append(exp_prop.loc[group].values)
 
    x = np.arange(len(bar_labels))
    bottoms = np.zeros(len(bar_labels))
 
    for i, cons in enumerate(cons_order):
        vals = np.array([row[i] for row in bar_data])
        ax.bar(
            x, vals, bottom=bottoms,
            color=cons_colors.get(cons, "#888"), edgecolor="black",
            linewidth=0.4, width=0.7, label=cons,
        )
        for j, (val, bot) in enumerate(zip(vals, bottoms)):
            if val > 0.03:
                ax.text(
                    x[j], bot + val / 2, f"{val * 100:.1f}%",
                    ha="center", va="center", fontsize=6,
                    color="white" if cons in ("nonsense", "missense") else "black",
                )
        bottoms += vals
 
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction")
    ax.set_title("Consequences: observed vs expected (single-nt substitution null)")
 
    for idx, group in enumerate(groups):
        n_obs = int(obs.loc[group].sum())
        ax.text(idx * 2, 1.02, f"n = {n_obs:,}",
                ha="center", va="bottom", fontsize=6.5)
 
    for idx, group in enumerate(groups):
        s = group_stats[group]
        x_center = idx * 2 + 0.5
        ax.text(
            x_center, 1.08,
            f"χ² p = {s['p']:.2e} {s['sig']}",
            ha="center", va="bottom", fontsize=6.5,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray",
                      pad=2, linewidth=0.4),
        )
 
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="Consequence")
 
    sns.despine()
    plt.tight_layout()
    if save:
        save_figure(fig, "consequences_observed_vs_expected", dataset=dataset)
 
    print(f"\n── Consequences: observed vs expected ({dataset}) ──")
    print("\nObserved proportions:")
    print(obs_prop.round(4))
    print("\nExpected proportions (enumeration null):")
    print(exp_prop.round(4))
    print("\nχ² goodness-of-fit per group:")
    for g, s in group_stats.items():
        print(f"  {g}: χ² = {s['chi2']:.2f}, df = {s['dof']}, "
              f"p = {s['p']:.2e} {s['sig']}")
 
    return {
        "observed": obs,
        "expected": exp,
        "observed_prop": obs_prop,
        "expected_prop": exp_prop,
        "group_stats": group_stats,
    }
 

 
GROUP_COLORS_PALE = {"pos": "#B8DFAA", "neg": "#E5BEBE"}
 
 
def _box_colors_ordered():
    """Returns the 4 colors in the order: neg_obs, neg_exp, pos_obs, pos_exp."""
    return [
        GROUP_COLORS["neg"],        # neg observed (saturated)
        GROUP_COLORS_PALE["neg"],   # neg expected (pale)
        GROUP_COLORS["pos"],        # pos observed (saturated)
        GROUP_COLORS_PALE["pos"],   # pos expected (pale)
    ]
 
 
# ════════════════════════════════════════════════════════════════════════════
# Per-region proportion builders
# ════════════════════════════════════════════════════════════════════════════
 
def _per_region_rg_event_proportions(
    df_events: pd.DataFrame,
    null_results: dict,
) -> pd.DataFrame:
    """
    For each region, compute observed and expected proportion of each RG event
    category. Returns a long-form dataframe suitable for seaborn boxplots with:
        region_id, group, rg_event, source ('observed'/'expected'), proportion
    """
    df_events = df_events[df_events["rg_change_event"].notna()].copy()
 
    # Observed proportions per region
    obs_counts = (
        df_events.groupby(["region_id", "group", "rg_change_event"])
                 .size()
                 .reset_index(name="count")
    )
    totals = (
        obs_counts.groupby(["region_id", "group"])["count"]
                  .sum().reset_index(name="total")
    )
    obs = obs_counts.merge(totals, on=["region_id", "group"])
    obs["proportion"] = obs["count"] / obs["total"]
    obs = obs[["region_id", "group", "rg_change_event", "proportion"]].rename(
        columns={"rg_change_event": "rg_event"}
    )
    obs["source"] = "observed"
 
    # Expected proportions per region (from enumeration)
    per_region_enum = null_results["per_region_enumerations"]
    region_groups = {rid: null_results["rg_events_null"]
                     .set_index(["group"]).index for rid in per_region_enum}
 
    exp_rows = []
    # Need to know each region's group — rebuild from df_events for safety
    group_lookup = (
        df_events[["region_id", "group"]].drop_duplicates()
        .set_index("region_id")["group"].to_dict()
    )
    for rid, enum_df in per_region_enum.items():
        group = group_lookup.get(rid)
        if group is None:
            continue
        missense_enum = enum_df[enum_df["consequence"] == "missense"]
        if len(missense_enum) == 0:
            continue
        props = missense_enum["rg_event"].value_counts(normalize=True)
        for event, prop in props.items():
            exp_rows.append({
                "region_id": rid, "group": group,
                "rg_event": event, "proportion": prop,
            })
    exp = pd.DataFrame(exp_rows)
    exp["source"] = "expected"
 
    return pd.concat([obs, exp], ignore_index=True)
 
 
def _per_region_consequence_proportions(
    df_observed: pd.DataFrame,
    null_results: dict,
    categories: list[str],
) -> pd.DataFrame:
    """
    Analog to the RG event version, but for consequence classes.
    `categories` is the list of consequence classes to compare
    (must be producible by single-nt substitution).
    """
    df = df_observed.copy()
 
    # Refine consequence into the null-comparable categories
    def _refine(cons):
        if not isinstance(cons, str):
            return "other"
        for term in cons.split("&"):
            if term == "synonymous_variant": return "synonymous"
            if term == "missense_variant":   return "missense"
            if term == "stop_gained":        return "nonsense"
        return "other"
 
    df["cons_refined"] = df["Consequence"].apply(_refine)
    df = df[df["cons_refined"].isin(categories)]
 
    obs_counts = (
        df.groupby(["region_id", "group", "cons_refined"])
          .size().reset_index(name="count")
    )
    totals = (
        obs_counts.groupby(["region_id", "group"])["count"]
                  .sum().reset_index(name="total")
    )
    obs = obs_counts.merge(totals, on=["region_id", "group"])
    obs["proportion"] = obs["count"] / obs["total"]
    obs = obs[["region_id", "group", "cons_refined", "proportion"]].rename(
        columns={"cons_refined": "consequence"}
    )
    obs["source"] = "observed"
 
    # Expected per region from enumeration
    per_region_enum = null_results["per_region_enumerations"]
    group_lookup = (
        df_observed[["region_id", "group"]].drop_duplicates()
        .set_index("region_id")["group"].to_dict()
    )
 
    exp_rows = []
    for rid, enum_df in per_region_enum.items():
        group = group_lookup.get(rid)
        if group is None:
            continue
        # Restrict enumeration to the same categories
        enum_sub = enum_df[enum_df["consequence"].isin(categories)]
        if len(enum_sub) == 0:
            continue
        props = enum_sub["consequence"].value_counts(normalize=True)
        for cons, prop in props.items():
            exp_rows.append({
                "region_id": rid, "group": group,
                "consequence": cons, "proportion": prop,
            })
    exp = pd.DataFrame(exp_rows)
    exp["source"] = "expected"
 
    return pd.concat([obs, exp], ignore_index=True)
 
 
# ════════════════════════════════════════════════════════════════════════════
# Shared plotting helper: 4-box panel (neg_obs, neg_exp, pos_obs, pos_exp)
# ════════════════════════════════════════════════════════════════════════════
 
def _four_box_panel(ax, data, y_col, title):
    """
    Draw the 4-box comparison on `ax`:
        neg observed, neg expected, pos observed, pos expected.
 
    `data` is a long-form df with columns ['group', 'source', <y_col>].
    Returns (p_neg_obs_vs_exp, p_pos_obs_vs_exp, p_pos_vs_neg_observed).
    """
    # Build an ordered categorical "box" column
    def _label(row):
        return f"{row['group']}\n{row['source']}"
    data = data.copy()
    data["box"] = data.apply(_label, axis=1)
 
    box_order = [
        "neg\nobserved", "neg\nexpected", "pos\nobserved", "pos\nexpected"
    ]
    colors = _box_colors_ordered()
 
    sns.boxplot(
        data=data, x="box", y=y_col, order=box_order,
        palette=colors, width=0.55, fliersize=0,
        linewidth=0.6, ax=ax,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 3.5},
    )
    sns.stripplot(
        data=data, x="box", y=y_col, order=box_order,
        color="black", size=1.0, alpha=0.3, jitter=0.15, ax=ax,
    )
 
    # Statistical tests
    def _pair(g_a, s_a, g_b, s_b):
        a = data[(data["group"] == g_a) & (data["source"] == s_a)][y_col].dropna()
        b = data[(data["group"] == g_b) & (data["source"] == s_b)][y_col].dropna()
        if len(a) < 3 or len(b) < 3:
            return (np.nan, "n.s.")
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return (float(p), significance_stars(p))
 
    p_neg_dev, sig_neg_dev = _pair("neg", "observed", "neg", "expected")
    p_pos_dev, sig_pos_dev = _pair("pos", "observed", "pos", "expected")
    p_between,  sig_between = _pair("pos", "observed", "neg", "observed")
 
    # Significance annotations
    ymax = data[y_col].quantile(0.98)
    if pd.isna(ymax) or ymax <= 0:
        ymax = 1.0
    # Bracket: neg_obs (0) vs neg_exp (1)
    ax.plot([0, 1], [ymax * 1.05, ymax * 1.05], color="black", lw=0.5)
    ax.text(0.5, ymax * 1.07, sig_neg_dev, ha="center", va="bottom", fontsize=7)
    # Bracket: pos_obs (2) vs pos_exp (3)
    ax.plot([2, 3], [ymax * 1.05, ymax * 1.05], color="black", lw=0.5)
    ax.text(2.5, ymax * 1.07, sig_pos_dev, ha="center", va="bottom", fontsize=7)
    # Bracket: neg_obs (0) vs pos_obs (2) — above the inner brackets
    ax.plot([0, 2], [ymax * 1.18, ymax * 1.18], color="black", lw=0.5)
    ax.text(1.0, ymax * 1.20, sig_between, ha="center", va="bottom", fontsize=7)
 
    ax.set_ylim(0, ymax * 1.30)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=6.5)
 
    return {
        "p_neg_observed_vs_expected": p_neg_dev,
        "p_pos_observed_vs_expected": p_pos_dev,
        "p_pos_vs_neg_observed":      p_between,
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# Plot 1: RG events observed vs expected, one subplot per event
# ════════════════════════════════════════════════════════════════════════════
 
RG_EVENT_TYPES_ORDERED = ["no_change", "loss", "gain", "movement"]
 
 
def plot_rg_events_vs_expected_boxes(
    df_events: pd.DataFrame,
    null_results: dict,
    dataset: str = "gnomad",
    save: bool = True,
) -> dict:
    """
    Four subplots (one per event), each showing 4 boxes:
    neg observed, neg expected, pos observed, pos expected.
    """
    long_df = _per_region_rg_event_proportions(df_events, null_results)
 
    fig, axes = plt.subplots(
        1, 4,
        figsize=(FIGSIZE_DOUBLE[0] * 1.1, FIGSIZE_SINGLE[1] + 0.5),
        sharey=False,
    )
 
    results = {}
    for ax, event in zip(axes, RG_EVENT_TYPES_ORDERED):
        sub = long_df[long_df["rg_event"] == event]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        stats_dict = _four_box_panel(ax, sub, "proportion", event)
        results[event] = stats_dict
        if ax is axes[0]:
            ax.set_ylabel("Per-region proportion")
        else:
            ax.set_ylabel("")
 
    fig.suptitle(f"RG change events: observed vs expected ({dataset})", y=1.02)
    sns.despine(fig=fig)
    plt.tight_layout()
 
    if save:
        save_figure(fig, "rg_events_vs_expected_boxes", dataset=dataset)
 
    print(f"\n── RG change events: observed vs expected ({dataset}) ──")
    for event, s in results.items():
        print(f"  {event}:")
        print(f"    neg observed vs expected: p = {s['p_neg_observed_vs_expected']:.2e}")
        print(f"    pos observed vs expected: p = {s['p_pos_observed_vs_expected']:.2e}")
        print(f"    pos vs neg (observed):    p = {s['p_pos_vs_neg_observed']:.2e}")
    return results
 
 
# ════════════════════════════════════════════════════════════════════════════
# Plot 2: Consequence distribution observed vs expected, per-region box plots
# ════════════════════════════════════════════════════════════════════════════
 
def plot_consequences_vs_expected_boxes(
    df_observed: pd.DataFrame,
    null_results: dict,
    dataset: str = "gnomad",
    save: bool = True,
) -> dict:
    """
    Three subplots (synonymous, missense, nonsense), each showing 4 boxes
    (neg_obs, neg_exp, pos_obs, pos_exp). Categories restricted to those
    that can arise from single-nt substitutions.
    """
    categories = ["synonymous", "missense", "nonsense"]
    long_df = _per_region_consequence_proportions(
        df_observed, null_results, categories=categories
    )
 
    fig, axes = plt.subplots(
        1, len(categories),
        figsize=(FIGSIZE_DOUBLE[0], FIGSIZE_SINGLE[1] + 0.5),
        sharey=False,
    )
 
    results = {}
    for ax, cons in zip(axes, categories):
        sub = long_df[long_df["consequence"] == cons]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        stats_dict = _four_box_panel(ax, sub, "proportion", cons)
        results[cons] = stats_dict
        if ax is axes[0]:
            ax.set_ylabel("Per-region proportion")
        else:
            ax.set_ylabel("")
 
    fig.suptitle(
        f"Consequence distribution: observed vs expected ({dataset})",
        y=1.02,
    )
    sns.despine(fig=fig)
    plt.tight_layout()
 
    if save:
        save_figure(fig, "consequences_vs_expected_boxes", dataset=dataset)
 
    print(f"\n── Consequences: observed vs expected ({dataset}) ──")
    for cons, s in results.items():
        print(f"  {cons}:")
        print(f"    neg observed vs expected: p = {s['p_neg_observed_vs_expected']:.2e}")
        print(f"    pos observed vs expected: p = {s['p_pos_observed_vs_expected']:.2e}")
        print(f"    pos vs neg (observed):    p = {s['p_pos_vs_neg_observed']:.2e}")
 
    return results
 

  
def _rg_ratio(seq: str) -> float:
    """Return n_R / n_G for a sequence. NaN if no G residues."""
    if not isinstance(seq, str) or len(seq) == 0:
        return np.nan
    n_r = seq.count("R")
    n_g = seq.count("G")
    if n_g == 0:
        return np.nan
    return n_r / n_g
 
 
def compute_delta_rg_ratio(
    df_rg: pd.DataFrame,
    region_by_id: dict,
) -> pd.DataFrame:
    """
    For every missense variant, compute the R/G ratio before and after the
    substitution and the relative delta.
 
    Adds columns: wt_rg_ratio, mut_rg_ratio, delta_rg_ratio_rel
    Returns a new dataframe (missense-only) with these columns appended.
    """
    df = df_rg[
        df_rg["Consequence"].fillna("").str.contains("missense_variant")
    ].copy()
 
    wt_ratios = []
    mut_ratios = []
    rel_deltas = []
    affects_rg = []
 
    for row in df.itertuples(index=False):
        region = region_by_id.get(row.region_id)
        if region is None or pd.isna(row.protein_position_int) or row.after_aa is None:
            wt_ratios.append(np.nan)
            mut_ratios.append(np.nan)
            rel_deltas.append(np.nan)
            affects_rg.append(False)
            continue
 
        seq_wt = region["prot_seq"]
        pos = int(row.protein_position_int) - int(row.region_start_aa)
        if pos < 0 or pos >= len(seq_wt):
            wt_ratios.append(np.nan)
            mut_ratios.append(np.nan)
            rel_deltas.append(np.nan)
            affects_rg.append(False)
            continue
 
        seq_mut = seq_wt[:pos] + row.after_aa + seq_wt[pos + 1:]
        wt_ratio = _rg_ratio(seq_wt)
        mut_ratio = _rg_ratio(seq_mut)
        if pd.notna(wt_ratio) and wt_ratio != 0:
            rel_delta = (mut_ratio - wt_ratio) / wt_ratio
        else:
            rel_delta = np.nan
 
        wt_ratios.append(wt_ratio)
        mut_ratios.append(mut_ratio)
        rel_deltas.append(rel_delta)
        # R/G-affecting = before or after is R or G
        affects_rg.append(
            (row.before_aa in ("R", "G")) or (row.after_aa in ("R", "G"))
        )
 
    df["wt_rg_ratio"] = wt_ratios
    df["mut_rg_ratio"] = mut_ratios
    df["delta_rg_ratio_rel"] = rel_deltas
    df["affects_rg_residue"] = affects_rg
    return df
 
 
# ════════════════════════════════════════════════════════════════════════════
# Plot A: per-variant distribution, filtered to R/G-affecting missense
# ════════════════════════════════════════════════════════════════════════════
 
def plot_delta_rg_ratio_per_variant(
    df_rg: pd.DataFrame,
    region_by_id: dict,
    dataset: str = "gnomad",
    save: bool = True,
) -> tuple[plt.Figure, dict]:
    """
    [Dataset-agnostic]
    Box plot of relative delta RG ratio, restricted to missense variants that
    touch an R or G (before_aa ∈ {R,G} or after_aa ∈ {R,G}).
    One data point per variant; compared between pos and neg.
    """
    df = compute_delta_rg_ratio(df_rg, region_by_id)
    df = df[df["affects_rg_residue"] & df["delta_rg_ratio_rel"].notna()]
 
    pos_vals = df.loc[df["group"] == "pos", "delta_rg_ratio_rel"]
    neg_vals = df.loc[df["group"] == "neg", "delta_rg_ratio_rel"]
    _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    sig = significance_stars(p)
 
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    sns.boxplot(
        data=df, x="group", y="delta_rg_ratio_rel",
        order=["neg", "pos"],
        palette=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
        width=0.5, fliersize=0, linewidth=0.6, ax=ax,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 4},
    )
    sns.stripplot(
        data=df, x="group", y="delta_rg_ratio_rel",
        order=["neg", "pos"], color="black", size=1.0, alpha=0.25,
        jitter=0.2, ax=ax,
    )
 
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
 
    ymax = df["delta_rg_ratio_rel"].abs().quantile(0.98)
    y_bar = ymax * 1.1
    ax.plot([0, 1], [y_bar, y_bar], color="black", lw=0.6)
    ax.text(0.5, y_bar * 1.05, sig, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(-ymax * 1.3, ymax * 1.3)
 
    ax.set_title("Relative Δ RG ratio per variant\n(R/G-affecting missense only)")
    ax.set_ylabel("(mutant_ratio − WT_ratio) / WT_ratio")
    ax.set_xlabel("")
 
    stats_text = (
        f"p = {p:.1e} {sig}\n"
        f"n_pos = {len(pos_vals):,}\n"
        f"n_neg = {len(neg_vals):,}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=6.5, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2))
 
    sns.despine()
    plt.tight_layout()
 
    if save:
        save_figure(fig, "delta_rg_ratio_per_variant", dataset=dataset)
 
    results = {
        "p": float(p), "sig": sig,
        "median_pos": float(pos_vals.median()),
        "median_neg": float(neg_vals.median()),
        "mean_pos": float(pos_vals.mean()),
        "mean_neg": float(neg_vals.mean()),
        "n_pos": int(len(pos_vals)),
        "n_neg": int(len(neg_vals)),
    }
 
    print(f"\n── Δ RG ratio per variant (R/G-affecting missense) [{dataset}] ──")
    print(f"  pos: median = {results['median_pos']:+.4f}, "
          f"mean = {results['mean_pos']:+.4f}, n = {results['n_pos']:,}")
    print(f"  neg: median = {results['median_neg']:+.4f}, "
          f"mean = {results['mean_neg']:+.4f}, n = {results['n_neg']:,}")
    print(f"  Mann-Whitney p = {p:.2e} {sig}")
 
    return fig, results
 
 
# ════════════════════════════════════════════════════════════════════════════
# Plot B: per-region mean delta across ALL missense variants
# ════════════════════════════════════════════════════════════════════════════
 
def plot_delta_rg_ratio_per_region(
    df_rg: pd.DataFrame,
    region_by_id: dict,
    dataset: str = "gnomad",
    save: bool = True,
) -> tuple[plt.Figure, dict]:
    """
    [Dataset-agnostic]
    Box plot of per-region mean relative delta RG ratio, across ALL missense
    variants in the region. One data point per region; compared between pos
    and neg.
    """
    df = compute_delta_rg_ratio(df_rg, region_by_id)
    df = df[df["delta_rg_ratio_rel"].notna()]
 
    per_region = (
        df.groupby(["region_id", "group"])["delta_rg_ratio_rel"]
          .mean()
          .reset_index(name="mean_delta_rg_ratio_rel")
    )
 
    pos_vals = per_region.loc[per_region["group"] == "pos", "mean_delta_rg_ratio_rel"]
    neg_vals = per_region.loc[per_region["group"] == "neg", "mean_delta_rg_ratio_rel"]
    _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    sig = significance_stars(p)
 
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    sns.boxplot(
        data=per_region, x="group", y="mean_delta_rg_ratio_rel",
        order=["neg", "pos"],
        palette=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
        width=0.5, fliersize=0, linewidth=0.6, ax=ax,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 4},
    )
    sns.stripplot(
        data=per_region, x="group", y="mean_delta_rg_ratio_rel",
        order=["neg", "pos"], color="black", size=1.5, alpha=0.4,
        jitter=0.15, ax=ax,
    )
 
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
 
    ymax = per_region["mean_delta_rg_ratio_rel"].abs().quantile(0.98)
    y_bar = ymax * 1.1
    ax.plot([0, 1], [y_bar, y_bar], color="black", lw=0.6)
    ax.text(0.5, y_bar * 1.05, sig, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(-ymax * 1.3, ymax * 1.3)
 
    ax.set_title("Mean relative Δ RG ratio per region\n(all missense)")
    ax.set_ylabel("Mean (mutant_ratio − WT_ratio) / WT_ratio")
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
        save_figure(fig, "delta_rg_ratio_per_region", dataset=dataset)
 
    results = {
        "p": float(p), "sig": sig,
        "median_pos": float(pos_vals.median()),
        "median_neg": float(neg_vals.median()),
        "mean_pos": float(pos_vals.mean()),
        "mean_neg": float(neg_vals.mean()),
        "n_pos": int(len(pos_vals)),
        "n_neg": int(len(neg_vals)),
    }
 
    print(f"\n── Per-region mean Δ RG ratio (all missense) [{dataset}] ──")
    print(f"  pos: median = {results['median_pos']:+.4f}, "
          f"mean = {results['mean_pos']:+.4f}, n = {results['n_pos']}")
    print(f"  neg: median = {results['median_neg']:+.4f}, "
          f"mean = {results['mean_neg']:+.4f}, n = {results['n_neg']}")
    print(f"  Mann-Whitney p = {p:.2e} {sig}")
 
    return fig, results
 