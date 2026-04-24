"""
Save as: src/analysis_visualization/homolog_entropy.py

Per-motif conservation analysis from homolog alignments.

Computes Shannon entropy at each motif position across homolog hits, then
aggregates to per-motif features (mean, variance, fraction conserved,
fraction invariant). Reports both gap-excluding and gap-including variants.

Gene-agnostic: entropy is computed per motif from its own homologs; gene
identity does not directly drive the result.

Includes the query sequence as a row in the alignment by default (one
extra observation per motif).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.analysis_visualization.plot_config import (
    GROUP_COLORS, save_figure, significance_stars,
)


# ════════════════════════════════════════════════════════════════════════════
# Amino acid alphabet constants
# ════════════════════════════════════════════════════════════════════════════

AA_NO_GAP = list("ACDEFGHIKLMNPQRSTVWY")       # 20 AAs
AA_WITH_GAP = AA_NO_GAP + ["-"]                # 21 states

# Conservation thresholds (nats — no, actually log2, so these are bits)
H_INVARIANT = 0.0
H_CONSERVED_BITS = 0.5


# ════════════════════════════════════════════════════════════════════════════
# Position-level entropy
# ════════════════════════════════════════════════════════════════════════════

def _shannon_entropy(counts: np.ndarray) -> float:
    """
    Shannon entropy of a count vector. Ignores zero-count cells in the log.
    Returns entropy in bits.
    """
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def positional_entropy(
    seqs: list, exclude_gaps: bool = False,
) -> np.ndarray:
    """
    [Dataset-agnostic]
    Column-wise Shannon entropy across a list of equal-length sequences.

    If exclude_gaps=True: gap positions are removed before counting, and the
    entropy of that column is computed only over the non-gap AAs. A column
    with all gaps returns NaN.

    If exclude_gaps=False: gaps are treated as a 21st symbol.
    """
    if len(seqs) == 0:
        return np.array([])

    length = len(seqs[0])
    alphabet = AA_NO_GAP if exclude_gaps else AA_WITH_GAP
    entropies = np.full(length, np.nan, dtype=float)

    for i in range(length):
        col = [s[i] for s in seqs]
        if exclude_gaps:
            col = [c for c in col if c != "-"]
        if not col:
            continue  # NaN stays
        counts = np.array([col.count(aa) for aa in alphabet], dtype=float)
        entropies[i] = _shannon_entropy(counts)

    return entropies


# ════════════════════════════════════════════════════════════════════════════
# Per-motif feature extraction
# ════════════════════════════════════════════════════════════════════════════

def compute_motif_entropy_features(
    df: pd.DataFrame,
    query_col: str = "UniqueID",
    motif_col: str = "orig_motif_index",
    group_col: str = "group",
    qseq_col: str = "qseq_rg_region",
    hseq_col: str = "hseq_rg_region",
    min_homologs: int = 10,
    include_query_in_alignment: bool = True,
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    Per-motif entropy aggregations.

    Parameters
    ----------
    min_homologs : int
        Motifs with fewer than this many (valid, equal-length) homolog sequences
        are skipped. Entropy on small columns is unreliable.
    include_query_in_alignment : bool
        If True, the query sequence is added as an extra row in the alignment
        before computing column-wise entropy.

    Returns
    -------
    DataFrame with one row per motif and columns:
        motif_uid, UniqueID, orig_motif_index, group
        n_alignment_rows      — rows used in entropy calculation
        motif_length
        mean_entropy          — average entropy across positions (with gaps)
        mean_entropy_nogap    — average entropy across positions (gaps excluded)
        std_entropy           — std of per-position entropy (with gaps)
        std_entropy_nogap     — std of per-position entropy (gaps excluded)
        fraction_invariant    — fraction of positions with H = 0 (no-gap basis)
        fraction_conserved    — fraction of positions with H < 0.5 bits
        mean_gap_fraction     — average fraction of gapped rows per position
    """
    records = []
    skipped_low_n = 0
    skipped_bad_aln = 0

    for (uid, midx), grp in df.groupby([query_col, motif_col]):
        seqs = grp[hseq_col].dropna().tolist()
        if len(grp) == 0:
            continue
        qseq = grp[qseq_col].iloc[0]

        # Filter to valid, equal-length sequences
        L = len(qseq) if include_query_in_alignment else (len(seqs[0]) if seqs else 0)
        seqs = [s for s in seqs if isinstance(s, str) and len(s) == L]

        if include_query_in_alignment and isinstance(qseq, str) and len(qseq) == L:
            alignment = [qseq] + seqs
        else:
            alignment = seqs

        if len(alignment) < min_homologs:
            skipped_low_n += 1
            continue
        if L == 0:
            skipped_bad_aln += 1
            continue

        # Compute positional entropies
        ent_with_gap = positional_entropy(alignment, exclude_gaps=False)
        ent_no_gap = positional_entropy(alignment, exclude_gaps=True)

        # Gap fraction per position (average across positions)
        gap_fracs = np.array([
            sum(1 for s in alignment if s[i] == "-") / len(alignment)
            for i in range(L)
        ])

        # Use no-gap entropies for conservation thresholds since they reflect
        # AA-level diversity, not alignment gappiness
        frac_invariant = float(np.mean(ent_no_gap == H_INVARIANT))
        frac_conserved = float(np.mean(ent_no_gap < H_CONSERVED_BITS))

        records.append({
            "motif_uid": f"{uid}_m{midx}",
            "UniqueID": uid,
            "orig_motif_index": midx,
            "group": grp[group_col].iloc[0],
            "n_alignment_rows": len(alignment),
            "motif_length": L,
            "mean_entropy": float(np.nanmean(ent_with_gap)),
            "mean_entropy_nogap": float(np.nanmean(ent_no_gap)),
            "std_entropy": float(np.nanstd(ent_with_gap)),
            "std_entropy_nogap": float(np.nanstd(ent_no_gap)),
            "fraction_invariant": frac_invariant,
            "fraction_conserved": frac_conserved,
            "mean_gap_fraction": float(np.mean(gap_fracs)),
        })

    out = pd.DataFrame(records)
    print(f"  Entropy features computed for {len(out)} motifs "
          f"(skipped: {skipped_low_n} for <{min_homologs} homologs, "
          f"{skipped_bad_aln} for bad alignment).")
    return out


# ════════════════════════════════════════════════════════════════════════════
# Statistical comparison
# ════════════════════════════════════════════════════════════════════════════

def test_entropy_features(
    features_df: pd.DataFrame,
    feature_names: list = None,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    Mann-Whitney U for each entropy feature, BH-FDR across features.
    """
    if feature_names is None:
        feature_names = [
            "mean_entropy_nogap", "mean_entropy",
            "std_entropy_nogap", "std_entropy",
            "fraction_invariant", "fraction_conserved",
            "mean_gap_fraction",
        ]

    pos = features_df[features_df[group_col] == pos_label]
    neg = features_df[features_df[group_col] == neg_label]

    rows = []
    for f in feature_names:
        pos_vals = pos[f].dropna()
        neg_vals = neg[f].dropna()
        if len(pos_vals) < 5 or len(neg_vals) < 5:
            rows.append({
                "feature": f, "n_pos": int(len(pos_vals)),
                "n_neg": int(len(neg_vals)),
                "median_pos": np.nan, "median_neg": np.nan,
                "u_stat": np.nan, "p_raw": np.nan, "p_fdr": np.nan,
                "sig": "n.s.",
            })
            continue
        u, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
        rows.append({
            "feature": f,
            "n_pos": int(len(pos_vals)),
            "n_neg": int(len(neg_vals)),
            "median_pos": float(pos_vals.median()),
            "median_neg": float(neg_vals.median()),
            "u_stat": float(u),
            "p_raw": float(p),
        })

    out = pd.DataFrame(rows)
    valid = out["p_raw"].notna()
    if valid.sum() > 0:
        _, corrected, _, _ = multipletests(
            out.loc[valid, "p_raw"].values, method="fdr_bh",
        )
        out.loc[valid, "p_fdr"] = corrected
    else:
        out["p_fdr"] = np.nan
    out["sig"] = out["p_fdr"].apply(
        lambda p: significance_stars(p) if pd.notna(p) else "n.s."
    )
    return out


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

def _box_strip_panel(
    ax, pos_vals, neg_vals, ylabel, title, p_fdr, sig,
):
    """Box + strip + significance annotation, consistent with Phase 1 style."""
    plot_data = [neg_vals.values, pos_vals.values]
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

    bp = ax.boxplot(
        plot_data, positions=[0, 1], widths=0.22, showfliers=False,
        patch_artist=True, zorder=3,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    # Strip
    for x_pos, vals, color in zip([0, 1], plot_data, colors):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.full(len(vals), x_pos) + jitter, vals,
            color=color, alpha=0.45, s=5, edgecolor="none", zorder=2,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["neg", "pos"])
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(
        0.5, 0.96, f"p_fdr = {p_fdr:.2e} {sig}",
        transform=ax.transAxes, ha="center", va="top", fontsize=7,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
    )


def plot_entropy_features(
    features_df: pd.DataFrame,
    test_results: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
    feature_names: list = None,
) -> plt.Figure:
    """
    One subplot per entropy feature. Pos vs neg distributions with box/strip
    and significance annotation.
    """
    if feature_names is None:
        feature_names = [
            "mean_entropy_nogap", "mean_entropy",
            "std_entropy_nogap", "fraction_conserved",
            "fraction_invariant", "mean_gap_fraction",
        ]

    pretty_labels = {
        "mean_entropy":          "Mean entropy (with gaps)",
        "mean_entropy_nogap":    "Mean entropy (no gaps)",
        "std_entropy":           "Entropy std (with gaps)",
        "std_entropy_nogap":     "Entropy std (no gaps)",
        "fraction_invariant":    "Fraction invariant (H = 0)",
        "fraction_conserved":    "Fraction conserved (H < 0.5)",
        "mean_gap_fraction":     "Mean gap fraction",
    }

    pos = features_df[features_df["group"] == "pos"]
    neg = features_df[features_df["group"] == "neg"]

    n = len(feature_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = axes.flatten()

    results_lookup = test_results.set_index("feature")

    for i, feat in enumerate(feature_names):
        ax = axes[i]
        pos_vals = pos[feat].dropna()
        neg_vals = neg[feat].dropna()

        if feat in results_lookup.index:
            row = results_lookup.loc[feat]
            p_fdr = float(row["p_fdr"]) if pd.notna(row["p_fdr"]) else np.nan
            sig = row["sig"]
        else:
            p_fdr, sig = np.nan, "n.s."

        _box_strip_panel(
            ax, pos_vals, neg_vals,
            ylabel="", title=pretty_labels.get(feat, feat),
            p_fdr=p_fdr, sig=sig,
        )

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Per-motif entropy & conservation features ({dataset})",
        fontsize=11, y=1.0,
    )
    plt.tight_layout()

    if save:
        save_figure(fig, "homolog_entropy_features", dataset=dataset)

    # Summary print
    print(f"\n── Entropy feature tests ({dataset}) ──")
    print(test_results[[
        "feature", "n_pos", "n_neg",
        "median_pos", "median_neg", "p_raw", "p_fdr", "sig",
    ]].to_string(index=False))

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_entropy_analysis(
    df: pd.DataFrame,
    min_homologs: int = 10,
    include_query_in_alignment: bool = True,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    [Dataset-agnostic]
    End-to-end per-motif entropy analysis: compute features, run stats, plot.
    """
    features_df = compute_motif_entropy_features(
        df,
        min_homologs=min_homologs,
        include_query_in_alignment=include_query_in_alignment,
    )
    test_results = test_entropy_features(features_df)
    plot_entropy_features(features_df, test_results, dataset=dataset, save=save)

    return {
        "features_df": features_df,
        "test_results": test_results,
    }



# ════════════════════════════════════════════════════════════════════════════
# Per-position entropy extraction
# ════════════════════════════════════════════════════════════════════════════

def compute_per_position_entropies(
    df: pd.DataFrame,
    query_col: str = "UniqueID",
    motif_col: str = "orig_motif_index",
    group_col: str = "group",
    qseq_col: str = "qseq_rg_region",
    hseq_col: str = "hseq_rg_region",
    min_homologs: int = 10,
    include_query_in_alignment: bool = True,
    exclude_gaps: bool = True,
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    Flatten per-motif alignments to per-position rows.

    Returns a long-format DataFrame with one row per (motif, position):
        motif_uid, UniqueID, orig_motif_index, group,
        position (0-indexed within motif),
        query_aa (AA at this position in query),
        entropy (Shannon entropy across homologs at this column),
        is_rg (query_aa in {"R", "G"}),
        n_alignment_rows, motif_length.
    """
    records = []
    skipped = 0

    for (uid, midx), grp in df.groupby([query_col, motif_col]):
        qseq = grp[qseq_col].iloc[0]
        seqs = grp[hseq_col].dropna().tolist()
        if not isinstance(qseq, str) or len(qseq) == 0:
            continue

        L = len(qseq)
        seqs = [s for s in seqs if isinstance(s, str) and len(s) == L]
        if include_query_in_alignment:
            alignment = [qseq] + seqs
        else:
            alignment = seqs

        if len(alignment) < min_homologs:
            skipped += 1
            continue

        ent = positional_entropy(alignment, exclude_gaps=exclude_gaps)

        motif_uid = f"{uid}_m{midx}"
        group = grp[group_col].iloc[0]
        for i, (aa, h) in enumerate(zip(qseq, ent)):
            if pd.isna(h):
                continue
            records.append({
                "motif_uid": motif_uid,
                "UniqueID": uid,
                "orig_motif_index": midx,
                "group": group,
                "position": i,
                "query_aa": aa,
                "entropy": float(h),
                "is_rg": aa in {"R", "G"},
                "n_alignment_rows": len(alignment),
                "motif_length": L,
            })

    print(f"  Per-position entropies computed: {len(records):,} positions "
          f"across motifs (skipped {skipped} motifs for low N).")
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# Analysis A: per-AA entropy, pos vs neg
# ════════════════════════════════════════════════════════════════════════════

def test_per_aa_entropy(
    pos_df: pd.DataFrame,
    aa_list: list = None,
    min_per_group: int = 10,
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    For each AA, Mann-Whitney on per-position entropies (pos vs neg).
    BH-FDR across the 20 AAs.
    """
    if aa_list is None:
        aa_list = AA_NO_GAP

    rows = []
    for aa in aa_list:
        pos_vals = pos_df.loc[
            (pos_df["group"] == "pos") & (pos_df["query_aa"] == aa),
            "entropy",
        ].dropna()
        neg_vals = pos_df.loc[
            (pos_df["group"] == "neg") & (pos_df["query_aa"] == aa),
            "entropy",
        ].dropna()

        if len(pos_vals) < min_per_group or len(neg_vals) < min_per_group:
            rows.append({
                "aa": aa, "n_pos": int(len(pos_vals)),
                "n_neg": int(len(neg_vals)),
                "median_pos": float(pos_vals.median()) if len(pos_vals) else np.nan,
                "median_neg": float(neg_vals.median()) if len(neg_vals) else np.nan,
                "u_stat": np.nan, "p_raw": np.nan, "p_fdr": np.nan, "sig": "n.s.",
            })
            continue

        u, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
        rows.append({
            "aa": aa,
            "n_pos": int(len(pos_vals)),
            "n_neg": int(len(neg_vals)),
            "median_pos": float(pos_vals.median()),
            "median_neg": float(neg_vals.median()),
            "u_stat": float(u),
            "p_raw": float(p),
        })

    out = pd.DataFrame(rows)
    valid = out["p_raw"].notna()
    if valid.sum() > 0:
        _, corrected, _, _ = multipletests(
            out.loc[valid, "p_raw"].values, method="fdr_bh",
        )
        out.loc[valid, "p_fdr"] = corrected
    else:
        out["p_fdr"] = np.nan
    out["sig"] = out["p_fdr"].apply(
        lambda p: significance_stars(p) if pd.notna(p) else "n.s."
    )
    return out


def plot_per_aa_entropy(
    pos_df: pd.DataFrame,
    test_results: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
    sort_by: str = "effect",
) -> plt.Figure:
    """
    Box plot per AA showing entropy distributions pos vs neg.

    sort_by:
      "effect" — by median_neg − median_pos (biggest conservation effect first)
      "alphabetical" — A..Y
      "aa_group" — physchem grouping (Pos / Neg / Polar / Aromatic / Hydrophobic / CGP)
    """
    AAs_sorted = test_results.copy()
    if sort_by == "effect":
        AAs_sorted["effect"] = AAs_sorted["median_neg"] - AAs_sorted["median_pos"]
        AAs_sorted = AAs_sorted.sort_values("effect", ascending=False)
    elif sort_by == "alphabetical":
        AAs_sorted = AAs_sorted.sort_values("aa")
    aa_order = AAs_sorted["aa"].tolist()

    fig, ax = plt.subplots(figsize=(13, 5))

    positions_pos = np.arange(len(aa_order)) - 0.2
    positions_neg = np.arange(len(aa_order)) + 0.2

    data_pos = [
        pos_df.loc[(pos_df["group"] == "pos") & (pos_df["query_aa"] == aa),
                    "entropy"].values
        for aa in aa_order
    ]
    data_neg = [
        pos_df.loc[(pos_df["group"] == "neg") & (pos_df["query_aa"] == aa),
                    "entropy"].values
        for aa in aa_order
    ]

    bp_pos = ax.boxplot(
        data_pos, positions=positions_pos, widths=0.32,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor=GROUP_COLORS["pos"], alpha=0.7, edgecolor="black"),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    bp_neg = ax.boxplot(
        data_neg, positions=positions_neg, widths=0.32,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor=GROUP_COLORS["neg"], alpha=0.7, edgecolor="black"),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    ax.set_xticks(np.arange(len(aa_order)))
    ax.set_xticklabels(aa_order, fontsize=9)
    ax.set_ylabel("Shannon entropy (no gaps)", fontsize=9)
    ax.set_xlabel("Query amino acid")
    ax.set_title(
        f"Per-AA position entropy, pos vs neg ({dataset}) — "
        f"sorted by {sort_by.replace('_', ' ')}",
        fontsize=10,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Annotate significance above each AA
    lookup = test_results.set_index("aa")
    ymax = max(
        max((np.percentile(d, 95) if len(d) > 0 else 0) for d in data_pos),
        max((np.percentile(d, 95) if len(d) > 0 else 0) for d in data_neg),
    )
    for i, aa in enumerate(aa_order):
        sig = lookup.loc[aa, "sig"]
        if sig != "n.s.":
            ax.text(
                i, ymax * 1.08, sig,
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["pos"], alpha=0.7, label="pos"),
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["neg"], alpha=0.7, label="neg"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

    plt.tight_layout()
    if save:
        save_figure(fig, f"homolog_entropy_per_aa_{sort_by}", dataset=dataset)

    # Summary print
    print(f"\n── Per-AA entropy tests ({dataset}) ──")
    print(test_results.sort_values("p_fdr")[
        ["aa", "n_pos", "n_neg", "median_pos", "median_neg", "p_fdr", "sig"]
    ].to_string(index=False))

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Analysis B: RG vs non-RG within-motif
# ════════════════════════════════════════════════════════════════════════════

def compute_rg_vs_nonrg_within_motif(
    pos_df: pd.DataFrame,
    min_positions: int = 2,
) -> pd.DataFrame:
    """
    [Dataset-agnostic, gene-agnostic]
    For each motif, compute mean entropy at R/G positions vs non-R/G positions.
    Returns per-motif DataFrame with:
        motif_uid, group,
        mean_entropy_rg, mean_entropy_nonrg,
        n_rg, n_nonrg,
        delta_entropy (= mean_rg − mean_nonrg)  — negative = RG more conserved
    """
    records = []
    for motif_uid, sub in pos_df.groupby("motif_uid"):
        rg = sub[sub["is_rg"]]["entropy"].dropna()
        nonrg = sub[~sub["is_rg"]]["entropy"].dropna()
        if len(rg) < min_positions or len(nonrg) < min_positions:
            continue
        mean_rg = float(rg.mean())
        mean_nonrg = float(nonrg.mean())
        records.append({
            "motif_uid": motif_uid,
            "group": sub["group"].iloc[0],
            "mean_entropy_rg": mean_rg,
            "mean_entropy_nonrg": mean_nonrg,
            "n_rg": int(len(rg)),
            "n_nonrg": int(len(nonrg)),
            "delta_entropy": mean_rg - mean_nonrg,
        })
    return pd.DataFrame(records)


def plot_rg_vs_nonrg_within_motif(
    comp_df: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    Three-panel plot:
      1. mean_entropy_rg: pos vs neg
      2. mean_entropy_nonrg: pos vs neg
      3. delta_entropy (rg − nonrg): pos vs neg, with 0 line
    Plus a one-sided sign test on delta_entropy (testing whether delta is
    consistently negative → RG more conserved than non-RG within motifs).
    """
    pos = comp_df[comp_df["group"] == "pos"]
    neg = comp_df[comp_df["group"] == "neg"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    def _plot_box(ax, pos_vals, neg_vals, ylabel, title):
        data = [neg_vals.values, pos_vals.values]
        colors = [GROUP_COLORS["neg"], GROUP_COLORS["pos"]]

        parts = ax.violinplot(
            data, positions=[0, 1], widths=0.75,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color); pc.set_alpha(0.35)
            pc.set_edgecolor("black"); pc.set_linewidth(0.4)

        bp = ax.boxplot(
            data, positions=[0, 1], widths=0.22, showfliers=False,
            patch_artist=True, zorder=3,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.9)
            patch.set_edgecolor("black")
        for median in bp["medians"]:
            median.set_color("black"); median.set_linewidth(1.2)

        for x_pos, vals, color in zip([0, 1], data, colors):
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(
                np.full(len(vals), x_pos) + jitter, vals,
                color=color, alpha=0.45, s=5, edgecolor="none", zorder=2,
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["neg", "pos"])
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        u, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
        sig = significance_stars(p)
        ax.text(
            0.5, 0.96, f"p = {p:.2e} {sig}",
            transform=ax.transAxes, ha="center", va="top", fontsize=7,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
        )
        return {"u": float(u), "p": float(p), "sig": sig}

    r1 = _plot_box(axes[0], pos["mean_entropy_rg"], neg["mean_entropy_rg"],
                    ylabel="Mean entropy at RG positions",
                    title="RG positions")
    r2 = _plot_box(axes[1], pos["mean_entropy_nonrg"], neg["mean_entropy_nonrg"],
                    ylabel="Mean entropy at non-RG positions",
                    title="Non-RG positions")
    r3 = _plot_box(axes[2], pos["delta_entropy"], neg["delta_entropy"],
                    ylabel="Δ entropy (RG − non-RG)",
                    title="Within-motif RG-specific conservation")
    axes[2].axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)

    # One-sample sign test: is delta < 0 consistently within each group?
    # i.e., is RG more conserved than non-RG within same motif?
    from scipy.stats import wilcoxon
    pos_delta = pos["delta_entropy"].dropna()
    neg_delta = neg["delta_entropy"].dropna()
    _, p_pos = wilcoxon(pos_delta, alternative="less")    # pos: delta < 0?
    _, p_neg = wilcoxon(neg_delta, alternative="less")    # neg: delta < 0?

    fig.suptitle(
        f"RG vs non-RG within-motif conservation ({dataset})\n"
        f"(negative delta = RG more conserved)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "homolog_rg_vs_nonrg_within_motif", dataset=dataset)

    # Summary
    print(f"\n── RG vs non-RG within-motif conservation ({dataset}) ──")
    print(f"  Motifs analyzed: pos = {len(pos)}, neg = {len(neg)}")
    print(f"\n  Mean entropy at RG positions:")
    print(f"    pos median = {pos['mean_entropy_rg'].median():.3f}, "
          f"neg median = {neg['mean_entropy_rg'].median():.3f}")
    print(f"    Mann-Whitney p = {r1['p']:.2e} {r1['sig']}")
    print(f"\n  Mean entropy at non-RG positions:")
    print(f"    pos median = {pos['mean_entropy_nonrg'].median():.3f}, "
          f"neg median = {neg['mean_entropy_nonrg'].median():.3f}")
    print(f"    Mann-Whitney p = {r2['p']:.2e} {r2['sig']}")
    print(f"\n  Delta entropy (RG − non-RG):")
    print(f"    pos median = {pos['delta_entropy'].median():.3f}, "
          f"neg median = {neg['delta_entropy'].median():.3f}")
    print(f"    Mann-Whitney pos-vs-neg p = {r3['p']:.2e} {r3['sig']}")
    print(f"\n  Within-group sign test (is delta < 0? i.e., RG more conserved):")
    print(f"    pos: Wilcoxon signed-rank p = {p_pos:.2e} "
          f"(median delta = {pos_delta.median():.3f})")
    print(f"    neg: Wilcoxon signed-rank p = {p_neg:.2e} "
          f"(median delta = {neg_delta.median():.3f})")

    return {
        "rg_test": r1,
        "nonrg_test": r2,
        "delta_test": r3,
        "within_pos_wilcoxon_p": float(p_pos),
        "within_neg_wilcoxon_p": float(p_neg),
        "comp_df": comp_df,
    }


# ════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_position_level_analysis(
    df: pd.DataFrame,
    min_homologs: int = 10,
    min_positions_per_motif: int = 2,
    include_query_in_alignment: bool = True,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    [Dataset-agnostic]
    End-to-end position-level entropy analysis: per-AA profile + RG vs non-RG
    within-motif + AA ranking.
    """
    per_pos = compute_per_position_entropies(
        df,
        min_homologs=min_homologs,
        include_query_in_alignment=include_query_in_alignment,
        exclude_gaps=True,
    )

    # Analysis A + D: per-AA entropy
    per_aa_results = test_per_aa_entropy(per_pos)
    plot_per_aa_entropy(per_pos, per_aa_results,
                        dataset=dataset, save=save, sort_by="effect")

    # Analysis B: RG vs non-RG within-motif
    comp_df = compute_rg_vs_nonrg_within_motif(
        per_pos, min_positions=min_positions_per_motif,
    )
    within_results = plot_rg_vs_nonrg_within_motif(
        comp_df, dataset=dataset, save=save,
    )

    return {
        "per_position_df": per_pos,
        "per_aa_results": per_aa_results,
        "within_motif_comp_df": comp_df,
        "within_motif_results": within_results,
    }