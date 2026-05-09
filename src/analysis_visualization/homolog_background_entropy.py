"""
Save as: src/analysis_visualization/homolog_background_entropy.py

Step 2: Slice subregions from the full-protein homolog alignments and
compute per-region mean entropy. Compare RG vs mIDR vs oIDR vs structured,
within and between pos/neg groups.

Critical detail: the full-protein qseq/hseq alignments contain gaps. To
slice protein-coordinate region [start, end], we walk through qseq counting
non-gap query positions until we reach `start`, then take characters
forward until we've covered `end - start + 1` non-gap query positions.

Deduplication: the same (UniqueID, hit_accession) alignment may appear in
many rows (once per RG motif). We deduplicate to one alignment per
(UniqueID, hit_accession) before slicing.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.analysis_visualization.plot_config import (
    GROUP_COLORS, save_figure, significance_stars,
)
from src.analysis_visualization.homolog_entropy import positional_entropy


# ════════════════════════════════════════════════════════════════════════════
# Alignment slicing
# ════════════════════════════════════════════════════════════════════════════

def slice_aligned_region(
    qseq_aligned: str,
    hseq_aligned: str,
    region_start: int,
    region_end: int,
    query_from: int = 1,
) -> Optional[Tuple[str, str]]:
    """
    Slice a sub-region from a paired alignment, given protein coordinates
    on the unaligned query.

    Parameters
    ----------
    qseq_aligned, hseq_aligned : str
        Paired alignment strings (same length, may contain '-').
    region_start, region_end : int
        1-indexed inclusive protein coordinates on the UNALIGNED query.
    query_from : int
        Position on the unaligned query where qseq_aligned starts (BLAST
        query_from; typically 1 if alignment covers the whole query).

    Returns
    -------
    (sliced_qseq, sliced_hseq) : tuple of str, or None if region falls
    outside the aligned range.
    """
    if not isinstance(qseq_aligned, str) or not isinstance(hseq_aligned, str):
        return None
    if len(qseq_aligned) != len(hseq_aligned):
        return None

    # Walk through qseq_aligned, tracking position on unaligned query
    current_query_pos = query_from - 1  # last non-gap position seen, 0-indexed
    target_start = region_start - 1     # 0-indexed
    target_end = region_end - 1         # 0-indexed inclusive

    region_aln_indices = []
    for aln_i, qchar in enumerate(qseq_aligned):
        if qchar != "-":
            current_query_pos += 1
            if target_start <= current_query_pos <= target_end:
                region_aln_indices.append(aln_i)
            elif current_query_pos > target_end:
                break

    if not region_aln_indices:
        return None

    a, b = region_aln_indices[0], region_aln_indices[-1] + 1
    return qseq_aligned[a:b], hseq_aligned[a:b]


# ════════════════════════════════════════════════════════════════════════════
# Build per-region alignment list
# ════════════════════════════════════════════════════════════════════════════

def build_region_alignments(
    df_combined: pd.DataFrame,
    classification_df: pd.DataFrame,
    qseq_col: str = "qseq",
    hseq_col: str = "hseq",
    hit_id_col: str = "hit_accession",
    query_from_col: str = "query_from",
    verbose: bool = True,
) -> dict:
    """
    For each (UniqueID, region_start, region_end, region_class), collect
    the list of sliced (q_slice, h_slice) pairs across all unique homolog
    alignments for that protein.

    Returns
    -------
    dict keyed by (UniqueID, region_start, region_end, region_class):
        {
          "qseq_slice": str            (the query slice — same across hits)
          "hseq_slices": list[str]     (one per unique homolog alignment)
          "n_alignments": int
        }
    """
    # First, deduplicate to one alignment per (UniqueID, hit_accession)
    # The same (query, hit) pair appears once per motif; keep first occurrence.
    dedup = df_combined.drop_duplicates(
        subset=["UniqueID", hit_id_col], keep="first"
    )
    if verbose:
        print(f"  Deduplicated alignment table: "
              f"{len(df_combined):,} rows → {len(dedup):,} "
              f"unique (query, hit) alignments")

    # Group by query for efficient access
    dedup_by_query = {
        uid: sub for uid, sub in dedup.groupby("UniqueID")
    }

    region_alignments = {}
    n_skipped_no_slice = 0
    n_total_regions = len(classification_df)

    for i, row in enumerate(classification_df.itertuples(index=False)):
        if verbose and (i + 1) % 500 == 0:
            print(f"  Slicing regions: {i + 1}/{n_total_regions}")
        uid = row.UniqueID
        if uid not in dedup_by_query:
            continue
        sub = dedup_by_query[uid]

        h_slices = []
        q_slice_canonical = None
        for r in sub.itertuples(index=False):
            sliced = slice_aligned_region(
                qseq_aligned=getattr(r, qseq_col),
                hseq_aligned=getattr(r, hseq_col),
                region_start=row.region_start,
                region_end=row.region_end,
                query_from=int(getattr(r, query_from_col)),
            )
            if sliced is None:
                continue
            q, h = sliced
            if q_slice_canonical is None:
                q_slice_canonical = q
            h_slices.append(h)

        if q_slice_canonical is None or len(h_slices) == 0:
            n_skipped_no_slice += 1
            continue

        key = (uid, int(row.region_start), int(row.region_end), row.region_class)
        region_alignments[key] = {
            "qseq_slice": q_slice_canonical,
            "hseq_slices": h_slices,
            "n_alignments": len(h_slices),
        }

    if verbose:
        print(f"  Built {len(region_alignments):,} region alignments "
              f"({n_skipped_no_slice} regions had no usable slice)")

    return region_alignments


# ════════════════════════════════════════════════════════════════════════════
# Per-region entropy
# ════════════════════════════════════════════════════════════════════════════

def compute_per_region_entropy(
    region_alignments: dict,
    classification_df: pd.DataFrame,
    df_combined: pd.DataFrame,
    min_alignments: int = 10,
    include_query_in_alignment: bool = True,
    exclude_gaps: bool = True,
) -> pd.DataFrame:
    """
    For each region in region_alignments, compute mean Shannon entropy across
    the homolog alignment columns. Returns DataFrame with one row per region.
    """
    # Map UniqueID -> group (positive/negative or pos/neg)
    group_map = (
        df_combined.drop_duplicates("UniqueID")[["UniqueID", "group"]]
        .set_index("UniqueID")["group"]
        .to_dict()
    )
    # Harmonize group labels
    def _norm_group(g):
        if g in ("positive", "pos"):
            return "pos"
        if g in ("negative", "neg"):
            return "neg"
        return g

    rows = []
    for key, info in region_alignments.items():
        uid, rstart, rend, rclass = key
        h_slices = info["hseq_slices"]
        q_slice = info["qseq_slice"]

        if include_query_in_alignment:
            alignment = [q_slice] + h_slices
        else:
            alignment = h_slices

        # Filter by alignment N
        if len(alignment) < min_alignments:
            continue

        # Filter to equal-length sequences (alignment slicing should give
        # equal lengths but defensive check)
        L = len(q_slice)
        alignment = [s for s in alignment if len(s) == L]
        if len(alignment) < min_alignments:
            continue

        ent = positional_entropy(alignment, exclude_gaps=exclude_gaps)
        ent = ent[~np.isnan(ent)]
        if len(ent) == 0:
            continue

        rows.append({
            "UniqueID": uid,
            "region_start": rstart,
            "region_end": rend,
            "region_class": rclass,
            "region_length": rend - rstart + 1,
            "group": _norm_group(group_map.get(uid)),
            "n_alignments": len(alignment),
            "mean_entropy": float(np.mean(ent)),
            "median_entropy": float(np.median(ent)),
            "fraction_invariant": float(np.mean(ent == 0)),
        })

    out = pd.DataFrame(rows)
    print(f"  Computed entropy for {len(out):,} regions "
          f"(filtered to ≥{min_alignments} alignments per region)")
    return out


# ════════════════════════════════════════════════════════════════════════════
# Statistical comparison
# ════════════════════════════════════════════════════════════════════════════

REGION_CLASSES_ORDERED = ["RG_motif", "mIDR", "oIDR", "structured"]


def test_region_class_comparisons(
    entropy_df: pd.DataFrame,
    metric: str = "mean_entropy",
) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney across region classes within each group, plus
    pos-vs-neg within each region class. BH-FDR across all tests.
    """
    rows = []

    # Within-group: pairwise across region classes
    for group in ("pos", "neg"):
        for i, c1 in enumerate(REGION_CLASSES_ORDERED):
            for c2 in REGION_CLASSES_ORDERED[i + 1:]:
                v1 = entropy_df.loc[
                    (entropy_df["group"] == group) &
                    (entropy_df["region_class"] == c1),
                    metric,
                ].dropna()
                v2 = entropy_df.loc[
                    (entropy_df["group"] == group) &
                    (entropy_df["region_class"] == c2),
                    metric,
                ].dropna()
                if len(v1) < 5 or len(v2) < 5:
                    p, u = np.nan, np.nan
                else:
                    u, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                rows.append({
                    "test_type": f"within-{group}",
                    "comparison": f"{c1} vs {c2}",
                    "n_a": int(len(v1)), "n_b": int(len(v2)),
                    "median_a": float(v1.median()) if len(v1) else np.nan,
                    "median_b": float(v2.median()) if len(v2) else np.nan,
                    "u_stat": float(u) if not np.isnan(u) else np.nan,
                    "p_raw": float(p) if not np.isnan(p) else np.nan,
                })

    # Pos vs neg within each region class
    for c in REGION_CLASSES_ORDERED:
        v_pos = entropy_df.loc[
            (entropy_df["group"] == "pos") &
            (entropy_df["region_class"] == c),
            metric,
        ].dropna()
        v_neg = entropy_df.loc[
            (entropy_df["group"] == "neg") &
            (entropy_df["region_class"] == c),
            metric,
        ].dropna()
        if len(v_pos) < 5 or len(v_neg) < 5:
            p, u = np.nan, np.nan
        else:
            u, p = stats.mannwhitneyu(v_pos, v_neg, alternative="two-sided")
        rows.append({
            "test_type": "pos-vs-neg",
            "comparison": c,
            "n_a": int(len(v_pos)), "n_b": int(len(v_neg)),
            "median_a": float(v_pos.median()) if len(v_pos) else np.nan,
            "median_b": float(v_neg.median()) if len(v_neg) else np.nan,
            "u_stat": float(u) if not np.isnan(u) else np.nan,
            "p_raw": float(p) if not np.isnan(p) else np.nan,
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

def plot_region_class_entropy(
    entropy_df: pd.DataFrame,
    test_results: pd.DataFrame,
    metric: str = "mean_entropy",
    metric_label: str = "Mean Shannon entropy (no gaps)",
    dataset: str = "homologs",
    save: bool = True,
) -> plt.Figure:
    """
    8-violin plot: 4 region classes × 2 groups, side by side.
    Includes within-group pairwise stars and pos-vs-neg stars.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Build positions: groups of 2 violins per region class
    n_classes = len(REGION_CLASSES_ORDERED)
    width = 0.36
    spacing = 0.05

    positions_pos = []
    positions_neg = []
    base_xticks = []
    for i, cls in enumerate(REGION_CLASSES_ORDERED):
        center = i * 1.2
        positions_neg.append(center - (width / 2 + spacing / 2))
        positions_pos.append(center + (width / 2 + spacing / 2))
        base_xticks.append(center)

    data_pos = [
        entropy_df.loc[
            (entropy_df["group"] == "pos") &
            (entropy_df["region_class"] == c),
            metric,
        ].dropna().values
        for c in REGION_CLASSES_ORDERED
    ]
    data_neg = [
        entropy_df.loc[
            (entropy_df["group"] == "neg") &
            (entropy_df["region_class"] == c),
            metric,
        ].dropna().values
        for c in REGION_CLASSES_ORDERED
    ]

    # Violins
    for vals, x_pos, color in zip(data_neg, positions_neg,
                                    [GROUP_COLORS["neg"]] * n_classes):
        if len(vals) > 1:
            parts = ax.violinplot(
                [vals], positions=[x_pos], widths=width,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.35)
                pc.set_edgecolor("black"); pc.set_linewidth(0.4)

    for vals, x_pos, color in zip(data_pos, positions_pos,
                                    [GROUP_COLORS["pos"]] * n_classes):
        if len(vals) > 1:
            parts = ax.violinplot(
                [vals], positions=[x_pos], widths=width,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.35)
                pc.set_edgecolor("black"); pc.set_linewidth(0.4)

    # Boxes inside violins
    for vals, x_pos, color in zip(data_neg, positions_neg,
                                    [GROUP_COLORS["neg"]] * n_classes):
        if len(vals) > 0:
            bp = ax.boxplot(
                [vals], positions=[x_pos], widths=width * 0.4,
                showfliers=False, patch_artist=True, zorder=3,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color); patch.set_alpha(0.9); patch.set_edgecolor("black")
            for median in bp["medians"]:
                median.set_color("black"); median.set_linewidth(1.2)

    for vals, x_pos, color in zip(data_pos, positions_pos,
                                    [GROUP_COLORS["pos"]] * n_classes):
        if len(vals) > 0:
            bp = ax.boxplot(
                [vals], positions=[x_pos], widths=width * 0.4,
                showfliers=False, patch_artist=True, zorder=3,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color); patch.set_alpha(0.9); patch.set_edgecolor("black")
            for median in bp["medians"]:
                median.set_color("black"); median.set_linewidth(1.2)

    # Strip plots
    for vals, x_pos, color in zip(data_neg + data_pos,
                                    positions_neg + positions_pos,
                                    [GROUP_COLORS["neg"]] * n_classes +
                                    [GROUP_COLORS["pos"]] * n_classes):
        if len(vals) > 0:
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(
                np.full(len(vals), x_pos) + jitter, vals,
                color=color, alpha=0.4, s=4, edgecolor="none", zorder=2,
            )

    ax.set_xticks(base_xticks)
    ax.set_xticklabels(REGION_CLASSES_ORDERED, fontsize=10)
    ax.set_ylabel(metric_label)
    ax.set_title(
        f"Per-region entropy by class and group ({dataset})",
        fontsize=11,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Annotate pos-vs-neg significance stars above each region class
    pos_neg_results = test_results[test_results["test_type"] == "pos-vs-neg"]
    pos_neg_lookup = pos_neg_results.set_index("comparison")
    ymax = max(
        max(np.percentile(d, 95) if len(d) > 0 else 0 for d in data_pos),
        max(np.percentile(d, 95) if len(d) > 0 else 0 for d in data_neg),
    )
    for i, cls in enumerate(REGION_CLASSES_ORDERED):
        if cls in pos_neg_lookup.index:
            sig = pos_neg_lookup.loc[cls, "sig"]
            if sig and sig != "n.s.":
                ax.text(
                    base_xticks[i], ymax * 1.06, sig,
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                )

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["pos"], alpha=0.7, label="pos"),
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["neg"], alpha=0.7, label="neg"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    plt.tight_layout()
    if save:
        save_figure(fig, "homolog_entropy_by_region_class", dataset=dataset)

    # Summary print
    print(f"\n── Region-class entropy comparisons ({dataset}) ──")
    print(test_results[[
        "test_type", "comparison", "n_a", "n_b",
        "median_a", "median_b", "p_raw", "p_fdr", "sig",
    ]].sort_values(["test_type", "p_fdr"]).to_string(index=False))

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_background_entropy_analysis(
    df_combined: pd.DataFrame,
    classification_df: pd.DataFrame,
    min_alignments: int = 10,
    include_query_in_alignment: bool = True,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    [Homolog-specific]
    End-to-end: build alignment slices for all classified regions,
    compute entropy, run stats, plot.
    """
    print("Step 1/3: Building region alignments by slicing full-protein qseq/hseq...")
    region_alignments = build_region_alignments(
        df_combined, classification_df,
    )

    print("\nStep 2/3: Computing per-region entropy...")
    entropy_df = compute_per_region_entropy(
        region_alignments, classification_df, df_combined,
        min_alignments=min_alignments,
        include_query_in_alignment=include_query_in_alignment,
    )

    print("\nStep 3/3: Statistical tests + plotting...")
    test_results = test_region_class_comparisons(entropy_df)
    plot_region_class_entropy(
        entropy_df, test_results,
        dataset=dataset, save=save,
    )

    return {
        "region_alignments": region_alignments,
        "entropy_df": entropy_df,
        "test_results": test_results,
    }

# Add to src/analysis_visualization/homolog_background_entropy.py

def compute_entropy_relative_to_protein(
    entropy_df: pd.DataFrame,
    whole_protein_entropy_df: pd.DataFrame,
    metric_col: str = "mean_entropy",
    whole_protein_col: str = "mean_entropy_whole_protein",
) -> pd.DataFrame:
    """
    [Homolog-specific]
    Add per-region "delta from whole-protein" entropy.

    delta_from_wp = region_entropy - whole_protein_entropy(host gene)

    Negative delta → region more conserved than its host protein's average.
    """
    # Map UniqueID → whole-protein entropy
    wp_map = (
        whole_protein_entropy_df.set_index("UniqueID")[whole_protein_col]
        .to_dict()
    )

    out = entropy_df.copy()
    out["whole_protein_entropy"] = out["UniqueID"].map(wp_map)
    out["delta_from_whole_protein"] = (
        out[metric_col] - out["whole_protein_entropy"]
    )
    # Also ratio for supplementary view, clipped to avoid inf
    out["ratio_to_whole_protein"] = out[metric_col] / out["whole_protein_entropy"].replace(0, np.nan)

    n_missing = out["whole_protein_entropy"].isna().sum()
    if n_missing:
        print(f"  Warning: {n_missing} regions have no whole-protein entropy "
              f"(likely proteins with <min_homologs). Dropping them.")
        out = out[out["whole_protein_entropy"].notna()]

    return out


def plot_region_class_entropy_relative(
    entropy_df_rel: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
    metric_col: str = "delta_from_whole_protein",
    metric_label: str = "Δ entropy (region − whole protein)",
) -> plt.Figure:
    """
    Same 8-violin plot structure as plot_region_class_entropy, but showing
    the delta from whole-protein entropy instead of absolute entropy.
    Negative delta = region more conserved than its host protein average.
    """
    # Use the same plotting code as before but change the metric column
    test_results = test_region_class_comparisons(
        entropy_df_rel, metric=metric_col,
    )

    fig = plot_region_class_entropy(
        entropy_df_rel, test_results,
        metric=metric_col,
        metric_label=metric_label,
        dataset=dataset + "_relative",
        save=save,
    )
    return fig, test_results