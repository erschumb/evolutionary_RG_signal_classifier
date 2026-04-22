"""
Amino acid substitution matrix analysis.

Save as: src/analysis_visualization/substitution_matrix.py

Computes observed amino acid substitution frequencies from missense variants
and compares pos vs neg groups via log2 odds ratios with FDR-corrected Fisher's
exact tests.

Design choices (as of this refactor):
  - No α smoothing. Cells with insufficient data (pos_count + neg_count < MIN_TOTAL)
    are masked to NaN. This is the honest approach — don't fabricate observations.
  - Log2 OR computed from raw counts via Fisher's exact (not from normalized
    frequencies). Cells with any zero count produce inf/−inf OR values; these
    are also masked.
  - Benjamini-Hochberg FDR correction across all tested (non-masked) cells.
  - 20×20 heatmap with AA-group boxes overlaid (Pos / Neg / Polar / Aromatic /
    Hydrophobic / C-G-P) — not a separate "grouped" plot, but boxes indicate
    group membership for readability.

Applies identically to gnomAD missense and homolog single-position differences.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from src.analysis_visualization.plot_config import (
    GROUP_COLORS, save_figure, significance_stars,
)


# ════════════════════════════════════════════════════════════════════════════
# AA grouping
# ════════════════════════════════════════════════════════════════════════════

AA_GROUPS = {
    "R": "Pos", "K": "Pos", "H": "Pos",
    "D": "Neg", "E": "Neg",
    "S": "Polar", "T": "Polar", "N": "Polar", "Q": "Polar",
    "F": "Aromatic", "W": "Aromatic", "Y": "Aromatic",
    "A": "Hydrophobic", "V": "Hydrophobic", "I": "Hydrophobic",
    "L": "Hydrophobic", "M": "Hydrophobic",
    "G": "C/G/P", "P": "C/G/P", "C": "C/G/P",
}

GROUP_ORDER = ["Aromatic", "C/G/P", "Hydrophobic", "Neg", "Polar", "Pos"]

# Ordered AA list grouped by physicochemical class
ORDERED_AA = []
for _g in GROUP_ORDER:
    ORDERED_AA.extend(sorted([aa for aa, grp in AA_GROUPS.items() if grp == _g]))


def _group_slices(order: list[str], groups: list[str]) -> dict:
    """Return {group_name: (start_idx, end_idx_inclusive)} for drawing boxes."""
    out = {}
    start = 0
    for g in groups:
        aas = [aa for aa in order if AA_GROUPS[aa] == g]
        if not aas:
            continue
        out[g] = (start, start + len(aas) - 1)
        start += len(aas)
    return out


GROUP_SLICES_COL = _group_slices(ORDERED_AA, GROUP_ORDER)
GROUP_SLICES_ROW = _group_slices(ORDERED_AA, GROUP_ORDER[::-1])


# ════════════════════════════════════════════════════════════════════════════
# Colormaps
# ════════════════════════════════════════════════════════════════════════════

def _make_diverging_cmap():
    """Red-white-green diverging colormap for log2 OR plots."""
    reds = plt.cm.Reds_r(np.linspace(0, 1, 128))
    greens = plt.cm.Greens(np.linspace(0, 1, 128))
    center = np.array([[1, 1, 1, 1]])
    colors = np.vstack((reds, center, greens))
    return mcolors.LinearSegmentedColormap.from_list("RedWhiteGreen", colors)


_DIVERGING_CMAP = _make_diverging_cmap()

_CMAP_DICT = {
    "pos": "Greens",
    "neg": "Reds",
    "enrichment": _DIVERGING_CMAP,
}


# ════════════════════════════════════════════════════════════════════════════
# Count matrix construction
# ════════════════════════════════════════════════════════════════════════════

def compute_substitution_counts(
    df: pd.DataFrame,
    before_col: str = "before_aa",
    after_col: str = "after_aa",
    consequence_col: str = "Consequence",
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    Build a 20×20 substitution count matrix (AA_from × AA_to) from a dataframe
    of missense variants. Returns a DataFrame with AA rows and columns, reindexed
    by ORDERED_AA.
    """
    sub = df[
        df[consequence_col].fillna("").str.contains("missense_variant") &
        df[before_col].notna() & df[after_col].notna() &
        (df[before_col].str.len() == 1) & (df[after_col].str.len() == 1)
    ]
    counts = (
        sub.groupby([before_col, after_col])
           .size()
           .unstack(fill_value=0)
    )
    # Reindex to full 20×20 with consistent ordering
    return counts.reindex(index=ORDERED_AA, columns=ORDERED_AA, fill_value=0)


def row_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Row-normalize a count matrix to substitution frequencies. Rows with zero
    total counts remain all-zero (no smoothing applied).
    """
    row_sums = counts.sum(axis=1)
    # Avoid divide-by-zero: rows with zero total become zero (not NaN)
    return counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0)


# ════════════════════════════════════════════════════════════════════════════
# Enrichment + significance testing
# ════════════════════════════════════════════════════════════════════════════

def compute_enrichment(
    counts_pos: pd.DataFrame,
    counts_neg: pd.DataFrame,
    min_total: int = 5,
) -> dict:
    """
    [Dataset-agnostic]
    Per-cell Fisher's exact test between pos and neg substitution counts, with
    FDR correction across all tested cells.

    Masking strategy: cells with (pos_count + neg_count) < min_total OR with
    any zero in the 2×2 contingency table are masked to NaN. No α smoothing.

    Returns a dict with:
        freq_pos, freq_neg     — row-normalized frequency matrices (20×20)
        log2_or                — log2 odds ratio matrix (20×20, masked cells NaN)
        pval                   — raw Fisher p-values (20×20, masked cells NaN)
        fdr                    — BH-corrected p-values (20×20, masked cells NaN)
        counts_pos, counts_neg — original count matrices (for annotation)
        n_tested               — number of cells that passed masking
    """
    # Align matrices
    counts_pos = counts_pos.reindex(index=ORDERED_AA, columns=ORDERED_AA, fill_value=0)
    counts_neg = counts_neg.reindex(index=ORDERED_AA, columns=ORDERED_AA, fill_value=0)

    freq_pos = row_normalize(counts_pos)
    freq_neg = row_normalize(counts_neg)

    # Per-cell log2 OR and raw Fisher p
    log2_or = pd.DataFrame(np.nan, index=ORDERED_AA, columns=ORDERED_AA, dtype=float)
    pval = pd.DataFrame(np.nan, index=ORDERED_AA, columns=ORDERED_AA, dtype=float)

    tested_cells = []  # (aa_from, aa_to, p-value) for FDR
    for aa_from in ORDERED_AA:
        row_total_pos = counts_pos.loc[aa_from].sum()
        row_total_neg = counts_neg.loc[aa_from].sum()
        for aa_to in ORDERED_AA:
            if aa_from == aa_to:
                continue  # diagonal = silent-looking, not a substitution
            pos_c = int(counts_pos.loc[aa_from, aa_to])
            neg_c = int(counts_neg.loc[aa_from, aa_to])

            # Mask: insufficient total observations for this cell
            if pos_c + neg_c < min_total:
                continue

            pos_out = row_total_pos - pos_c
            neg_out = row_total_neg - neg_c

            # Strict mask: any zero in the 2×2 → OR undefined/infinite
            if 0 in (pos_c, pos_out, neg_c, neg_out):
                # We can still compute the Fisher p-value (handles zeros)
                _, p = fisher_exact([[pos_c, pos_out], [neg_c, neg_out]])
                pval.loc[aa_from, aa_to] = p
                tested_cells.append((aa_from, aa_to, p))
                # OR stays NaN (masked)
                continue

            odds, p = fisher_exact([[pos_c, pos_out], [neg_c, neg_out]])
            log2_or.loc[aa_from, aa_to] = np.log2(odds)
            pval.loc[aa_from, aa_to] = p
            tested_cells.append((aa_from, aa_to, p))

    # BH FDR correction across all tested cells
    fdr = pd.DataFrame(np.nan, index=ORDERED_AA, columns=ORDERED_AA, dtype=float)
    if tested_cells:
        raw_ps = [c[2] for c in tested_cells]
        _, corrected, _, _ = multipletests(raw_ps, method="fdr_bh")
        for (aa_from, aa_to, _p), p_fdr in zip(tested_cells, corrected):
            fdr.loc[aa_from, aa_to] = p_fdr

    return {
        "freq_pos": freq_pos,
        "freq_neg": freq_neg,
        "log2_or": log2_or,
        "pval": pval,
        "fdr": fdr,
        "counts_pos": counts_pos,
        "counts_neg": counts_neg,
        "n_tested": len(tested_cells),
    }


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

def _add_group_boxes(ax, shape: tuple[int, int]) -> None:
    """Overlay AA-group boxes on both axes."""
    n_rows, n_cols = shape
    for g, (start, end) in GROUP_SLICES_COL.items():
        rect = patches.Rectangle(
            (start, 0), width=(end - start + 1), height=n_rows,
            fill=False, edgecolor="black", linewidth=1.2,
        )
        ax.add_patch(rect)
    for g, (start, end) in GROUP_SLICES_ROW.items():
        rect = patches.Rectangle(
            (0, start), width=n_cols, height=(end - start + 1),
            fill=False, edgecolor="black", linewidth=1.2,
        )
        ax.add_patch(rect)


def _add_significance_stars(ax, fdr_df: pd.DataFrame, row_order: list[str]) -> None:
    """Overlay *, **, *** on enrichment heatmap based on FDR-corrected p-values."""
    for i, aa_from in enumerate(row_order):
        for j, aa_to in enumerate(fdr_df.columns):
            p = fdr_df.loc[aa_from, aa_to]
            if pd.isna(p):
                continue
            stars = significance_stars(p)
            if stars != "n.s.":
                ax.text(
                    j + 0.5, i + 0.5, stars,
                    color="black", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                )


def plot_substitution_matrix(
    enrichment: dict,
    dataset: str = "gnomad",
    save: bool = True,
    vmax_freq: float | None = None,
    vmax_or: float | None = None,
    show_counts: bool = True,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Three-panel heatmap:
      1. Positive group substitution frequencies
      2. Negative group substitution frequencies
      3. log2(pos/neg) enrichment with FDR significance stars

    Rows plotted in reverse AA order (anti-diagonal) for readability — matches
    the convention from the original code.
    """
    # For visualization, rows are reversed (anti-diagonal)
    row_order = ORDERED_AA[::-1]
    freq_pos = enrichment["freq_pos"].reindex(index=row_order, columns=ORDERED_AA)
    freq_neg = enrichment["freq_neg"].reindex(index=row_order, columns=ORDERED_AA)
    log2_or  = enrichment["log2_or"].reindex(index=row_order, columns=ORDERED_AA)
    fdr      = enrichment["fdr"].reindex(index=row_order, columns=ORDERED_AA)
    pval      = enrichment["pval"].reindex(index=row_order, columns=ORDERED_AA)
    counts_pos = enrichment["counts_pos"].reindex(index=row_order, columns=ORDERED_AA)
    counts_neg = enrichment["counts_neg"].reindex(index=row_order, columns=ORDERED_AA)

    # Determine color scales
    if vmax_freq is None:
        vmax_freq = max(freq_pos.max().max(), freq_neg.max().max())
    if vmax_or is None:
        vmax_or = np.nanmax(np.abs(log2_or.values))
        if pd.isna(vmax_or) or vmax_or == 0:
            vmax_or = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel 1: Pos frequencies ────────────────────────────────────────────
    sns.heatmap(
        freq_pos, ax=axes[0],
        cmap=_CMAP_DICT["pos"],
        vmin=0, vmax=vmax_freq,
        annot=counts_pos if show_counts else False,
        fmt="d",
        annot_kws={"size": 7},
        cbar_kws={"label": "Row-normalized frequency"},
    )
    axes[0].set_title("Positive group")
    axes[0].set_xlabel("AA to")
    axes[0].set_ylabel("AA from")
    _add_group_boxes(axes[0], freq_pos.shape)

    # ── Panel 2: Neg frequencies ────────────────────────────────────────────
    sns.heatmap(
        freq_neg, ax=axes[1],
        cmap=_CMAP_DICT["neg"],
        vmin=0, vmax=vmax_freq,
        annot=counts_neg if show_counts else False,
        fmt="d",
        annot_kws={"size": 7},
        cbar_kws={"label": "Row-normalized frequency"},
    )
    axes[1].set_title("Negative group")
    axes[1].set_xlabel("AA to")
    axes[1].set_ylabel("")
    _add_group_boxes(axes[1], freq_neg.shape)

    # ── Panel 3: Enrichment (log2 OR) ───────────────────────────────────────
    cmap = _CMAP_DICT["enrichment"].copy()
    cmap.set_bad(color="#DDDDDD")   # masked cells = light gray

    sns.heatmap(
        log2_or, ax=axes[2],
        cmap=cmap, center=0,
        vmin=-vmax_or, vmax=vmax_or,
        cbar_kws={"label": "log₂(OR)  pos vs neg"},
    )
    axes[2].set_title("Enrichment (significance = FDR)")
    axes[2].set_xlabel("AA to")
    axes[2].set_ylabel("")
    _add_group_boxes(axes[2], log2_or.shape)
    _add_significance_stars(axes[2], fdr, row_order)

    # Super title
    title = f"Amino acid substitution matrix ({dataset})"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=13, y=1.00)

    plt.tight_layout()
    if save:
        save_figure(fig, "substitution_matrix", dataset=dataset)

    # Print summary
    tested = enrichment["n_tested"]
    sig_mask = enrichment["fdr"] < 0.05
    n_sig = int(sig_mask.sum().sum())
    print(f"\n── Substitution matrix ({dataset}) ──")
    print(f"  Cells tested (passed min_total filter): {tested}")
    print(f"  Cells significant at FDR < 0.05: {n_sig}")
    if n_sig > 0:
        # Print top significant transitions by |log2 OR|
        sig_cells = []
        for aa_from in ORDERED_AA:
            for aa_to in ORDERED_AA:
                if aa_from == aa_to:
                    continue
                p = enrichment["fdr"].loc[aa_from, aa_to]
                if pd.notna(p) and p < 0.05:
                    lor = enrichment["log2_or"].loc[aa_from, aa_to]
                    if pd.notna(lor):
                        sig_cells.append({
                            "from": aa_from, "to": aa_to,
                            "log2_or": lor, "fdr": p,
                            "pos_count": int(enrichment["counts_pos"].loc[aa_from, aa_to]),
                            "neg_count": int(enrichment["counts_neg"].loc[aa_from, aa_to]),
                        })
        if sig_cells:
            sig_df = pd.DataFrame(sig_cells).reindex(
                columns=["from", "to", "log2_or", "fdr", "pos_count", "neg_count"]
            )
            sig_df = sig_df.reindex(sig_df["log2_or"].abs().sort_values(ascending=False).index)
            print("\n  Top significant transitions (by |log2 OR|):")
            print(sig_df.head(15).to_string(index=False))

    return fig


# ════════════════════════════════════════════════════════════════════════════
# One-call convenience wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_substitution_analysis(
    df: pd.DataFrame,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    min_total: int = 5,
    dataset: str = "gnomad",
    save: bool = True,
    **plot_kwargs,
) -> dict:
    """
    [Dataset-agnostic]
    End-to-end: split df by group, build count matrices, compute enrichment,
    plot. Returns the enrichment dict (for classifier feature extraction).
    """
    df_pos = df[df[group_col] == pos_label]
    df_neg = df[df[group_col] == neg_label]

    counts_pos = compute_substitution_counts(df_pos)
    counts_neg = compute_substitution_counts(df_neg)

    enrichment = compute_enrichment(counts_pos, counts_neg, min_total=min_total)

    plot_substitution_matrix(enrichment, dataset=dataset, save=save, **plot_kwargs)

    return enrichment