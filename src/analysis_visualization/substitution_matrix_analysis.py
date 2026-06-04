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
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import binomtest
from matplotlib.colors import LogNorm

from src.analysis_visualization.plot_config import (
    GROUP_COLORS, save_figure, significance_stars,
)

# The enumeration function lives in rg_analysis.py
from src.analysis_visualization.rg_analysis import (
    enumerate_single_nt_substitutions,
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

 
 
# ════════════════════════════════════════════════════════════════════════════
# AF-filtered single analysis
# ════════════════════════════════════════════════════════════════════════════
 
def run_substitution_analysis_af_filtered(
    df: pd.DataFrame,
    af_column: str = "AF_joint",
    af_min: float | None = None,
    af_max: float | None = None,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    min_total: int = 5,
) -> dict:
    """
    [gnomAD-specific]
    Run the substitution enrichment analysis on a subset of variants filtered
    by allele frequency. Returns the full enrichment dict (same structure as
    compute_enrichment), plus the number of variants retained per group.
 
    Set af_min / af_max to define the AF window:
      - Rare only:    af_min=None, af_max=1e-4
      - Common only:  af_min=1e-3, af_max=None
      - Both:         af_min=1e-4, af_max=1e-3
    """
    df = df.copy()
    # Drop variants without AF
    df = df[df[af_column].notna()]
    if af_min is not None:
        df = df[df[af_column] >= af_min]
    if af_max is not None:
        df = df[df[af_column] < af_max]
 
    df_pos = df[df[group_col] == pos_label]
    df_neg = df[df[group_col] == neg_label]
 
    counts_pos = compute_substitution_counts(df_pos)
    counts_neg = compute_substitution_counts(df_neg)
 
    enrichment = compute_enrichment(counts_pos, counts_neg, min_total=min_total)
 
    # Add metadata about the filter
    enrichment["af_min"] = af_min
    enrichment["af_max"] = af_max
    enrichment["n_pos_variants"] = int(len(df_pos))
    enrichment["n_neg_variants"] = int(len(df_neg))
 
    return enrichment
 
 
# ════════════════════════════════════════════════════════════════════════════
# Side-by-side comparison plot
# ════════════════════════════════════════════════════════════════════════════
 
def _af_label(af_min: float | None, af_max: float | None) -> str:
    """Human-readable label for an AF window."""
    if af_min is None and af_max is None:
        return "all AF"
    if af_min is None:
        return f"AF < {af_max:.0e}"
    if af_max is None:
        return f"AF ≥ {af_min:.0e}"
    return f"{af_min:.0e} ≤ AF < {af_max:.0e}"
 
 
def _heatmap_enrichment_panel(
    ax, log2_or: pd.DataFrame, fdr: pd.DataFrame, row_order: list[str],
    vmax_or: float, title: str, show_ylabels: bool = True,
) -> None:
    """One enrichment subplot — used for both rare and common in the comparison."""
    cmap = _CMAP_DICT["enrichment"].copy()
    cmap.set_bad(color="#DDDDDD")
 
    sns.heatmap(
        log2_or.reindex(index=row_order, columns=ORDERED_AA),
        ax=ax, cmap=cmap, center=0, vmin=-vmax_or, vmax=vmax_or,
        cbar_kws={"label": "log₂(OR)  pos vs neg", "shrink": 0.7},
        yticklabels=show_ylabels,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("AA to")
    ax.set_ylabel("AA from" if show_ylabels else "")
    _add_group_boxes(ax, log2_or.shape)
    _add_significance_stars(
        ax, fdr.reindex(index=row_order, columns=ORDERED_AA), row_order,
    )
 
 
def plot_af_comparison_matrices(
    df: pd.DataFrame,
    af_rare_max: float = 1e-4,
    af_common_min: float = 1e-3,
    af_column: str = "AF_joint",
    dataset: str = "gnomad",
    min_total: int = 5,
    save: bool = True,
) -> dict:
    """
    [gnomAD-specific]
    Compute two AF-filtered substitution matrices (rare & common) and plot
    their enrichment panels side by side for comparison.
 
    Returns dict with both enrichment results.
    """
    # Rare
    rare = run_substitution_analysis_af_filtered(
        df, af_column=af_column, af_min=None, af_max=af_rare_max,
        min_total=min_total,
    )
    # Common
    common = run_substitution_analysis_af_filtered(
        df, af_column=af_column, af_min=af_common_min, af_max=None,
        min_total=min_total,
    )
 
    # Shared color scale across the two enrichment panels for comparability
    vmax_rare = np.nanmax(np.abs(rare["log2_or"].values))
    vmax_common = np.nanmax(np.abs(common["log2_or"].values))
    vmax_shared = max(
        (vmax_rare if pd.notna(vmax_rare) else 0),
        (vmax_common if pd.notna(vmax_common) else 0),
        1.0,  # minimum so fully-null matrices still render
    )
 
    # Shared row ordering (anti-diagonal)
    row_order = ORDERED_AA[::-1]
 
    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
 
    _heatmap_enrichment_panel(
        axes[0], rare["log2_or"], rare["fdr"], row_order, vmax_shared,
        title=(
            f"Rare variants ({_af_label(None, af_rare_max)})\n"
            f"n_pos = {rare['n_pos_variants']:,}, "
            f"n_neg = {rare['n_neg_variants']:,}, "
            f"FDR<0.05: {int((rare['fdr'] < 0.05).sum().sum())}"
        ),
        show_ylabels=True,
    )
 
    _heatmap_enrichment_panel(
        axes[1], common["log2_or"], common["fdr"], row_order, vmax_shared,
        title=(
            f"Common variants ({_af_label(af_common_min, None)})\n"
            f"n_pos = {common['n_pos_variants']:,}, "
            f"n_neg = {common['n_neg_variants']:,}, "
            f"FDR<0.05: {int((common['fdr'] < 0.05).sum().sum())}"
        ),
        show_ylabels=False,
    )
 
    fig.suptitle(
        f"AA substitution enrichment: rare vs common variants ({dataset})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "substitution_matrix_af_comparison", dataset=dataset)
 
    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n── AF comparison substitution matrices ({dataset}) ──")
    for label, res in [("Rare", rare), ("Common", common)]:
        print(f"\n  {label} ({_af_label(res['af_min'], res['af_max'])}):")
        print(f"    Variants: pos = {res['n_pos_variants']:,}, "
              f"neg = {res['n_neg_variants']:,}")
        print(f"    Cells tested: {res['n_tested']}")
        n_sig = int((res['fdr'] < 0.05).sum().sum())
        print(f"    Cells significant at FDR < 0.05: {n_sig}")
        if n_sig > 0:
            sig_cells = []
            for aa_from in ORDERED_AA:
                for aa_to in ORDERED_AA:
                    if aa_from == aa_to:
                        continue
                    p = res["fdr"].loc[aa_from, aa_to]
                    if pd.notna(p) and p < 0.05:
                        lor = res["log2_or"].loc[aa_from, aa_to]
                        if pd.notna(lor):
                            sig_cells.append({
                                "from": aa_from, "to": aa_to,
                                "log2_or": lor, "fdr": p,
                                "pos": int(res["counts_pos"].loc[aa_from, aa_to]),
                                "neg": int(res["counts_neg"].loc[aa_from, aa_to]),
                            })
            if sig_cells:
                sig_df = pd.DataFrame(sig_cells)
                sig_df = sig_df.reindex(
                    sig_df["log2_or"].abs().sort_values(ascending=False).index
                )
                print(f"    Top significant transitions:")
                print(sig_df.head(10).to_string(index=False))
 
    return {"rare": rare, "common": common}
 
 
# ════════════════════════════════════════════════════════════════════════════
# Build the expected substitution matrix per group (composition baseline)
# ════════════════════════════════════════════════════════════════════════════

def compute_expected_substitution_counts(
    region_by_id: dict,
    group_label: str,
    observed_total: int,
    rates: dict | None = None,      # <-- NEW
    cpg_filter: str = "all",      # "all" | "cpg" | "noncpg"   <-- NEW
) -> pd.DataFrame:
    """
    Aggregate possible single-nt missense substitutions across a group's regions,
    scaled to the observed total.

    rates=None      -> each possible substitution weighted +1 (composition null)
    rates=<dict>    -> each weighted by its trinucleotide mutation rate
                       (mutability null; the CpG-aware correction)
    """
    matrix = pd.DataFrame(0.0, index=ORDERED_AA, columns=ORDERED_AA)
    total_possible = 0.0
    n_skipped_edge = 0

    for rid, region in region_by_id.items():
        if region.get("group") != group_label:
            continue
        dna = region.get("dna", "")
        prot = region.get("prot_seq", "")
        if not dna or not prot or len(dna) != 3 * len(prot):
            continue
        enum_df = enumerate_single_nt_substitutions(dna, prot)
        missense_only = enum_df[enum_df["consequence"] == "missense"]
        for row in missense_only.itertuples(index=False):
            if row.aa_from not in ORDERED_AA or row.aa_to not in ORDERED_AA:
                continue
            # --- CpG stratification ---
            if cpg_filter != "all":
                is_cpg = is_cpg_transition(row.context, row.alt_base)
                if cpg_filter == "cpg" and not is_cpg:
                    continue
                if cpg_filter == "noncpg" and is_cpg:
                    continue
            # --- existing rate weighting ---
            if rates is None:
                w = 1.0
            else:
                if row.context is None:
                    continue
                w = rates.get((row.context, row.alt_base))
                if w is None:
                    continue
            matrix.loc[row.aa_from, row.aa_to] += w
            total_possible += w

    if rates is not None and n_skipped_edge:
        # 2 positions per region lose context; negligible but worth tracking
        pass
    if total_possible == 0:
        return matrix
    return matrix * (observed_total / total_possible)

# ════════════════════════════════════════════════════════════════════════════
# Build observed-vs-expected enrichment ratios per group
# ════════════════════════════════════════════════════════════════════════════



def binom_oe_test(obs_pos, obs_neg, exp_pos, exp_neg):
    """
    Conditional binomial test of whether a cell's observed pos/neg split matches
    its mutational-opportunity split. Null p0 = exp_pos/(exp_pos+exp_neg), where
    exp_* are the (rate-weighted, group-total-scaled) expected counts.
    Tests the SAME quantity the obs/exp effect size shows, unlike the raw-count
    Fisher test it replaces.
    """
    if exp_pos <= 0 or exp_neg <= 0:
        return np.nan
    T = int(round(obs_pos + obs_neg))
    if T == 0:
        return np.nan
    p0 = min(max(exp_pos / (exp_pos + exp_neg), 1e-12), 1 - 1e-12)
    return binomtest(int(round(obs_pos)), T, p0).pvalue

def compute_composition_normalized_enrichment(
    df: pd.DataFrame,
    region_by_id: dict,
    min_total: int = 5,
    rates: dict | None = None,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    cpg_filter: str = "all",      # "all" | "cpg" | "noncpg"
) -> dict:
    """
    [Dataset-agnostic]
    Compute observed vs expected (from codon composition) substitution counts
    per group, then compare the ratios between groups.
 
    Returns dict with:
        obs_pos, obs_neg         — observed 20×20 matrices
        exp_pos, exp_neg         — expected 20×20 matrices (from enumeration)
        ratio_pos, ratio_neg     — observed/expected per cell (NaN if exp=0)
        log2_ratio_diff          — log2(ratio_pos / ratio_neg): the selection signal
        pval, fdr                — Fisher p-values and BH-FDR, per cell
        n_tested
    """
    if cpg_filter == "cpg":
        df = df[df["cpg"]]
    elif cpg_filter == "noncpg":
        df = df[~df["cpg"]]
    df_pos = df[df[group_col] == pos_label]
    df_neg = df[df[group_col] == neg_label]
 
    obs_pos = compute_substitution_counts(df_pos)
    obs_neg = compute_substitution_counts(df_neg)
 
    exp_pos = compute_expected_substitution_counts(region_by_id, pos_label,
            observed_total=int(obs_pos.values.sum()), rates=rates, cpg_filter=cpg_filter)
    exp_neg = compute_expected_substitution_counts(region_by_id, neg_label,
            observed_total=int(obs_neg.values.sum()), rates=rates, cpg_filter=cpg_filter)
 
    # Enrichment ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_pos = obs_pos.values / exp_pos.values
        ratio_neg = obs_neg.values / exp_neg.values
    ratio_pos = pd.DataFrame(ratio_pos, index=ORDERED_AA, columns=ORDERED_AA)
    ratio_neg = pd.DataFrame(ratio_neg, index=ORDERED_AA, columns=ORDERED_AA)
    ratio_pos = ratio_pos.where(np.isfinite(ratio_pos))
    ratio_neg = ratio_neg.where(np.isfinite(ratio_neg))
 
    log2_ratio_diff = pd.DataFrame(
        np.nan, index=ORDERED_AA, columns=ORDERED_AA, dtype=float,
    )
    pval = pd.DataFrame(
        np.nan, index=ORDERED_AA, columns=ORDERED_AA, dtype=float,
    )
 
    tested_cells = []
 
    for aa_from in ORDERED_AA:
        for aa_to in ORDERED_AA:
            if aa_from == aa_to:
                continue
            pos_c = int(obs_pos.loc[aa_from, aa_to])
            neg_c = int(obs_neg.loc[aa_from, aa_to])
            if pos_c + neg_c < min_total:
                continue
 
            # # Fisher p-value from raw observed counts (same as before)
            # pos_total_row = int(obs_pos.loc[aa_from].sum())
            # neg_total_row = int(obs_neg.loc[aa_from].sum())
            # pos_out = pos_total_row - pos_c
            # neg_out = neg_total_row - neg_c
            # _, p = fisher_exact([[pos_c, pos_out], [neg_c, neg_out]])

            # obs/exp binomial test: does the pos/neg split match opportunity?
            p = binom_oe_test(
                pos_c, neg_c,
                exp_pos.loc[aa_from, aa_to], exp_neg.loc[aa_from, aa_to],
            )
            if pd.isna(p):
                continue

            # Composition-normalized effect size
            rp = ratio_pos.loc[aa_from, aa_to]
            rn = ratio_neg.loc[aa_from, aa_to]
            if pd.notna(rp) and pd.notna(rn) and rp > 0 and rn > 0:
                log2_ratio_diff.loc[aa_from, aa_to] = np.log2(rp / rn)
 
            pval.loc[aa_from, aa_to] = p
            tested_cells.append((aa_from, aa_to, p))
 
    # FDR correction
    fdr = pd.DataFrame(
        np.nan, index=ORDERED_AA, columns=ORDERED_AA, dtype=float,
    )
    if tested_cells:
        raw_ps = [c[2] for c in tested_cells]
        _, corrected, _, _ = multipletests(raw_ps, method="fdr_bh")
        for (aa_from, aa_to, _p), p_fdr in zip(tested_cells, corrected):
            fdr.loc[aa_from, aa_to] = p_fdr
 
    return {
        "obs_pos": obs_pos,
        "obs_neg": obs_neg,
        "exp_pos": exp_pos,
        "exp_neg": exp_neg,
        "ratio_pos": ratio_pos,
        "ratio_neg": ratio_neg,
        "log2_ratio_diff": log2_ratio_diff,
        "pval": pval,
        "fdr": fdr,
        "n_tested": len(tested_cells),
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# 3-panel plot: pos obs/exp | neg obs/exp | log2(pos_ratio / neg_ratio)
# ════════════════════════════════════════════════════════════════════════════
def plot_composition_normalized_matrix(
    result: dict,
    dataset: str = "gnomad",
    save: bool = True,
    vmax_ratio: float | None = None,
    vmax_diff: float | None = None,
    save_table: bool = False,                 # <-- ADD
    save_table_path: str | None = None,            # <-- ADD (default derived from dataset)
    ) -> plt.Figure:
    """
    Three-panel heatmap:
      1. Pos: log2(observed/expected)
      2. Neg: log2(observed/expected)
      3. log2(ratio_pos / ratio_neg) — composition-controlled group difference
         with FDR stars.
    """
    row_order = ORDERED_AA[::-1]

    # Replace 0 with NaN before log2 to avoid -inf
    ratio_pos = result["ratio_pos"].replace(0, np.nan)
    ratio_neg = result["ratio_neg"].replace(0, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        log2_ratio_pos = np.log2(ratio_pos.reindex(index=row_order, columns=ORDERED_AA))
        log2_ratio_neg = np.log2(ratio_neg.reindex(index=row_order, columns=ORDERED_AA))

    # Replace any remaining inf with NaN just to be safe
    log2_ratio_pos = log2_ratio_pos.replace([np.inf, -np.inf], np.nan)
    log2_ratio_neg = log2_ratio_neg.replace([np.inf, -np.inf], np.nan)

    log2_diff = result["log2_ratio_diff"].reindex(
        index=row_order, columns=ORDERED_AA,
    ).replace([np.inf, -np.inf], np.nan)
    fdr = result["fdr"].reindex(index=row_order, columns=ORDERED_AA)

    # Color scales — now computed on clean matrices
    if vmax_ratio is None:
        finite_pos_vals = log2_ratio_pos.values[np.isfinite(log2_ratio_pos.values)]
        finite_neg_vals = log2_ratio_neg.values[np.isfinite(log2_ratio_neg.values)]
        if len(finite_pos_vals) > 0 or len(finite_neg_vals) > 0:
            vmax_ratio = max(
                np.max(np.abs(finite_pos_vals)) if len(finite_pos_vals) > 0 else 0,
                np.max(np.abs(finite_neg_vals)) if len(finite_neg_vals) > 0 else 0,
                1.0,
            )
        else:
            vmax_ratio = 1.0

    if vmax_diff is None:
        finite_diff = log2_diff.values[np.isfinite(log2_diff.values)]
        if len(finite_diff) > 0:
            vmax_diff = np.max(np.abs(finite_diff))
            if vmax_diff == 0:
                vmax_diff = 1.0
        else:
            vmax_diff = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    cmap = _CMAP_DICT["enrichment"].copy()
    cmap.set_bad(color="#DDDDDD")

    # Panel 1
    sns.heatmap(
        log2_ratio_pos, ax=axes[0],
        cmap=cmap, center=0, vmin=-vmax_ratio, vmax=vmax_ratio,
        cbar_kws={"label": "log₂(observed / expected)"},
    )
    axes[0].set_title("Pos: deviation from composition null")
    axes[0].set_xlabel("AA to")
    axes[0].set_ylabel("AA from")
    _add_group_boxes(axes[0], log2_ratio_pos.shape)

    # Panel 2
    sns.heatmap(
        log2_ratio_neg, ax=axes[1],
        cmap=cmap, center=0, vmin=-vmax_ratio, vmax=vmax_ratio,
        cbar_kws={"label": "log₂(observed / expected)"},
    )
    axes[1].set_title("Neg: deviation from composition null")
    axes[1].set_xlabel("AA to")
    axes[1].set_ylabel("")
    _add_group_boxes(axes[1], log2_ratio_neg.shape)

    # Panel 3
    sns.heatmap(
        log2_diff, ax=axes[2],
        cmap=cmap, center=0, vmin=-vmax_diff, vmax=vmax_diff,
        cbar_kws={"label": "log₂(pos_ratio / neg_ratio)"},
    )
    axes[2].set_title("Composition-controlled group difference\n(significance = FDR)")
    axes[2].set_xlabel("AA to")
    axes[2].set_ylabel("")
    _add_group_boxes(axes[2], log2_diff.shape)
    _add_significance_stars(axes[2], fdr, row_order)

    fig.suptitle(
        f"Composition-normalized substitution matrix ({dataset})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "substitution_matrix_composition_normalized",
                    dataset=dataset)

    table = result_to_table(result) 
    # ── Optional CSV of ALL substitutions ───────────────────────────────────
    if save_table and not table.empty:
        path = save_table_path or f"substitution_matrix_composition_normalized_{dataset}.csv"
        table.to_csv(path, index=False)
        print(f"  Saved full substitution table ({len(table)} cells) -> {path}")

    # ── Printed summary (top significant cells) ─────────────────────────────
    n_sig = int(table["significant"].sum()) if not table.empty else 0
    print(f"\n── Composition-normalized substitution matrix ({dataset}) ──")
    print(f"  Cells tested: {result['n_tested']}")
    print(f"  Cells significant at FDR < 0.05: {n_sig}")
    if n_sig > 0:
        sig_df = table[table["significant"]].drop(columns=["significant", "pval"])
        print("\n  Top composition-controlled group differences:")
        print(sig_df.head(15).to_string(index=False))

    return table, fig
 

def result_to_table(result: dict) -> pd.DataFrame:
    """Melt the enrichment result dict into the tidy per-substitution table
    (from/to/obs_pos/obs_neg/exp_pos/exp_neg/ratio_pos/ratio_neg/log2_ratio_diff/pval/fdr/significant).
    This is exactly the table build_score_table consumes."""
    # ── Build the full results table (all tested cells) ─────────────────────
    rows = []
    for aa_from in ORDERED_AA:
        for aa_to in ORDERED_AA:
            if aa_from == aa_to:
                continue
            p = result["fdr"].loc[aa_from, aa_to]
            if pd.isna(p):              # not tested (failed min_total / no opportunity)
                continue
            rows.append({
                "from": aa_from, "to": aa_to,
                "from_group": AA_GROUPS[aa_from], "to_group": AA_GROUPS[aa_to],
                "log2_ratio_diff": result["log2_ratio_diff"].loc[aa_from, aa_to],
                "pval": result["pval"].loc[aa_from, aa_to],
                "fdr": p,
                "significant": bool(p < 0.05),
                "obs_pos": int(result["obs_pos"].loc[aa_from, aa_to]),
                "obs_neg": int(result["obs_neg"].loc[aa_from, aa_to]),
                "exp_pos": float(result["exp_pos"].loc[aa_from, aa_to]),
                "exp_neg": float(result["exp_neg"].loc[aa_from, aa_to]),
                "ratio_pos": float(result["ratio_pos"].loc[aa_from, aa_to]),
                "ratio_neg": float(result["ratio_neg"].loc[aa_from, aa_to]),
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.reindex(
            table["log2_ratio_diff"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    # print(table)
    return table


# ════════════════════════════════════════════════════════════════════════════
# One-call wrapper
# ════════════════════════════════════════════════════════════════════════════
 
def run_composition_normalized_analysis(
    df: pd.DataFrame,
    region_by_id: dict,
    min_total: int = 5,
    dataset: str = "gnomad",
    save: bool = True,
) -> dict:
    """
    [Dataset-agnostic]
    Full pipeline: compute observed and expected substitution matrices,
    per-cell Fisher tests, composition-normalized enrichment plot.
    """
    result = compute_composition_normalized_enrichment(
        df, region_by_id, min_total=min_total,
    )
    plot_composition_normalized_matrix(result, dataset=dataset, save_table=save)
    return result
 


 
# ════════════════════════════════════════════════════════════════════════════
# Codon table and hierarchical ordering
# ════════════════════════════════════════════════════════════════════════════
 
_CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F',
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    'TAT': 'Y', 'TAC': 'Y',
    'TGT': 'C', 'TGC': 'C',
    'TGG': 'W',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H',
    'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}
 
# Build the ordered codon list: AA physchem group → AA → codon (sorted)
ORDERED_CODONS = []
_CODON_AA_GROUP = []  # parallel: (codon, aa, aa_physchem_group)
for physchem_group in GROUP_ORDER:
    # AAs in this physchem group, sorted (matches ORDERED_AA logic)
    aas_in_group = sorted([aa for aa, g in AA_GROUPS.items() if g == physchem_group])
    for aa in aas_in_group:
        # Codons for this AA, sorted alphabetically for stability
        codons_for_aa = sorted([c for c, a in _CODON_TABLE.items() if a == aa])
        for codon in codons_for_aa:
            ORDERED_CODONS.append(codon)
            _CODON_AA_GROUP.append((codon, aa, physchem_group))
 
CODON_COUNT = len(ORDERED_CODONS)  # should be 61 (excludes stops)
 
 
def _codon_group_slices() -> dict:
    """
    Return slices for hierarchical boxes on the codon (row) axis.
    Two levels:
      - physchem group (outer box)
      - AA (inner box)
    """
    physchem_slices = {}
    aa_slices = {}
 
    start = 0
    current_physchem = None
    physchem_start = 0
    current_aa = None
    aa_start = 0
 
    for i, (codon, aa, group) in enumerate(_CODON_AA_GROUP):
        if group != current_physchem:
            if current_physchem is not None:
                physchem_slices[current_physchem] = (physchem_start, i - 1)
            current_physchem = group
            physchem_start = i
        if aa != current_aa:
            if current_aa is not None:
                aa_slices[current_aa] = (aa_start, i - 1)
            current_aa = aa
            aa_start = i
 
    # Close final open slices
    n = len(_CODON_AA_GROUP)
    physchem_slices[current_physchem] = (physchem_start, n - 1)
    aa_slices[current_aa] = (aa_start, n - 1)
 
    return {"physchem": physchem_slices, "aa": aa_slices}
 
 
CODON_GROUP_SLICES = _codon_group_slices()
 
 
# Column (target AA) ordering — reuse existing ORDERED_AA.
# Build column-axis slices identical to the AA matrix's GROUP_SLICES_COL.
def _aa_group_slices_col() -> dict:
    """AA physchem group slices on the column (target) axis."""
    out = {}
    start = 0
    for g in GROUP_ORDER:
        aas = [aa for aa in ORDERED_AA if AA_GROUPS[aa] == g]
        if not aas:
            continue
        out[g] = (start, start + len(aas) - 1)
        start += len(aas)
    return out
 
 
AA_COL_SLICES = _aa_group_slices_col()
 
 
# ════════════════════════════════════════════════════════════════════════════
# Parse VEP Codons column
# ════════════════════════════════════════════════════════════════════════════
 
def _parse_ref_codon(codons_field: str | None) -> str | None:
    """
    VEP formats the Codons column as 'refCODON/altCODON' with uppercase bases
    marking the changed position (e.g. 'Gga/Aga' = GGA→AGA, change at pos 0).
    This returns the reference codon in uppercase.
    """
    if not isinstance(codons_field, str) or "/" not in codons_field:
        return None
    ref, _ = codons_field.split("/", 1)
    ref = ref.strip().upper()
    if len(ref) != 3 or any(c not in "ACGT" for c in ref):
        return None
    return ref
 
 
# ════════════════════════════════════════════════════════════════════════════
# Build codon × target-AA count matrix
# ════════════════════════════════════════════════════════════════════════════
 
def compute_codon_substitution_counts(
    df: pd.DataFrame,
    codons_col: str = "Codons",
    after_aa_col: str = "after_aa",
    consequence_col: str = "Consequence",
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    Build a 61×20 substitution count matrix (ref_codon × target_aa) from
    missense variants.
    """
    sub = df[
        df[consequence_col].fillna("").str.contains("missense_variant") &
        df[codons_col].notna() &
        df[after_aa_col].notna() &
        (df[after_aa_col].str.len() == 1) &
        df[after_aa_col].isin(ORDERED_AA)
    ].copy()
 
    sub["ref_codon"] = sub[codons_col].apply(_parse_ref_codon)
    sub = sub[sub["ref_codon"].isin(ORDERED_CODONS)]
 
    counts = (
        sub.groupby(["ref_codon", after_aa_col])
           .size()
           .unstack(fill_value=0)
    )
    return counts.reindex(index=ORDERED_CODONS, columns=ORDERED_AA, fill_value=0)
 
 
def row_normalize_codon(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalize a codon substitution matrix. Zero-sum rows stay zero."""
    row_sums = counts.sum(axis=1)
    return counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
 
 
# ════════════════════════════════════════════════════════════════════════════
# Enrichment + Fisher + FDR
# ════════════════════════════════════════════════════════════════════════════
 
def compute_codon_enrichment(
    counts_pos: pd.DataFrame,
    counts_neg: pd.DataFrame,
    min_total: int = 5,
) -> dict:
    """
    [Dataset-agnostic]
    Per-cell Fisher's exact with BH-FDR across all tested cells.
    Masking: any zero in 2×2 → log2 OR masked; cells below min_total skipped.
    """
    counts_pos = counts_pos.reindex(
        index=ORDERED_CODONS, columns=ORDERED_AA, fill_value=0
    )
    counts_neg = counts_neg.reindex(
        index=ORDERED_CODONS, columns=ORDERED_AA, fill_value=0
    )
 
    freq_pos = row_normalize_codon(counts_pos)
    freq_neg = row_normalize_codon(counts_neg)
 
    log2_or = pd.DataFrame(np.nan, index=ORDERED_CODONS, columns=ORDERED_AA, dtype=float)
    pval = pd.DataFrame(np.nan, index=ORDERED_CODONS, columns=ORDERED_AA, dtype=float)
 
    tested_cells = []
    for codon in ORDERED_CODONS:
        src_aa = _CODON_TABLE[codon]
        row_total_pos = counts_pos.loc[codon].sum()
        row_total_neg = counts_neg.loc[codon].sum()
        for target_aa in ORDERED_AA:
            # Skip silent cells (target = encoded AA)
            if target_aa == src_aa:
                continue
            pos_c = int(counts_pos.loc[codon, target_aa])
            neg_c = int(counts_neg.loc[codon, target_aa])
            if pos_c + neg_c < min_total:
                continue
 
            pos_out = row_total_pos - pos_c
            neg_out = row_total_neg - neg_c
 
            if 0 in (pos_c, pos_out, neg_c, neg_out):
                _, p = fisher_exact([[pos_c, pos_out], [neg_c, neg_out]])
                pval.loc[codon, target_aa] = p
                tested_cells.append((codon, target_aa, p))
                continue
 
            odds, p = fisher_exact([[pos_c, pos_out], [neg_c, neg_out]])
            log2_or.loc[codon, target_aa] = np.log2(odds)
            pval.loc[codon, target_aa] = p
            tested_cells.append((codon, target_aa, p))
 
    fdr = pd.DataFrame(np.nan, index=ORDERED_CODONS, columns=ORDERED_AA, dtype=float)
    if tested_cells:
        raw_ps = [c[2] for c in tested_cells]
        _, corrected, _, _ = multipletests(raw_ps, method="fdr_bh")
        for (codon, target_aa, _p), p_fdr in zip(tested_cells, corrected):
            fdr.loc[codon, target_aa] = p_fdr
 
    return {
        "freq_pos": freq_pos,
        "freq_neg": freq_neg,
        "counts_pos": counts_pos,
        "counts_neg": counts_neg,
        "log2_or": log2_or,
        "pval": pval,
        "fdr": fdr,
        "n_tested": len(tested_cells),
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# Plotting helpers for hierarchical boxes
# ════════════════════════════════════════════════════════════════════════════
 
def _add_codon_hierarchical_boxes(ax, shape: tuple[int, int]) -> None:
    """
    Overlay two levels of boxes on the y-axis (codons) and one on x-axis (AAs).
    Inner codon→AA boxes (thin), outer physchem-group boxes (thick).
    Column boxes are AA physchem groups.
    """
    n_rows, n_cols = shape
 
    # ── Rows (codons): inner AA boxes (thin) ────────────────────────────
    for aa, (start, end) in CODON_GROUP_SLICES["aa"].items():
        rect = patches.Rectangle(
            (0, start), width=n_cols, height=(end - start + 1),
            fill=False, edgecolor="gray", linewidth=0.5,
        )
        ax.add_patch(rect)
 
    # ── Rows (codons): outer physchem group boxes (thick) ───────────────
    for group, (start, end) in CODON_GROUP_SLICES["physchem"].items():
        rect = patches.Rectangle(
            (0, start), width=n_cols, height=(end - start + 1),
            fill=False, edgecolor="black", linewidth=1.5,
        )
        ax.add_patch(rect)
 
    # ── Columns (target AAs): physchem group boxes ──────────────────────
    for group, (start, end) in AA_COL_SLICES.items():
        rect = patches.Rectangle(
            (start, 0), width=(end - start + 1), height=n_rows,
            fill=False, edgecolor="black", linewidth=1.2,
        )
        ax.add_patch(rect)
 
 
def _add_significance_stars_codon(
    ax, fdr_df: pd.DataFrame, row_order: list[str]
) -> None:
    """Overlay *, **, *** on cells passing FDR threshold."""
    for i, codon in enumerate(row_order):
        for j, aa_to in enumerate(fdr_df.columns):
            p = fdr_df.loc[codon, aa_to]
            if pd.isna(p):
                continue
            stars = significance_stars(p)
            if stars != "n.s.":
                ax.text(
                    j + 0.5, i + 0.5, stars,
                    color="black", ha="center", va="center",
                    fontsize=7, fontweight="bold",
                )
 
 
# ════════════════════════════════════════════════════════════════════════════
# Three-panel plot
# ════════════════════════════════════════════════════════════════════════════
 
def plot_codon_substitution_matrix(
    enrichment: dict,
    dataset: str = "gnomad",
    save: bool = True,
    vmax_freq: float | None = None,
    vmax_or: float | None = None,
    show_counts: bool = False,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Three-panel heatmap:
      1. Pos frequencies (codon × target AA, row-normalized)
      2. Neg frequencies
      3. log2(pos/neg) enrichment with FDR stars
    Hierarchical row boxes: codons grouped by AA, AAs grouped by physchem.
    """
    # Standard row ordering (top→bottom matches ORDERED_CODONS, so don't reverse)
    row_order = ORDERED_CODONS
    freq_pos = enrichment["freq_pos"].reindex(index=row_order, columns=ORDERED_AA)
    freq_neg = enrichment["freq_neg"].reindex(index=row_order, columns=ORDERED_AA)
    log2_or = enrichment["log2_or"].reindex(index=row_order, columns=ORDERED_AA)
    fdr = enrichment["fdr"].reindex(index=row_order, columns=ORDERED_AA)
    counts_pos = enrichment["counts_pos"].reindex(index=row_order, columns=ORDERED_AA)
    counts_neg = enrichment["counts_neg"].reindex(index=row_order, columns=ORDERED_AA)
 
    if vmax_freq is None:
        vmax_freq = max(freq_pos.max().max(), freq_neg.max().max())
    if vmax_or is None:
        finite = log2_or.values[np.isfinite(log2_or.values)]
        vmax_or = np.max(np.abs(finite)) if len(finite) > 0 else 1.0
        if vmax_or == 0:
            vmax_or = 1.0
 
    # Taller figure for 61 rows
    fig, axes = plt.subplots(1, 3, figsize=(20, 14))
 
    # ── Panel 1: pos ────────────────────────────────────────────────────
    sns.heatmap(
        freq_pos, ax=axes[0],
        cmap=_CMAP_DICT["pos"], vmin=0, vmax=vmax_freq,
        annot=counts_pos if show_counts else False,
        fmt="d" if show_counts else "",
        annot_kws={"size": 5},
        cbar_kws={"label": "Row-normalized frequency", "shrink": 0.4},
    )
    axes[0].set_title("Positive group")
    axes[0].set_xlabel("AA to")
    axes[0].set_ylabel("Reference codon")
    _add_codon_hierarchical_boxes(axes[0], freq_pos.shape)
 
    # ── Panel 2: neg ────────────────────────────────────────────────────
    sns.heatmap(
        freq_neg, ax=axes[1],
        cmap=_CMAP_DICT["neg"], vmin=0, vmax=vmax_freq,
        annot=counts_neg if show_counts else False,
        fmt="d" if show_counts else "",
        annot_kws={"size": 5},
        cbar_kws={"label": "Row-normalized frequency", "shrink": 0.4},
    )
    axes[1].set_title("Negative group")
    axes[1].set_xlabel("AA to")
    axes[1].set_ylabel("")
    _add_codon_hierarchical_boxes(axes[1], freq_neg.shape)
 
    # ── Panel 3: enrichment ─────────────────────────────────────────────
    cmap = _CMAP_DICT["enrichment"].copy()
    cmap.set_bad(color="#DDDDDD")
 
    sns.heatmap(
        log2_or, ax=axes[2],
        cmap=cmap, center=0, vmin=-vmax_or, vmax=vmax_or,
        cbar_kws={"label": "log₂(OR) pos vs neg", "shrink": 0.4},
    )
    axes[2].set_title("Enrichment (significance = FDR)")
    axes[2].set_xlabel("AA to")
    axes[2].set_ylabel("")
    _add_codon_hierarchical_boxes(axes[2], log2_or.shape)
    _add_significance_stars_codon(axes[2], fdr, row_order)
 
    title = f"Codon-level substitution matrix ({dataset})"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=13, y=1.0)
 
    plt.tight_layout()
    if save:
        save_figure(fig, "substitution_matrix_codon", dataset=dataset)
 
    # Printed summary
    n_sig = int((fdr < 0.05).sum().sum())
    print(f"\n── Codon-level substitution matrix ({dataset}) ──")
    print(f"  Cells tested: {enrichment['n_tested']}")
    print(f"  Cells significant at FDR < 0.05: {n_sig}")
    if n_sig > 0:
        rows = []
        for codon in ORDERED_CODONS:
            src_aa = _CODON_TABLE[codon]
            for target_aa in ORDERED_AA:
                if target_aa == src_aa:
                    continue
                p = enrichment["fdr"].loc[codon, target_aa]
                if pd.notna(p) and p < 0.05:
                    lor = enrichment["log2_or"].loc[codon, target_aa]
                    if pd.notna(lor):
                        rows.append({
                            "codon": codon, "src_aa": src_aa, "target_aa": target_aa,
                            "transition": f"{src_aa}({codon})→{target_aa}",
                            "log2_or": lor, "fdr": p,
                            "pos": int(enrichment["counts_pos"].loc[codon, target_aa]),
                            "neg": int(enrichment["counts_neg"].loc[codon, target_aa]),
                        })
        if rows:
            sig_df = pd.DataFrame(rows)
            sig_df = sig_df.reindex(
                sig_df["log2_or"].abs().sort_values(ascending=False).index
            )
            print("\n  Top significant codon transitions (by |log2 OR|):")
            print(sig_df.head(15).to_string(index=False))
 
    return fig
 
 
# ════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ════════════════════════════════════════════════════════════════════════════
 
def run_codon_substitution_analysis(
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
    End-to-end codon substitution matrix: build counts, enrichment, plot.
    """
    df_pos = df[df[group_col] == pos_label]
    df_neg = df[df[group_col] == neg_label]
 
    counts_pos = compute_codon_substitution_counts(df_pos)
    counts_neg = compute_codon_substitution_counts(df_neg)
 
    enrichment = compute_codon_enrichment(counts_pos, counts_neg, min_total=min_total)
    plot_codon_substitution_matrix(enrichment, dataset=dataset, save=save, **plot_kwargs)
 
    return enrichment
 
# ════════════════════════════════════════════════════════════════════════════
# Marginal substitution distribution analysis
# ════════════════════════════════════════════════════════════════════════════

from scipy.stats import chi2_contingency

# ════════════════════════════════════════════════════════════════════════════
# SNP-reachability dict — single nucleotide substitution constraints
# ════════════════════════════════════════════════════════════════════════════

_CODON_TABLE_SNP = {
    'TTT': 'F', 'TTC': 'F',
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    'TAT': 'Y', 'TAC': 'Y',
    'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

def _build_snp_reachability() -> dict[str, list[str]]:
    """
    For each amino acid, return the sorted list of amino acids reachable
    via exactly one nucleotide substitution (missense only, no stops).
    Uses ORDERED_AA ordering for the target lists.
    """
    src_codons: dict[str, list[str]] = {}
    for codon, aa in _CODON_TABLE_SNP.items():
        src_codons.setdefault(aa, []).append(codon)

    reachability: dict[str, list[str]] = {}
    for src_aa, codons in src_codons.items():
        reachable: set[str] = set()
        for codon in codons:
            for pos in range(3):
                for base in "ACGT":
                    if base == codon[pos]:
                        continue
                    mutant = codon[:pos] + base + codon[pos + 1:]
                    tgt = _CODON_TABLE_SNP.get(mutant)  # None = stop codon
                    if tgt is not None and tgt != src_aa:
                        reachable.add(tgt)
        # Return in ORDERED_AA order for consistent plotting
        reachability[src_aa] = [aa for aa in ORDERED_AA if aa in reachable]

    return reachability

# Built once at module load — import this wherever needed
SNP_REACHABLE: dict[str, list[str]] = _build_snp_reachability()


def compute_marginal_substitution_distributions(
    df: pd.DataFrame,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    before_col: str = "before_aa",
    after_col: str = "after_aa",
    consequence_col: str = "Consequence",
    min_total: int = 50,
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    For each source AA, compare the distribution of SNP-reachable target AAs
    between pos and neg groups via chi-squared + KL divergence.
    Only biochemically possible single-nucleotide substitutions are tested.
    FDR correction over 150 reachable pairs (not 380).
    """
    missense = df[
        df[consequence_col].fillna("").str.contains("missense_variant") &
        df[before_col].notna() & df[after_col].notna() &
        df[before_col].isin(ORDERED_AA) & df[after_col].isin(ORDERED_AA)
    ].copy()
    missense = missense[missense[before_col] != missense[after_col]]

    # Drop SNP-impossible rows — they shouldn't be in VEP missense output
    # but guard against annotation artifacts
    missense = missense[
        missense.apply(
            lambda r: r[after_col] in SNP_REACHABLE.get(r[before_col], []),
            axis=1,
        )
    ]

    pos_df = missense[missense[group_col] == pos_label]
    neg_df = missense[missense[group_col] == neg_label]

    records = []
    tested_rows = []

    for src in ORDERED_AA:
        tgts = SNP_REACHABLE[src]  # only reachable targets

        pos_sub = pos_df[pos_df[before_col] == src]
        neg_sub = neg_df[neg_df[before_col] == src]

        pos_counts = np.array(
            [(pos_sub[after_col] == t).sum() for t in tgts], dtype=float
        )
        neg_counts = np.array(
            [(neg_sub[after_col] == t).sum() for t in tgts], dtype=float
        )

        n_pos = pos_counts.sum()
        n_neg = neg_counts.sum()
        total = n_pos + n_neg

        rec = {
            "source_aa": src,
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "n_reachable_targets": len(tgts),
            "total": int(total),
            "reachable_targets": tgts,
            "counts_pos": dict(zip(tgts, pos_counts.astype(int))),
            "counts_neg": dict(zip(tgts, neg_counts.astype(int))),
            "tested": total >= min_total and n_pos >= 5 and n_neg >= 5,
            "chi2": np.nan, "p_value": np.nan, "fdr": np.nan,
            "kl_symmetric": np.nan,
            "freq_pos": {}, "freq_neg": {},
        }

        if rec["tested"]:
            eps = 0.5
            freq_pos_smooth = (pos_counts + eps) / (n_pos + eps * len(tgts))
            freq_neg_smooth = (neg_counts + eps) / (n_neg + eps * len(tgts))

            rec["freq_pos"] = dict(zip(tgts, (pos_counts / n_pos).round(4)))
            rec["freq_neg"] = dict(zip(tgts, (neg_counts / n_neg).round(4)))

            contingency = np.vstack([pos_counts, neg_counts])
            mask = contingency.sum(axis=0) > 0
            chi2_stat, p, _, _ = chi2_contingency(contingency[:, mask])
            rec["chi2"] = chi2_stat
            rec["p_value"] = p

            kl_pn = np.sum(freq_pos_smooth * np.log(freq_pos_smooth / freq_neg_smooth))
            kl_np = np.sum(freq_neg_smooth * np.log(freq_neg_smooth / freq_pos_smooth))
            rec["kl_symmetric"] = (kl_pn + kl_np) / 2

            tested_rows.append((src, chi2_stat, p))

        records.append(rec)

    result_df = pd.DataFrame(records).set_index("source_aa")
    if tested_rows:
        raw_ps = [r[2] for r in tested_rows]
        _, corrected, _, _ = multipletests(raw_ps, method="fdr_bh")
        for (src, _, _), fdr_val in zip(tested_rows, corrected):
            result_df.loc[src, "fdr"] = fdr_val

    return result_df


def plot_marginal_substitution_distributions(
    result_df: pd.DataFrame,
    dataset: str = "gnomad",
    save: bool = True,
    fdr_threshold: float = 0.05,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Grid of paired bar charts — one panel per tested source AA.
    X-axis shows only SNP-reachable target AAs (varies per panel).
    Significant panels (FDR < threshold) are highlighted in title.
    """
    pos_color = GROUP_COLORS.get("pos", "#4daf4a")
    neg_color = GROUP_COLORS.get("neg", "#e41a1c")

    group_palette = {
        "Pos": "#2166ac", "Neg": "#d73027", "Polar": "#74add1",
        "Aromatic": "#984ea3", "Hydrophobic": "#4dac26", "C/G/P": "#8c510a",
    }

    tested = result_df[result_df["tested"]]
    panel_order = [aa for aa in ORDERED_AA if aa in tested.index]
    n_panels = len(panel_order)
    if n_panels == 0:
        print("No source AAs passed the min_total filter.")
        return None

    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 3.4),
        sharey=False,
    )
    axes_flat = axes.flatten() if n_panels > 1 else [axes]

    for ax_idx, src in enumerate(panel_order):
        ax = axes_flat[ax_idx]
        row = tested.loc[src]
        tgts = row["reachable_targets"]  # only SNP-reachable, in ORDERED_AA order

        freq_pos = np.array([row["freq_pos"].get(t, 0.0) for t in tgts])
        freq_neg = np.array([row["freq_neg"].get(t, 0.0) for t in tgts])
        cnt_pos  = np.array([row["counts_pos"].get(t, 0)  for t in tgts])
        cnt_neg  = np.array([row["counts_neg"].get(t, 0)  for t in tgts])

        x = np.arange(len(tgts))
        width = 0.38
        bars_p = ax.bar(x - width/2, freq_pos, width,
                        color=pos_color, alpha=0.85, label="pos")
        bars_n = ax.bar(x + width/2, freq_neg, width,
                        color=neg_color, alpha=0.85, label="neg")

        ax.set_xticks(x)
        ax.set_xticklabels(tgts, fontsize=8)
        for tick, tgt in zip(ax.get_xticklabels(), tgts):
            tick.set_color(group_palette.get(AA_GROUPS.get(tgt, ""), "black"))

        # Group separators on x-axis
        prev_g = AA_GROUPS.get(tgts[0])
        for i, t in enumerate(tgts[1:], 1):
            g = AA_GROUPS.get(t)
            if g != prev_g:
                ax.axvline(i - 0.5, color="gray", lw=0.5,
                           linestyle="--", alpha=0.5)
                prev_g = g

        ax.set_ylabel("Frequency", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlim(-0.6, len(tgts) - 0.4)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

        fdr_val = row["fdr"]
        kl = row["kl_symmetric"]
        is_sig = pd.notna(fdr_val) and fdr_val < fdr_threshold
        stars = significance_stars(fdr_val) if pd.notna(fdr_val) else ""
        sig_str = f" {stars}" if stars != "n.s." else ""
        title_color = "darkred" if is_sig else "black"
        title_weight = "bold" if is_sig else "normal"
        n_tgts = row["n_reachable_targets"]
        ax.set_title(
            f"{src}  [{n_tgts} reachable]  "
            f"(n_pos={row['n_pos']:,}, n_neg={row['n_neg']:,})\n"
            f"FDR={fdr_val:.2e}{sig_str}   KL={kl:.3f}",
            fontsize=7.5, color=title_color, fontweight=title_weight,
        )

    for ax_idx in range(len(panel_order), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor=pos_color, alpha=0.85, label="pos"),
            Patch(facecolor=neg_color, alpha=0.85, label="neg"),
        ],
        loc="lower right", fontsize=10, framealpha=0.9,
    )

    title = f"Marginal substitution distributions — SNP-reachable pairs only ({dataset})"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        save_figure(fig, "marginal_substitution_distributions", dataset=dataset)

    return fig

def run_marginal_substitution_analysis(
    df: pd.DataFrame,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    before_col: str = "before_aa",
    after_col: str = "after_aa",
    consequence_col: str = "Consequence",
    min_total: int = 50,
    fdr_threshold: float = 0.05,
    dataset: str = "gnomad",
    save: bool = True,
    sig_only: bool = False,
    source_aas: list[str] | None = None,
    plot_kind: str = "distribution",  # 'distribution' | 'enrichment_heatmap' | 'enrichment_bars' | 'all'
) -> pd.DataFrame:
    """
    End-to-end marginal substitution analysis.

    plot_kind:
      'distribution'        – existing pos/neg side-by-side bars (default)
      'enrichment_heatmap'  – log2(pos/neg) heatmap over (source × target)
      'enrichment_bars'     – per-source-AA divergent bars
      'all'                 – everything
    """
    result_df = compute_marginal_substitution_distributions(
        df, group_col=group_col, pos_label=pos_label, neg_label=neg_label,
        before_col=before_col, after_col=after_col,
        consequence_col=consequence_col, min_total=min_total,
    )

    tested = result_df[result_df["tested"]].copy()
    sig = tested[tested["fdr"] < fdr_threshold].sort_values(
        "kl_symmetric", ascending=False
    )

    if source_aas is not None:
        aas_to_plot = [aa for aa in source_aas if aa in result_df.index]
    elif sig_only:
        aas_to_plot = sig.index.tolist()
    else:
        aas_to_plot = None

    # --- Distribution plots (existing behaviour) ---
    if plot_kind in ("distribution", "all"):
        if sig_only or source_aas is not None:
            plot_marginal_substitution_distributions_focused(
                result_df, source_aas=aas_to_plot,
                dataset=dataset, save=save, fdr_threshold=fdr_threshold,
            )
        else:
            plot_marginal_substitution_distributions(
                result_df, dataset=dataset, save=save,
                fdr_threshold=fdr_threshold,
            )

    # --- Enrichment plots (new) ---
    if plot_kind in ("enrichment_heatmap", "enrichment_bars", "all"):
        enrich_df = compute_substitution_enrichment(
            df, group_col=group_col, pos_label=pos_label, neg_label=neg_label,
            before_col=before_col, after_col=after_col,
            consequence_col=consequence_col,
            min_source_total=min_total,
        )

        if plot_kind in ("enrichment_heatmap", "all"):
            plot_substitution_enrichment_heatmap(
                enrich_df, dataset=dataset, fdr_threshold=fdr_threshold,
                save=save,
            )

        if plot_kind in ("enrichment_bars", "all"):
            bars_aas = aas_to_plot if aas_to_plot else sorted(enrich_df["source"].unique())
            plot_substitution_enrichment_bars(
                enrich_df, source_aas=bars_aas, dataset=dataset,
                fdr_threshold=fdr_threshold, save=save,
            )

    # ── Printed summary (unchanged) ──
    print(f"\n── Marginal substitution distributions ({dataset}) ──")
    print(f"  Source AAs tested (n ≥ {min_total}): {len(tested)}")
    print(f"  Significant at FDR < {fdr_threshold}: {len(sig)}")
    if len(sig) > 0:
        print("\n  Significant source AAs (ranked by KL divergence):")
        summary = sig[["n_pos", "n_neg", "chi2", "p_value", "fdr", "kl_symmetric"]].copy()
        summary.columns = ["n_pos", "n_neg", "chi2", "p_raw", "fdr", "KL_sym"]
        summary["chi2"] = summary["chi2"].round(2)
        summary["p_raw"] = summary["p_raw"].apply(lambda x: f"{x:.2e}")
        summary["fdr"] = summary["fdr"].apply(lambda x: f"{x:.2e}")
        summary["KL_sym"] = summary["KL_sym"].round(4)
        print(summary.to_string())

    not_tested = result_df[~result_df["tested"]]
    if len(not_tested) > 0:
        print(f"\n  Skipped (< {min_total} total observations):",
              ", ".join(not_tested.index.tolist()))

    return result_df

def plot_marginal_substitution_distributions_focused(
    result_df: pd.DataFrame,
    source_aas: list[str],
    dataset: str = "gnomad",
    save: bool = True,
    fdr_threshold: float = 0.05,
    title_suffix: str = "",
) -> plt.Figure | None:
    """
    Publication-ready marginal substitution plot for a selected subset of
    source AAs (typically significant ones). Single row of panels, styled
    to match physchem_analysis.py — no redundant gray panels, tighter layout.
    """
    if not source_aas:
        print("No source AAs to plot.")
        return None

    # Filter to AAs that actually have data
    available = [aa for aa in source_aas if aa in result_df.index
                 and result_df.loc[aa, "tested"]]
    if not available:
        print("None of the requested AAs passed the min_total filter.")
        return None

    n = len(available)
    fig, axes = plt.subplots(
        1, n,
        figsize=(n * 3.6, 3.8),
        sharey=False,
    )
    if n == 1:
        axes = [axes]

    pos_color = GROUP_COLORS["pos"]
    neg_color = GROUP_COLORS["neg"]
    group_palette = {
        "Pos": "#2166ac", "Neg": "#d73027", "Polar": "#74add1",
        "Aromatic": "#984ea3", "Hydrophobic": "#4dac26", "C/G/P": "#8c510a",
    }

    for ax, src in zip(axes, available):
        row = result_df.loc[src]
        tgts = row["reachable_targets"]

        freq_pos = np.array([row["freq_pos"].get(t, 0.0) for t in tgts])
        freq_neg = np.array([row["freq_neg"].get(t, 0.0) for t in tgts])

        x = np.arange(len(tgts))
        width = 0.38
        ax.bar(x - width/2, freq_pos, width, color=pos_color,
               alpha=0.85, label="pos")
        ax.bar(x + width/2, freq_neg, width, color=neg_color,
               alpha=0.85, label="neg")

        # X tick labels colored by physicochemical group
        ax.set_xticks(x)
        ax.set_xticklabels(tgts, fontsize=9, fontweight="bold")
        for tick, tgt in zip(ax.get_xticklabels(), tgts):
            tick.set_color(group_palette.get(AA_GROUPS.get(tgt, ""), "black"))

        # Group separators
        prev_g = AA_GROUPS.get(tgts[0])
        for i, t in enumerate(tgts[1:], 1):
            g = AA_GROUPS.get(t)
            if g != prev_g:
                ax.axvline(i - 0.5, color="gray", lw=0.5,
                           linestyle="--", alpha=0.5)
                prev_g = g

        ax.set_xlim(-0.6, len(tgts) - 0.4)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

        # Y label only on leftmost panel
        if ax is axes[0]:
            ax.set_ylabel("Substitution frequency", fontsize=9)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=8)

        # Stats in top-right corner of each panel
        fdr_val = row["fdr"]
        kl      = row["kl_symmetric"]
        stars   = significance_stars(fdr_val) if pd.notna(fdr_val) else ""
        is_sig  = pd.notna(fdr_val) and fdr_val < fdr_threshold
        stats_text = (
            f"χ² FDR={fdr_val:.2e}\n"
            f"{stars}\n"
        )
        ax.text(
            0.5, 1.02, stats_text,
            transform=ax.transAxes, fontsize=7,
            va="top", ha="center",
            color="black",
            bbox=dict(facecolor="white", alpha=0.85,
                      edgecolor="none", pad=2),
        )

        # Panel title: source AA + group + counts
        # grp = AA_GROUPS.get(src, "")
        ax.set_title(
            f"{src}\n"
            f"n_pos={row['n_pos']:,}  n_neg={row['n_neg']:,}",
            fontsize=9,
            fontweight="bold" if is_sig else "normal",
            color= "black",pad=15

        )

    # Shared legend on last panel
    from matplotlib.patches import Patch
    axes[-1].legend(
        handles=[
            Patch(facecolor=pos_color, alpha=0.85, label="pos"),
            Patch(facecolor=neg_color, alpha=0.85, label="neg"),
        ],
        loc="upper left", fontsize=8, framealpha=0.9,
    )

    title = f"Marginal substitution distributions ({dataset})"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()

    if save:
        suffix = "sig" if not title_suffix else title_suffix.replace(" ", "_")
        save_figure(fig, f"marginal_substitution_focused_{suffix}", dataset=dataset)

    return fig

# ════════════════════════════════════════════════════════════════════════════
# Grouped (physicochemical) substitution matrix
# ════════════════════════════════════════════════════════════════════════════

# Map each AA to its group index in GROUP_ORDER for fast lookup
_AA_TO_GROUP = {aa: grp for aa, grp in AA_GROUPS.items()}


def plot_grouped_substitution_matrix(
    result: dict,
    dataset: str = "gnomad",
    save: bool = True,
    vmax_or: float | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Two-panel figure:
      Left:  log2 OR heatmap (6×6) with FDR significance stars.
             Diagonal masked gray (same-group, not tested).
      Right: paired bar chart of row-normalized frequencies per source group,
             giving the absolute frequency context alongside the OR.
    """
    log2_or  = result["log2_or"]
    fdr      = result["fdr"]
    freq_pos = result["freq_pos"]
    freq_neg = result["freq_neg"]
    counts_pos = result["counts_pos"]
    counts_neg = result["counts_neg"]

    if vmax_or is None:
        finite = log2_or.values[np.isfinite(log2_or.values)]
        vmax_or = float(np.max(np.abs(finite))) if len(finite) else 1.0

    pos_color = GROUP_COLORS.get("pos", "#4daf4a")
    neg_color = GROUP_COLORS.get("neg", "#e41a1c")

    cmap = _make_diverging_cmap()
    cmap.set_bad(color="#DDDDDD")

    fig = plt.figure(figsize=(16, 6))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.35)
    ax_heat = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    # ── Panel 1: log2 OR heatmap ─────────────────────────────────────────
    # Mask diagonal
    plot_or = log2_or.copy().astype(float)
    for g in GROUP_ORDER[::-1]:
        plot_or.loc[g, g] = np.nan

    sns.heatmap(
        plot_or,
        ax=ax_heat,
        cmap=cmap, center=0,
        vmin=-vmax_or, vmax=vmax_or,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "log₂(OR)  pos vs neg", "shrink": 0.8},
        square=True,
    )
    ax_heat.set_title("Grouped substitution enrichment\n(SNP-reachable pairs only)")
    ax_heat.set_xlabel("AA group → (target)")
    ax_heat.set_ylabel("AA group from (source)")
    ax_heat.tick_params(axis="both", labelsize=9)

    # Significance stars
    for i, gf in enumerate(GROUP_ORDER[::-1]):
        for j, gt in enumerate(GROUP_ORDER):
            if gf == gt:
                continue
            p = fdr.loc[gf, gt]
            if pd.isna(p):
                continue
            stars = significance_stars(p)
            if stars != "n.s.":
                ax_heat.text(
                    j + 0.5, i + 0.5, stars,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="black",
                )

    # ── Panel 2: paired bar chart — row-normalized frequencies ───────────
    n_groups = len(GROUP_ORDER)
    x = np.arange(n_groups)          # target groups on x-axis
    n_src = n_groups                  # one cluster per source group
    cluster_width = 0.7
    bar_w = cluster_width / 2

    # Vertical offset between source-group clusters
    y_offsets = np.linspace(0, (n_src - 1) * 0.55, n_src)
    cmap_src = plt.cm.tab10(np.linspace(0, 0.6, n_src))

    for src_idx, gf in enumerate(GROUP_ORDER[::-1]):
        fp = np.array([freq_pos.loc[gf, gt] for gt in GROUP_ORDER])
        fn = np.array([freq_neg.loc[gf, gt] for gt in GROUP_ORDER])
        offset = src_idx * 0.55        # stack source groups vertically

        # Plot as grouped bars per source, slightly offset vertically
        # Actually use a cleaner approach: facet by source group with color
        pass  # see below — use a proper grouped approach

    # Better: one subplot-row per source group would be ideal but complex.
    # Instead: grouped bars where x=target group, hue=pos/neg,
    # with a separate line per source group shown via transparency gradient.
    # For 6 source groups × 6 targets × 2 bars = 72 bars — use small multiples.

    # Replace ax_bar with a 2×3 grid of mini bar charts
    ax_bar.set_visible(False)
    fig.set_size_inches(16, 10)

    # Re-create layout with proper mini-grid for bar panels
    gs2 = fig.add_gridspec(
        2, 3,
        left=0.42, right=0.98,
        top=0.92, bottom=0.08,
        hspace=0.55, wspace=0.35,
    )

    for src_idx, gf in enumerate(GROUP_ORDER[::-1]):
        row, col = divmod(src_idx, 3)
        ax = fig.add_subplot(gs2[row, col])

        tgt_groups = [gt for gt in GROUP_ORDER if gt != gf]
        x = np.arange(len(tgt_groups))
        fp = np.array([freq_pos.loc[gf, gt] for gt in tgt_groups])
        fn = np.array([freq_neg.loc[gf, gt] for gt in tgt_groups])
        cp = np.array([counts_pos.loc[gf, gt] for gt in tgt_groups])
        cn = np.array([counts_neg.loc[gf, gt] for gt in tgt_groups])

        width = 0.35
        ax.bar(x - width/2, fp, width, color=pos_color, alpha=0.85, label="pos")
        ax.bar(x + width/2, fn, width, color=neg_color, alpha=0.85, label="neg")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [gt[:3] for gt in tgt_groups],   # abbreviated group names
            fontsize=8, rotation=30, ha="right",
        )
        ax.set_ylabel("Row freq.", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

        # Stars per bar pair from FDR
        ymax = max(fp.max(), fn.max()) if (len(fp) and len(fn)) else 0.1
        for i, gt in enumerate(tgt_groups):
            p = fdr.loc[gf, gt]
            stars = significance_stars(p) if pd.notna(p) else ""
            if stars and stars != "n.s.":
                ax.text(i, ymax * 1.05, stars, ha="center",
                        fontsize=8, fontweight="bold", color="darkred")

        n_pos_total = int(counts_pos.loc[gf].sum())
        n_neg_total = int(counts_neg.loc[gf].sum())
        ax.set_title(
            f"from {gf}\n(n_pos={n_pos_total:,}, n_neg={n_neg_total:,})",
            fontsize=8,
        )

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor=pos_color, alpha=0.85, label="pos"),
            Patch(facecolor=neg_color, alpha=0.85, label="neg"),
        ],
        loc="upper right", bbox_to_anchor=(0.99, 0.99),
        fontsize=9, framealpha=0.9,
    )

    title = f"Grouped substitution matrix — physicochemical groups ({dataset})"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=13, x=0.5, y=0.99)

    if save:
        save_figure(fig, "substitution_matrix_grouped", dataset=dataset)

    # ── Printed summary ───────────────────────────────────────────────────
    n_sig = int((result["fdr"] < 0.05).sum().sum())
    print(f"\n── Grouped substitution matrix ({dataset}) ──")
    print(f"  Cells tested (off-diagonal): {result['n_tested']}")
    print(f"  Cells significant at FDR < 0.05: {n_sig}")
    if n_sig > 0:
        rows = []
        for gf in GROUP_ORDER[::-1]:
            for gt in GROUP_ORDER:
                if gf == gt: continue
                p = result["fdr"].loc[gf, gt]
                lor = result["log2_or"].loc[gf, gt]
                if pd.notna(p) and p < 0.05:
                    rows.append({
                        "from": gf, "to": gt,
                        "log2_or": round(lor, 3) if pd.notna(lor) else np.nan,
                        "fdr": p,
                        "counts_pos": int(counts_pos.loc[gf, gt]),
                        "counts_neg": int(counts_neg.loc[gf, gt]),
                    })
        sig_df = pd.DataFrame(rows).sort_values("log2_or", key=abs, ascending=False)
        print("\n  Significant group transitions:")
        print(sig_df.to_string(index=False))

    return fig


# def run_grouped_substitution_analysis(
#     df: pd.DataFrame,
#     group_col: str = "group",
#     pos_label: str = "pos",
#     neg_label: str = "neg",
#     before_col: str = "before_aa",
#     after_col: str = "after_aa",
#     consequence_col: str = "Consequence",
#     dataset: str = "gnomad",
#     save: bool = True,
#     **plot_kwargs,
# ) -> dict:
#     """[Dataset-agnostic] End-to-end grouped substitution analysis."""
#     result = compute_grouped_substitution_matrix(
#         df, group_col=group_col, pos_label=pos_label, neg_label=neg_label,
#         before_col=before_col, after_col=after_col,
#         consequence_col=consequence_col,
#     )
#     plot_grouped_substitution_matrix(result, dataset=dataset, save=save, **plot_kwargs)
#     return result





def compute_substitution_enrichment(
    df: pd.DataFrame,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    before_col: str = "before_aa",
    after_col: str = "after_aa",
    consequence_col: str = "Consequence",
    pseudocount: float = 0.5,
    min_source_total: int = 50,
    restrict_to_snp_reachable: bool = True,
) -> pd.DataFrame:
    """
    Compute per-(source, target) AA enrichment of pos vs neg substitutions.

    For each source AA, conditions on substitutions from that AA, then computes:
      - freq_pos, freq_neg : conditional P(target | source) in each group
      - log2_enrichment    : log2((freq_pos + eps) / (freq_neg + eps))
      - p_value            : Fisher exact on the 2x2 [(target, ~target) x (pos, neg)]
      - fdr                : BH-corrected p (global across all tested cells)

    Returns long-form DataFrame with one row per (source, target) cell tested.
    """
    # Filter to missense
    mis = df[df[consequence_col].str.contains("missense", case=False, na=False)].copy()
    mis = mis[mis[before_col].isin(ORDERED_AA) & mis[after_col].isin(ORDERED_AA)]

    rows = []
    for src in ORDERED_AA:
        sub_pos = mis[(mis[group_col] == pos_label) & (mis[before_col] == src)]
        sub_neg = mis[(mis[group_col] == neg_label) & (mis[before_col] == src)]
        n_pos_total = len(sub_pos)
        n_neg_total = len(sub_neg)

        if n_pos_total + n_neg_total < min_source_total:
            continue

        # Optionally restrict targets to SNP-reachable ones
        if restrict_to_snp_reachable:
            try:
                from src.analysis_visualization.substitution_matrix_analysis import SNP_REACHABLE
                targets = [t for t in ORDERED_AA if t in SNP_REACHABLE.get(src, ORDERED_AA) and t != src]
            except Exception:
                targets = [t for t in ORDERED_AA if t != src]
        else:
            targets = [t for t in ORDERED_AA if t != src]

        for tgt in targets:
            k_pos = (sub_pos[after_col] == tgt).sum()
            k_neg = (sub_neg[after_col] == tgt).sum()

            # Conditional frequencies with pseudocount smoothing
            f_pos = (k_pos + pseudocount) / (n_pos_total + pseudocount * len(targets))
            f_neg = (k_neg + pseudocount) / (n_neg_total + pseudocount * len(targets))
            log2e = np.log2(f_pos / f_neg)

            # Fisher exact on contingency table
            table = [[k_pos, n_pos_total - k_pos],
                     [k_neg, n_neg_total - k_neg]]
            try:
                _, p = fisher_exact(table, alternative="two-sided")
            except ValueError:
                p = 1.0

            rows.append({
                "source": src, "target": tgt,
                "k_pos": k_pos, "k_neg": k_neg,
                "n_pos_total": n_pos_total, "n_neg_total": n_neg_total,
                "freq_pos": f_pos, "freq_neg": f_neg,
                "log2_enrichment": log2e,
                "p_value": p,
            })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["fdr"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def plot_substitution_enrichment_heatmap(
    enrich_df: pd.DataFrame,
    dataset: str = "gnomad",
    fdr_threshold: float = 0.05,
    vmax: float | None = None,
    save: bool = False,
    figsize: tuple = (11, 9),
):
    """
    Heatmap of log2(pos/neg) enrichment per (source, target) AA cell.
    Cells significant at FDR are marked with a dot; cells with NaN (not tested) are grey.
    """
    matrix = enrich_df.pivot(index="source", columns="target",
                             values="log2_enrichment")
    matrix = matrix.reindex(index=ORDERED_AA, columns=ORDERED_AA)

    sig_matrix = enrich_df.pivot(index="source", columns="target", values="fdr")
    sig_matrix = sig_matrix.reindex(index=ORDERED_AA, columns=ORDERED_AA)

    # Symmetric colour range
    if vmax is None:
        vmax = np.nanpercentile(np.abs(matrix.values), 98)
        vmax = max(vmax, 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#dddddd")

    im = ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect="equal")

    # Significance markers
    for i, src in enumerate(ORDERED_AA):
        for j, tgt in enumerate(ORDERED_AA):
            fdr = sig_matrix.loc[src, tgt]
            if pd.notna(fdr) and fdr < fdr_threshold:
                ax.plot(j, i, marker="o", color="black",
                        markersize=3.5, markeredgewidth=0)

    ax.set_xticks(range(len(ORDERED_AA)))
    ax.set_yticks(range(len(ORDERED_AA)))
    ax.set_xticklabels(ORDERED_AA)
    ax.set_yticklabels(ORDERED_AA)
    ax.set_xlabel("Target amino acid")
    ax.set_ylabel("Source amino acid")
    ax.set_title(
        f"Substitution enrichment, pos vs neg ({dataset})\n"
        f"log$_2$(freq$_{{pos}}$ / freq$_{{neg}}$); dots: FDR < {fdr_threshold}",
        fontsize=11,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("log$_2$ enrichment (pos / neg)")
    # Anchor the green/red ends to your group colours via tick labels
    cbar.ax.text(1.5, vmax, "  pos↑", va="center", ha="left",
                 color=GROUP_COLORS.get("pos", "#4daf4a"), fontweight="bold", transform=cbar.ax.transData)
    cbar.ax.text(1.5, -vmax, "  neg↑", va="center", ha="left",
                 color=GROUP_COLORS.get("neg", "#e41a1c"), fontweight="bold", transform=cbar.ax.transData)

    plt.tight_layout()
    if save:
        path = f"figures/substitution_enrichment_heatmap_{dataset}.pdf"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.show()
    return fig


def plot_substitution_enrichment_bars(
    enrich_df: pd.DataFrame,
    source_aas: list[str] | None = None,
    dataset: str = "gnomad",
    fdr_threshold: float = 0.05,
    save: bool = False,
    ncols: int = 4,
):
    """
    Per-source-AA divergent bar chart: log2 enrichment per target AA.
    Bars coloured by direction (pos-enriched green, neg-enriched red);
    significant bars edged in black.
    """
    if source_aas is None:
        source_aas = sorted(enrich_df["source"].unique())

    n = len(source_aas)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows),
                             squeeze=False, sharex=False)

    # Shared y-limit so panels are comparable
    ymax = enrich_df["log2_enrichment"].abs().quantile(0.98)
    ymax = max(ymax, 0.5)

    for i, src in enumerate(source_aas):
        ax = axes[i // ncols, i % ncols]
        sub = enrich_df[enrich_df["source"] == src].copy()
        sub = sub.sort_values("log2_enrichment")

        colors = [GROUP_COLORS.get("pos", "#4daf4a") if v > 0 else GROUP_COLORS.get("neg", "#e41a1c") for v in sub["log2_enrichment"]]
        edges = ["black" if f < fdr_threshold else "none" for f in sub["fdr"]]
        lws = [1.2 if f < fdr_threshold else 0 for f in sub["fdr"]]

        ax.barh(sub["target"], sub["log2_enrichment"],
                color=colors, edgecolor=edges, linewidth=lws)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlim(-ymax, ymax)
        ax.set_title(f"{src} →", fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        if i % ncols == 0:
            ax.set_ylabel("target AA", fontsize=9)
        if i // ncols == nrows - 1:
            ax.set_xlabel("log$_2$(pos/neg)", fontsize=9)

    # Blank remaining axes
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(
        f"Per-source-AA substitution enrichment ({dataset}) — "
        f"edged bars: FDR < {fdr_threshold}",
        y=1.005, fontsize=11,
    )
    plt.tight_layout()
    if save:
        path = f"figures/substitution_enrichment_bars_{dataset}.pdf"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.show()
    return fig

def run_mutability_normalized_analysis(
    df: pd.DataFrame,
    region_by_id: dict,
    rates_path: str = "/mnt/d/phd/scripts/16_ev_signature_predictor/data/samocha_mutation_rates/fordist_1KG_mutation_rate_table.txt",
    min_total: int = 5,
    dataset: str = "gnomad",
    save_table: bool = True,
    save_table_path: str = None
) -> dict:
    """
    Observed vs MUTABILITY-EXPECTED substitution enrichment.
    Same machinery as run_composition_normalized_analysis, but the null is
    weighted by the Samocha trinucleotide rates (CpG-aware) instead of flat.
    """
    rates = load_mutation_rates(rates_path)
    result = compute_composition_normalized_enrichment(
        df, region_by_id, min_total=min_total, rates=rates,   # pass-through
    )
    table, fig = plot_composition_normalized_matrix(result, dataset=f"{dataset}_mutability", save_table=save_table, save_table_path=save_table_path)
    return result, table

# ════════════════════════════════════════════════════════════════════════════
# Mutation-rate loading (Samocha 2014 trinucleotide model)
# ════════════════════

def load_mutation_rates(path: str = "/mnt/d/phd/scripts/16_ev_signature_predictor/data/samocha_mutation_rates/fordist_1KG_mutation_rate_table.txt") -> dict:
    """
    Load the Samocha 2014 trinucleotide mutation rate table.
    Returns {(trinucleotide_context, alt_base): rate}.
    context is the reference trinucleotide centered on the mutated base;
    alt_base is the central base after substitution.
    Strand-symmetric + all 64 contexts present => coding-strand context is
    valid for both plus- and minus-strand genes with no correction.
    """
    mu = pd.read_csv(path, sep=r"\s+", header=None, names=["from_tri", "to_tri", "mu"])
    return {(r.from_tri, r.to_tri[1]): r.mu for r in mu.itertuples()}

# red (top-left) - white (diagonal) - green (bottom-right) diverging field
_RWG = mcolors.LinearSegmentedColormap.from_list(
    "RWG", [(0.84, 0.19, 0.15), (1, 1, 1), (0.30, 0.69, 0.29)])


def _count_to_size(counts, smin=25, smax=420):
    """Log-scaled observed count -> marker size, anchored to the 10/100/1000 legend ticks."""
    c = np.log10(np.clip(counts, 1, None))
    lo, hi = np.log10(10), np.log10(1000)
    frac = np.clip((c - lo) / (hi - lo), 0, 1)
    return smin + (smax - smin) * frac

def plot_obs_exp_scatter(
    result: dict,
    dataset: str = "gnomad",
    save: bool = True,
    min_total: int = 5,
    pseudocount: float = 0.5,
    n_label: int = 12,
    always_label_sources: list[str] | None = None,
    fdr_threshold: float = 0.05,
    show_significance: bool = False,
    lim: float | None = None,                 # None=auto; number=symmetric [-lim, lim]
    bg_gradient: bool = True,                 # red-white-green diagonal field
    bg_alpha: float = 0.22,
    bg_span: float | None = None,             # None=auto (95th pct of |x-y|); set e.g. 2.0 to fix
    highlight_sources: list[str] | None = None,   # e.g. ["R"] -> emphasise, fade the rest
    faint_alpha: float = 0.12,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Per-substitution obs/exp scatter (main figure).
      x = log2(obs/exp) POS, y = log2(obs/exp) NEG, one point per substitution.
    Off-diagonal = group-specific deviation; the red->white->green background
    encodes direction (red top-left = neg-enriched/pos-depleted, green bottom-
    right = pos-enriched/neg-depleted, white = no difference). Marker SIZE encodes
    observed count (log scale; legend ticks at 10/100/1000). Significance (optional
    ring) uses the conditional binomial obs/exp test computed inline.

    highlight_sources: if given (e.g. ["R"] or ["R","G"]), points from those source
    AAs are emphasised + labelled, all others faded (faint_alpha) and unlabelled.
    """
    obs_pos, obs_neg = result["obs_pos"], result["obs_neg"]
    exp_pos, exp_neg = result["exp_pos"], result["exp_neg"]
    always = set(always_label_sources or [])
    hl = set(highlight_sources or [])

    recs = []
    for af in ORDERED_AA:
        for at in ORDERED_AA:
            if af == at:
                continue
            # print()
            op, on = obs_pos.loc[af, at], obs_neg.loc[af, at]
            ep, en = exp_pos.loc[af, at], exp_neg.loc[af, at]
            if ep <= 0 or en <= 0 or (op + on) < min_total:
                continue
            x = np.log2((op + pseudocount) / ep)
            y = np.log2((on + pseudocount) / en)
            recs.append(dict(af=af, at=at, x=x, y=y, diff=x - y,
                             total=op + on, p=binom_oe_test(op, on, ep, en)))
    d = pd.DataFrame(recs)
    # print(d)
    if d.empty:
        print("no cells pass filter")
        return None
    d = d.reset_index(drop=True)
    ok = d["p"].notna()
    d["fdr"] = np.nan
    if ok.any():
        d.loc[ok, "fdr"] = multipletests(d.loc[ok, "p"], method="fdr_bh")[1]
    d["abs_diff"] = d["diff"].abs()

    if lim is None:
        lim = float(np.nanmax(np.abs(np.r_[d.x, d.y]))) * 1.15

    fig, ax = plt.subplots(figsize=(8.8, 8.2))

    # background diverging field: colour by (x - y) = signed distance from diagonal
    if bg_gradient:
        if bg_span is None:
            bg_span = float(np.nanpercentile(d["abs_diff"], 95)) or 1.0
        N = 400
        g = np.linspace(-lim, lim, N)
        GX, GY = np.meshgrid(g, g)
        field = np.clip((GX - GY) / (2 * bg_span), -1, 1)
        ax.imshow(field, extent=[-lim, lim, -lim, lim], origin="lower",
                  cmap=_RWG, vmin=-1, vmax=1, alpha=bg_alpha, aspect="equal", zorder=0)

    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#777777", lw=1.0, zorder=1)

    sizes = _count_to_size(d["total"].values)

    if hl:
        is_hl = d["af"].isin(hl).values
        ax.scatter(d.x[~is_hl], d.y[~is_hl], s=sizes[~is_hl], c="#888888",
                   alpha=faint_alpha, edgecolor="none", zorder=2)
        ax.scatter(d.x[is_hl], d.y[is_hl], s=sizes[is_hl], c="#1f1f1f",
                   alpha=0.9, edgecolor="white", linewidth=0.5, zorder=4)
        label_mask = is_hl
    else:
        ax.scatter(d.x, d.y, s=sizes, c="#333333", alpha=0.8,
                   edgecolor="white", linewidth=0.4, zorder=3)
        label_mask = np.ones(len(d), bool)

    if show_significance:
        sig = (d["fdr"] < fdr_threshold).values & label_mask
        ax.scatter(d.x[sig], d.y[sig], s=sizes[sig] + 55, facecolors="none",
                   edgecolor="yellow", linewidth=1.3, zorder=5)
    

    if hl:
        to_label = set(d.index[label_mask].tolist())
    else:
        to_label = set(d.sort_values("abs_diff", ascending=False).head(n_label).index.tolist())
        to_label |= set(d.index[d["af"].isin(always)].tolist())

    if show_significance:
        to_label |= set(d.index[sig].tolist())
    # for i in to_label:
    #     ax.annotate(f"{d.at[i,'af']}\u2192{d.at[i,'at']}", (d.at[i, "x"], d.at[i, "y"]),
    #                 textcoords="offset points", xytext=(5, 4), fontsize=11, zorder=6)
    for i in to_label:
        left = d.at[i, "x"] < d.at[i, "y"]          # above diagonal -> label on the left
        ax.annotate(f"{d.at[i,'af']}\u2192{d.at[i,'at']}", (d.at[i, "x"], d.at[i, "y"]),
                    textcoords="offset points",
                    xytext=(-5, 4) if left else (5, 4),
                    ha="right" if left else "left",
                    fontsize=11, zorder=6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("log\u2082(obs / exp)  —  positive set\n"
                  "\u2190 fewer than expected                more than expected \u2192",
                  fontsize=10.5, labelpad=8)
    ax.set_ylabel("log\u2082(obs / exp)  —  negative set\n"
                  "\u2190 fewer than expected                more than expected \u2192",
                  fontsize=10.5, labelpad=8)
    ax.text(lim * 0.96, -lim * 0.96, "pos-enriched / neg-depleted",
            ha="right", va="bottom", fontsize=9, color=(0.20, 0.50, 0.20))
    ax.text(-lim * 0.96, lim * 0.96, "neg-enriched / pos-depleted",
            ha="left", va="top", fontsize=9, color=(0.70, 0.15, 0.12))

    # size legend OUTSIDE the axes, fixed ticks
    for cval in (10, 100, 1000):
        ax.scatter([], [], s=_count_to_size(np.array([cval]))[0], c="#333333",
                   alpha=0.8, edgecolor="white", linewidth=0.4, label=f"{cval:,}")
    leg = ax.legend(title="observed count\n(pos+neg)", loc="center left",
                    bbox_to_anchor=(1.02, 0.5), fontsize=9, title_fontsize=9,
                    labelspacing=1.6, borderpad=1.0, handletextpad=1.2,
                    frameon=True, framealpha=0.9)
    ax.add_artist(leg)

    title = f"Per-substitution obs/exp: pos vs neg ({dataset})"
    if highlight_sources:
        title += f" — {','.join(highlight_sources)} highlighted"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save:
        save_figure(fig, "obs_exp_scatter", dataset=dataset)
    return d[d["af"].isin(hl)], fig



def compute_grouped_substitution_matrix(
    df: pd.DataFrame,
    region_by_id: dict,
    rates: dict,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    before_col: str = "before_aa",
    after_col: str = "after_aa",
    consequence_col: str = "Consequence",
    groups: dict | None = None,
    pseudocount: float = 0.5,
) -> dict:
    """
    Physicochemical-group substitution enrichment, obs/exp-corrected.

    Pools observed missense counts AND rate-weighted expected counts into
    group×group bins (SNP-reachable pairs only), then per off-diagonal cell:
      - log2_or : log2((obs_pos/exp_pos)/(obs_neg/exp_neg))  [ratio-of-ratios]
      - pval/fdr: conditional binomial obs/exp test (binom_oe_test) + BH-FDR

    `groups` defaults to AA_GROUPS; pass a different AA->group dict to test
    alternative groupings (column/row order still follows GROUP_ORDER, so a
    custom dict must use the same group labels as GROUP_ORDER to plot cleanly).
    """
    groups = groups or AA_GROUPS

    # ---------- observed pooled ----------
    missense = df[
        df[consequence_col].fillna("").str.contains("missense_variant") &
        df[before_col].notna() & df[after_col].notna() &
        df[before_col].isin(ORDERED_AA) & df[after_col].isin(ORDERED_AA)
    ].copy()
    missense = missense[missense.apply(
        lambda r: r[after_col] in SNP_REACHABLE.get(r[before_col], []), axis=1)]
    missense["grp_from"] = missense[before_col].map(groups)
    missense["grp_to"] = missense[after_col].map(groups)

    def _pool_obs(sub):
        c = pd.DataFrame(0, index=GROUP_ORDER[::-1], columns=GROUP_ORDER)
        for (gf, gt), g in sub.groupby(["grp_from", "grp_to"]):
            if gf in c.index and gt in c.columns:
                c.loc[gf, gt] = len(g)
        return c

    counts_pos = _pool_obs(missense[missense[group_col] == pos_label])
    counts_neg = _pool_obs(missense[missense[group_col] == neg_label])

    # ---------- expected pooled (rate-weighted, scaled to each group's obs total) ----------
    exp_pos20 = compute_expected_substitution_counts(
        region_by_id, pos_label, observed_total=int(counts_pos.values.sum()), rates=rates)
    exp_neg20 = compute_expected_substitution_counts(
        region_by_id, neg_label, observed_total=int(counts_neg.values.sum()), rates=rates)

    def _pool_exp(mat):
        c = pd.DataFrame(0.0, index=GROUP_ORDER[::-1], columns=GROUP_ORDER)
        for af in mat.index:
            ga = groups.get(af)
            if ga not in c.index:
                continue
            for at in mat.columns:
                if af == at:
                    continue
                gt = groups.get(at)
                if gt not in c.columns:
                    continue
                c.loc[ga, gt] += mat.loc[af, at]
        return c

    exp_pos = _pool_exp(exp_pos20)
    exp_neg = _pool_exp(exp_neg20)

    # ---------- row-normalized observed freqs (for the bar panel) ----------
    def _rn(c):
        rs = c.sum(axis=1).replace(0, np.nan)
        return c.div(rs, axis=0).fillna(0)
    freq_pos, freq_neg = _rn(counts_pos), _rn(counts_neg)

    # ---------- per-cell ratio-of-ratios + binomial obs/exp test ----------
    log2_or = pd.DataFrame(np.nan, index=GROUP_ORDER[::-1], columns=GROUP_ORDER)
    pval = pd.DataFrame(np.nan, index=GROUP_ORDER[::-1], columns=GROUP_ORDER)
    tested = []
    for gf in GROUP_ORDER[::-1]:
        for gt in GROUP_ORDER:
            if gf == gt:
                continue
            op, on = counts_pos.loc[gf, gt], counts_neg.loc[gf, gt]
            ep, en = exp_pos.loc[gf, gt], exp_neg.loc[gf, gt]
            if ep <= 0 or en <= 0 or (op + on) == 0:
                continue
            log2_or.loc[gf, gt] = np.log2(((op + pseudocount) / ep) /
                                          ((on + pseudocount) / en))
            p = binom_oe_test(op, on, ep, en)
            if pd.isna(p):
                continue
            pval.loc[gf, gt] = p
            tested.append((gf, gt, p))

    fdr = pd.DataFrame(np.nan, index=GROUP_ORDER[::-1], columns=GROUP_ORDER)
    if tested:
        corr = multipletests([t[2] for t in tested], method="fdr_bh")[1]
        for (gf, gt, _), pf in zip(tested, corr):
            fdr.loc[gf, gt] = pf

    return {
        "counts_pos": counts_pos, "counts_neg": counts_neg,
        "exp_pos": exp_pos, "exp_neg": exp_neg,
        "freq_pos": freq_pos, "freq_neg": freq_neg,
        "log2_or": log2_or,          # NOW ratio-of-ratios, not Fisher OR
        "pval": pval, "fdr": fdr,
        "n_tested": len(tested),
    }

def run_grouped_substitution_analysis(
    df: pd.DataFrame,
    region_by_id: dict,
    rates_path: str = "/mnt/d/phd/scripts/16_ev_signature_predictor/data/samocha_mutation_rates/fordist_1KG_mutation_rate_table.txt",
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    before_col: str = "before_aa",
    after_col: str = "after_aa",
    consequence_col: str = "Consequence",
    groups: dict | None = None,
    dataset: str = "gnomad",
    save: bool = True,
    **plot_kwargs,
) -> dict:
    """End-to-end grouped obs/exp substitution analysis (mutability-corrected)."""
    rates = load_mutation_rates(rates_path)
    result = compute_grouped_substitution_matrix(
        df, region_by_id, rates,
        group_col=group_col, pos_label=pos_label, neg_label=neg_label,
        before_col=before_col, after_col=after_col,
        consequence_col=consequence_col, groups=groups,
    )
    plot_grouped_substitution_matrix(result, dataset=dataset, save=save, **plot_kwargs)
    return result

# def plot_obs_exp_scatter_grouped(
#     result: dict,
#     groups: dict | None = None,
#     dataset: str = "gnomad",
#     save: bool = True,
#     min_total: int = 5,
#     pseudocount: float = 0.5,
#     fdr_threshold: float = 0.05,
#     show_significance: bool = True,
#     label_all: bool = True,
#     n_label: int = 12,
#     title_suffix: str = "",
# ):
#     """
#     Group-level obs/exp scatter.
#       x = log2(obs/exp) POS,  y = log2(obs/exp) NEG, one point per group-pair.
#     Pools a 20x20 result dict by `groups` (default AA_GROUPS; swap freely to
#     brainstorm groupings). Significance is the conditional binomial obs/exp test
#     computed inline + BH-FDR across group-pairs — does NOT use result['fdr'].
#     Returns (fig, dataframe_of_points).
#     """
#     groups = groups or AA_GROUPS

#     def _pool(mat):
#         glabels = sorted(set(groups.values()))
#         c = pd.DataFrame(0.0, index=glabels, columns=glabels)
#         for af in mat.index:
#             ga = groups.get(af)
#             if ga is None:
#                 continue
#             for at in mat.columns:
#                 if af == at:
#                     continue
#                 gt = groups.get(at)
#                 if gt is None:
#                     continue
#                 c.loc[ga, gt] += mat.loc[af, at]
#         return c

#     OP, ON = _pool(result["obs_pos"]), _pool(result["obs_neg"])
#     EP, EN = _pool(result["exp_pos"]), _pool(result["exp_neg"])

#     recs = []
#     for gf in OP.index:
#         for gt in OP.columns:
#             op, on = OP.loc[gf, gt], ON.loc[gf, gt]
#             ep, en = EP.loc[gf, gt], EN.loc[gf, gt]
#             if ep <= 0 or en <= 0 or (op + on) < min_total:
#                 continue
#             x = np.log2((op + pseudocount) / ep)
#             y = np.log2((on + pseudocount) / en)
#             recs.append(dict(gf=gf, gt=gt, x=x, y=y, diff=x - y,
#                              total=op + on, p=binom_oe_test(op, on, ep, en)))
#     d = pd.DataFrame(recs)
#     if d.empty:
#         print("no group-pairs pass filter")
#         return None, None
#     d = d.reset_index(drop=True)
#     ok = d["p"].notna()
#     d["fdr"] = np.nan
#     if ok.any():
#         d.loc[ok, "fdr"] = multipletests(d.loc[ok, "p"], method="fdr_bh")[1]
#     d["abs_diff"] = d["diff"].abs()

#     lim = float(np.nanmax(np.abs(np.r_[d.x, d.y]))) * 1.20
#     sizes = 40 + 260 * (d["total"] / d["total"].max())

#     fig, ax = plt.subplots(figsize=(8.6, 8.2))
#     ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
#     ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)
#     ax.plot([-lim, lim], [-lim, lim], ls="--", color="#888888", lw=1.0, zorder=1)
#     ax.fill_between([-lim, lim], [-lim - 0.4, lim - 0.4], [-lim + 0.4, lim + 0.4],
#                     color="#999999", alpha=0.08, zorder=0)
#     colours = [GROUP_COLORS["pos"] if v > 0 else GROUP_COLORS["neg"] for v in d["diff"]]

#     if show_significance:
#         sig = d["fdr"] < fdr_threshold
#         ax.scatter(d.x[~sig], d.y[~sig], s=sizes[~sig],
#                    c=[c for c, s in zip(colours, sig) if not s],
#                    alpha=0.55, edgecolor="none", zorder=3)
#         ax.scatter(d.x[sig], d.y[sig], s=sizes[sig],
#                    c=[c for c, s in zip(colours, sig) if s],
#                    alpha=0.9, edgecolor="black", linewidth=1.5, zorder=4)
#     else:
#         ax.scatter(d.x, d.y, s=sizes, c=colours, alpha=0.6, edgecolor="none", zorder=3)

#     lab_idx = (set(range(len(d))) if label_all
#                else set(d.sort_values("abs_diff", ascending=False).head(n_label).index))
#     for i in lab_idx:
#         sig_i = (d.at[i, "fdr"] < fdr_threshold) if pd.notna(d.at[i, "fdr"]) else False
#         ax.annotate(f"{d.at[i,'gf']}\u2192{d.at[i,'gt']}", (d.at[i, "x"], d.at[i, "y"]),
#                     textcoords="offset points", xytext=(5, 4), fontsize=8,
#                     fontweight="bold" if (show_significance and sig_i) else "normal", zorder=5)

#     ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
#     ax.set_xlabel("log\u2082(obs / exp)  —  POS group", fontsize=11)
#     ax.set_ylabel("log\u2082(obs / exp)  —  NEG group", fontsize=11)
#     ax.text(lim * 0.97, -lim * 0.97, "pos-enriched\nneg-depleted", ha="right", va="bottom",
#             fontsize=8, color=GROUP_COLORS["pos"], alpha=0.8)
#     ax.text(-lim * 0.97, lim * 0.97, "neg-enriched\npos-depleted", ha="left", va="top",
#             fontsize=8, color=GROUP_COLORS["neg"], alpha=0.8)

#     qs = np.unique(np.quantile(d["total"], [0.1, 0.5, 0.95]).round().astype(int))
#     handles = [plt.scatter([], [], s=40 + 260 * (q / d["total"].max()),
#                            c="#777777", alpha=0.6, edgecolor="none", label=f"{int(q)}")
#                for q in qs]
#     leg = ax.legend(handles=handles, title="obs count (pos+neg)", loc="lower left",
#                     fontsize=8, title_fontsize=8, labelspacing=1.2,
#                     borderpad=0.8, framealpha=0.9)
#     ax.add_artist(leg)

#     title = f"Group-level obs/exp: pos vs neg ({dataset})"
#     if title_suffix:
#         title += f" — {title_suffix}"
#     sub = "off-diagonal = group-specific; size ~ count"
#     if show_significance:
#         sub += f"; ring = obs/exp binom FDR<{fdr_threshold:g}"
#     ax.set_title(title + "\n" + sub, fontsize=10)
#     plt.tight_layout()
#     if save:
#         save_figure(fig, "obs_exp_scatter_grouped", dataset=dataset)
#     return fig, d


def is_cpg_transition(context: str | None, alt: str) -> bool:
    """
    True if a single-nt change is a CpG transition (the hypermutable class):
    C>T where the C is immediately 5' of a G, or G>A where the G is immediately
    3' of a C. `context` is the reference trinucleotide (5'flank, ref, 3'flank);
    `alt` is the new central base. CpG is strand-symmetric, so this gives the
    same answer on coding- or genomic-strand context.
    """
    if context is None or len(context) != 3:
        return False
    ref = context[1]
    if ref == "C" and context[2] == "G" and alt == "T":
        return True
    if ref == "G" and context[0] == "C" and alt == "A":
        return True
    return False

def annotate_cpg_status(df: pd.DataFrame, fasta_path: str,
                        chrom_col="CHROM", pos_col="POS",
                        ref_col="REF", alt_col="ALT") -> pd.DataFrame:
    """
    Add a boolean `cpg` column flagging CpG-transition variants, using genomic
    (plus-strand) context from a reference FASTA. POS is 1-based; pysam.fetch is
    0-based half-open, so the trinucleotide centered on POS is fetch(POS-2, POS+1).
    Strand-safe: gnomAD REF/ALT/POS are genomic plus-strand, and CpG is
    strand-symmetric, so this matches the coding-strand classification used for
    the expected side.
    """
    import pysam
    fa = pysam.FastaFile(fasta_path)
    out = df.copy()

    def _cpg(r):
        ref, alt = r[ref_col], r[alt_col]
        if len(str(ref)) != 1 or len(str(alt)) != 1:   # SNVs only
            return False
        try:
            tri = fa.fetch(str(r[chrom_col]), int(r[pos_col]) - 2,
                           int(r[pos_col]) + 1).upper()
        except (KeyError, ValueError):
            return False
        if len(tri) != 3:
            return False
        # tri = [POS-1, POS, POS+1]; tri[1] should equal REF
        return is_cpg_transition(tri, alt)

    out["cpg"] = out.apply(_cpg, axis=1)
    fa.close()
    return out

def _size_from_count(counts, ticks):
    """Log-scaled count -> marker size, anchored to the legend ticks."""
    lo, hi = np.log10(min(ticks)), np.log10(max(ticks))
    c = np.log10(np.clip(counts, 1, None))
    frac = np.clip((c - lo) / (hi - lo), 0, 1)
    return 25 + (420 - 25) * frac


def plot_obs_exp_scatter_grouped(
    result: dict,
    groups: dict | None = None,
    dataset: str = "gnomad",
    save: bool = True,
    min_total: int = 5,
    pseudocount: float = 0.5,
    fdr_threshold: float = 0.05,
    show_significance: bool = False,
    lim: float | None = None,
    bg_gradient: bool = True,
    bg_alpha: float = 0.22,
    bg_span: float | None = None,
    highlight_sources: list[str] | None = None,   # source GROUP labels, e.g. ["Pos"], ["C/G/P"]
    faint_alpha: float = 0.12,
    size_ticks: tuple = (100, 1000, 10000),
    label_all: bool = True,
    n_label: int | None = None,
    title_suffix: str = "",
):
    """
    Group-level obs/exp scatter, styled to match plot_obs_exp_scatter.
      x = log2(obs/exp) POS, y = log2(obs/exp) NEG, one point per group-pair.
    Pools a 20x20 result dict by `groups` (default AA_GROUPS; swap to brainstorm).
    Red->white->green background encodes direction; marker size encodes observed
    count; significance ring uses the conditional binomial obs/exp test inline.
    `highlight_sources` here = source GROUP labels (emphasise those, fade the rest).
    Returns (fig, dataframe_of_points).
    """
    groups = groups or AA_GROUPS
    hl = set(highlight_sources or [])

    def _pool(mat):
        gl = sorted(set(groups.values()))
        c = pd.DataFrame(0.0, index=gl, columns=gl)
        for af in mat.index:
            ga = groups.get(af)
            if ga is None:
                continue
            for at in mat.columns:
                if af == at:
                    continue
                gt = groups.get(at)
                if gt is None:
                    continue
                c.loc[ga, gt] += mat.loc[af, at]
        return c

    OP, ON = _pool(result["obs_pos"]), _pool(result["obs_neg"])
    EP, EN = _pool(result["exp_pos"]), _pool(result["exp_neg"])

    recs = []
    for gf in OP.index:
        for gt in OP.columns:
            op, on = OP.loc[gf, gt], ON.loc[gf, gt]
            ep, en = EP.loc[gf, gt], EN.loc[gf, gt]
            if ep <= 0 or en <= 0 or (op + on) < min_total:
                continue
            x = np.log2((op + pseudocount) / ep)
            y = np.log2((on + pseudocount) / en)
            recs.append(dict(gf=gf, gt=gt, x=x, y=y, diff=x - y,
                             total=op + on, p=binom_oe_test(op, on, ep, en)))
    d = pd.DataFrame(recs)
    if d.empty:
        print("no group-pairs pass filter")
        return None, None
    d = d.reset_index(drop=True)
    ok = d["p"].notna()
    d["fdr"] = np.nan
    if ok.any():
        d.loc[ok, "fdr"] = multipletests(d.loc[ok, "p"], method="fdr_bh")[1]
    d["abs_diff"] = d["diff"].abs()

    if lim is None:
        lim = float(np.nanmax(np.abs(np.r_[d.x, d.y]))) * 1.20

    fig, ax = plt.subplots(figsize=(8.8, 8.2))

    if bg_gradient:
        if bg_span is None:
            bg_span = float(np.nanpercentile(d["abs_diff"], 95)) or 1.0
        g = np.linspace(-lim, lim, 400)
        GX, GY = np.meshgrid(g, g)
        field = np.clip((GX - GY) / (2 * bg_span), -1, 1)
        ax.imshow(field, extent=[-lim, lim, -lim, lim], origin="lower",
                  cmap=_RWG, vmin=-1, vmax=1, alpha=bg_alpha, aspect="equal", zorder=0)

    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#777777", lw=1.0, zorder=1)

    sizes = _size_from_count(d["total"].values, size_ticks)

    if hl:
        is_hl = d["gf"].isin(hl).values
        ax.scatter(d.x[~is_hl], d.y[~is_hl], s=sizes[~is_hl], c="#888888",
                   alpha=faint_alpha, edgecolor="none", zorder=2)
        ax.scatter(d.x[is_hl], d.y[is_hl], s=sizes[is_hl], c="#1f1f1f",
                   alpha=0.9, edgecolor="white", linewidth=0.5, zorder=4)
        label_mask = is_hl
    else:
        ax.scatter(d.x, d.y, s=sizes, c="#333333", alpha=0.8,
                   edgecolor="white", linewidth=0.4, zorder=3)
        label_mask = np.ones(len(d), bool)

    if show_significance:
        sig = (d["fdr"] < fdr_threshold).values & label_mask
        ax.scatter(d.x[sig], d.y[sig], s=sizes[sig] + 55, facecolors="none",
                   edgecolor="yellow", linewidth=1.3, zorder=5)

    if hl:
        to_label = set(d.index[label_mask].tolist())
    elif label_all:
        to_label = set(range(len(d)))
    elif n_label is not None:
        to_label = set(d.sort_values("abs_diff", ascending=False).head(n_label).index)
    else:
        to_label = set(d.sort_values("abs_diff", ascending=False).head(12).index)
    print(to_label)
    print(n_label)
    if show_significance:
        to_label |= set(d.index[sig].tolist())
    print(to_label)
    for i in to_label:
        left = d.at[i, "x"] < d.at[i, "y"]          # above diagonal -> label on the left
        ax.annotate(f"{d.at[i,'gf']}\u2192{d.at[i,'gt']}", (d.at[i, "x"], d.at[i, "y"]),
                    textcoords="offset points",
                    xytext=(-5, 4) if left else (5, 4),
                    ha="right" if left else "left",
                    fontsize=11, zorder=6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("log\u2082(obs / exp)  —  positive set\n"
                  "\u2190 fewer than expected                more than expected \u2192",
                  fontsize=10.5, labelpad=8)
    ax.set_ylabel("log\u2082(obs / exp)  —  negative set\n"
                  "\u2190 fewer than expected                more than expected \u2192",
                  fontsize=10.5, labelpad=8)
    ax.text(lim * 0.96, -lim * 0.96, "pos-enriched / neg-depleted",
            ha="right", va="bottom", fontsize=8, color=(0.20, 0.50, 0.20))
    ax.text(-lim * 0.96, lim * 0.96, "neg-enriched / pos-depleted",
            ha="left", va="top", fontsize=8, color=(0.70, 0.15, 0.12))

    for cval in size_ticks:
        ax.scatter([], [], s=_size_from_count(np.array([cval]), size_ticks)[0],
                   c="#333333", alpha=0.8, edgecolor="white", linewidth=0.4, label=f"{cval:,}")
    leg = ax.legend(title="observed count\n(pos+neg)", loc="center left",
                    bbox_to_anchor=(1.02, 0.5), fontsize=9, title_fontsize=9,
                    labelspacing=1.6, borderpad=1.0, handletextpad=1.2,
                    frameon=True, framealpha=0.9)
    ax.add_artist(leg)

    title = f"Group-level obs/exp: pos vs neg ({dataset})"
    if highlight_sources:
        title += f" — {','.join(highlight_sources)} highlighted"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save:
        save_figure(fig, "obs_exp_scatter_grouped", dataset=dataset)
    return d, fig


# """
# Substitution-spectrum scoring: group-specific, selection-corrected log2-odds
# substitution matrix, built from the substitution-comparison dataframe.
 
# Importable module. No CLI / no __main__ block.
 
# Score per substitution a->b:
#     score = log2((obs_pos + alpha)/exp_pos) - log2((obs_neg + alpha)/exp_neg)
# Centered at 0: >0 enriched in positive (functional) group, <0 in negative.
# """
 
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm
 
 
def build_score_table(df, alpha=0.5, min_count=0, key_sep="->"):
    """
    Build the per-substitution score table.
 
    df must contain: 'from','to','obs_pos','obs_neg','exp_pos','exp_neg'.
    alpha     : pseudocount on observed counts (shrinks low-count scores -> 0).
    min_count : drop substitutions with total obs count below this (0 = keep all).
    key_sep   : separator for the lookup key, e.g. 'F->S'.
 
    Returns (table, lookup):
      table  : DataFrame indexed by key, cols [from,to,obs_pos,obs_neg,exp_pos,
               exp_neg,total_count,score_raw,score]
      lookup : dict {key -> score}  for Step 2
    """
    required = {"from", "to", "obs_pos", "obs_neg", "exp_pos", "exp_neg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataframe is missing required columns: {sorted(missing)}")
 
    t = df.copy()
    eps = 1e-9
    exp_pos = t["exp_pos"].clip(lower=eps)
    exp_neg = t["exp_neg"].clip(lower=eps)
 
    ratio_pos = (t["obs_pos"] + alpha) / exp_pos
    ratio_neg = (t["obs_neg"] + alpha) / exp_neg
    t["score"] = np.log2(ratio_pos) - np.log2(ratio_neg)
 
    ratio_pos_raw = t["obs_pos"].clip(lower=eps) / exp_pos
    ratio_neg_raw = t["obs_neg"].clip(lower=eps) / exp_neg
    t["score_raw"] = np.log2(ratio_pos_raw) - np.log2(ratio_neg_raw)
 
    t["total_count"] = t["obs_pos"] + t["obs_neg"]
    t["key"] = t["from"].astype(str) + key_sep + t["to"].astype(str)
 
    if min_count > 0:
        t = t[t["total_count"] >= min_count].copy()
 
    cols = ["key", "from", "to", "obs_pos", "obs_neg", "exp_pos", "exp_neg",
            "total_count", "score_raw", "score"]
    table = t[cols].set_index("key")
    lookup = table["score"].to_dict()
    return table, lookup
 
 
def plot_score_heatmap(table, ax=None, aa_order=None, cmap="RdBu_r",
                       annotate=False, title="Substitution score matrix"):
    """
    Quick diverging heatmap of the score matrix (source AA = rows, dest AA = cols).
    Empty (unobserved) substitution cells are left blank.
 
    table   : output of build_score_table (must have 'from','to','score').
    aa_order: optional explicit ordering of amino acids on both axes.
    annotate: write the score value in each cell.
    Returns the matplotlib Axes.
    """
    mat = table.pivot_table(index="from", columns="to", values="score", aggfunc="mean")
 
    if aa_order is not None:
        rows = [a for a in aa_order if a in mat.index]
        cols = [a for a in aa_order if a in mat.columns]
        mat = mat.reindex(index=rows, columns=cols)
 
    vmax = np.nanmax(np.abs(mat.values))
    vmax = 1.0 if not np.isfinite(vmax) or vmax == 0 else vmax
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.5*mat.shape[1]+2),
                                        max(3, 0.5*mat.shape[0]+1)))
    im = ax.imshow(mat.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    ax.set_xlabel("to (destination AA)")
    ax.set_ylabel("from (source AA)")
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("score  (<0 neg-enriched,  >0 pos-enriched)")
 
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=7, color="black")
    return ax
 
