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
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    Aggregate all possible single-nucleotide missense substitutions across
    regions in the given group. Scale the resulting matrix to match the
    observed total missense count.
 
    Returns a 20×20 DataFrame of expected counts per substitution (AA_from × AA_to).
    """
    matrix = pd.DataFrame(
        0.0, index=ORDERED_AA, columns=ORDERED_AA,
    )
    total_possible = 0
 
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
            if row.aa_from in ORDERED_AA and row.aa_to in ORDERED_AA:
                matrix.loc[row.aa_from, row.aa_to] += 1
                total_possible += 1
 
    # Scale to match observed total
    if total_possible == 0:
        return matrix
    scale = observed_total / total_possible
    return matrix * scale
 
 
# ════════════════════════════════════════════════════════════════════════════
# Build observed-vs-expected enrichment ratios per group
# ════════════════════════════════════════════════════════════════════════════
 
def compute_composition_normalized_enrichment(
    df: pd.DataFrame,
    region_by_id: dict,
    min_total: int = 5,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
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
    df_pos = df[df[group_col] == pos_label]
    df_neg = df[df[group_col] == neg_label]
 
    obs_pos = compute_substitution_counts(df_pos)
    obs_neg = compute_substitution_counts(df_neg)
 
    exp_pos = compute_expected_substitution_counts(
        region_by_id, pos_label, observed_total=int(obs_pos.values.sum())
    )
    exp_neg = compute_expected_substitution_counts(
        region_by_id, neg_label, observed_total=int(obs_neg.values.sum())
    )
 
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
 
            # Fisher p-value from raw observed counts (same as before)
            pos_total_row = int(obs_pos.loc[aa_from].sum())
            neg_total_row = int(obs_neg.loc[aa_from].sum())
            pos_out = pos_total_row - pos_c
            neg_out = neg_total_row - neg_c
            _, p = fisher_exact([[pos_c, pos_out], [neg_c, neg_out]])
 
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

    # ── Printed summary ────────────────────────────────────────────────────
    n_sig = int((result["fdr"] < 0.05).sum().sum())
    print(f"\n── Composition-normalized substitution matrix ({dataset}) ──")
    print(f"  Cells tested: {result['n_tested']}")
    print(f"  Cells significant at FDR < 0.05: {n_sig}")

    if n_sig > 0:
        sig_cells = []
        for aa_from in ORDERED_AA:
            for aa_to in ORDERED_AA:
                if aa_from == aa_to:
                    continue
                p = result["fdr"].loc[aa_from, aa_to]
                if pd.notna(p) and p < 0.05:
                    sig_cells.append({
                        "from": aa_from, "to": aa_to,
                        "log2_ratio_diff": result["log2_ratio_diff"].loc[aa_from, aa_to],
                        "fdr": p,
                        "obs_pos": int(result["obs_pos"].loc[aa_from, aa_to]),
                        "obs_neg": int(result["obs_neg"].loc[aa_from, aa_to]),
                        "exp_pos": float(result["exp_pos"].loc[aa_from, aa_to]),
                        "exp_neg": float(result["exp_neg"].loc[aa_from, aa_to]),
                        "ratio_pos": float(result["ratio_pos"].loc[aa_from, aa_to]),
                        "ratio_neg": float(result["ratio_neg"].loc[aa_from, aa_to]),
                    })
        sig_df = pd.DataFrame(sig_cells)
        sig_df = sig_df.reindex(
            sig_df["log2_ratio_diff"].abs().sort_values(ascending=False).index
        )
        print("\n  Top composition-controlled group differences:")
        print(sig_df.head(15).to_string(index=False))

    return fig
 
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
    plot_composition_normalized_matrix(result, dataset=dataset, save=save)
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
 
