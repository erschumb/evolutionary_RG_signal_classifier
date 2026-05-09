"""
Codon usage analysis.

Save as: src/analysis_visualization/codon_usage.py

Compares codon usage per amino acid between pos/neg region groups against
the human proteome reference (Kazusa database, homo sapiens 9606).

Helps diagnose whether substitution patterns reflect selection vs codon
composition differences between groups. Also controls for the CpG
hypermutability of codons like CGA, CGG, GCG, etc.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from src.analysis_visualization.plot_config import (
    GROUP_COLORS, save_figure, significance_stars,
)


# ════════════════════════════════════════════════════════════════════════════
# Human codon usage reference (Kazusa 2007, homo sapiens 9606)
# Frequencies in per-thousand codons.
# ════════════════════════════════════════════════════════════════════════════

HUMAN_CODON_USAGE = {
    # Phe
    "TTT": 17.6, "TTC": 20.3,
    # Leu
    "TTA":  7.7, "TTG": 12.9, "CTT": 13.2, "CTC": 19.6, "CTA":  7.2, "CTG": 39.6,
    # Ser
    "TCT": 15.2, "TCC": 17.7, "TCA": 12.2, "TCG":  4.4, "AGT": 12.1, "AGC": 19.5,
    # Tyr
    "TAT": 12.2, "TAC": 15.3,
    # stop
    "TAA":  1.0, "TAG":  0.8, "TGA":  1.6,
    # Cys
    "TGT": 10.6, "TGC": 12.6,
    # Trp
    "TGG": 13.2,
    # Pro
    "CCT": 17.5, "CCC": 19.8, "CCA": 16.9, "CCG":  6.9,
    # His
    "CAT": 10.9, "CAC": 15.1,
    # Gln
    "CAA": 12.3, "CAG": 34.2,
    # Arg
    "CGT":  4.5, "CGC": 10.4, "CGA":  6.2, "CGG": 11.4, "AGA": 12.2, "AGG": 12.0,
    # Ile
    "ATT": 16.0, "ATC": 20.8, "ATA":  7.5,
    # Met
    "ATG": 22.0,
    # Thr
    "ACT": 13.1, "ACC": 18.9, "ACA": 15.1, "ACG":  6.1,
    # Asn
    "AAT": 17.0, "AAC": 19.1,
    # Lys
    "AAA": 24.4, "AAG": 31.9,
    # Val
    "GTT": 11.0, "GTC": 14.5, "GTA":  7.1, "GTG": 28.1,
    # Ala
    "GCT": 18.4, "GCC": 27.7, "GCA": 15.8, "GCG":  7.4,
    # Asp
    "GAT": 21.8, "GAC": 25.1,
    # Glu
    "GAA": 29.0, "GAG": 39.6,
    # Gly
    "GGT": 10.8, "GGC": 22.2, "GGA": 16.5, "GGG": 16.5,
}


# Standard genetic code: codon -> amino acid
_CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F',
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    'TAT': 'Y', 'TAC': 'Y',
    'TAA': '*', 'TAG': '*', 'TGA': '*',
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


# AAs with multiple codons worth testing (skip M and W — single codon each)
MULTI_CODON_AA = sorted({aa for aa in _CODON_TABLE.values()
                         if aa not in ("*", "M", "W")})

# Codon lists per AA
_AA_TO_CODONS = {}
for codon, aa in _CODON_TABLE.items():
    _AA_TO_CODONS.setdefault(aa, []).append(codon)


# ════════════════════════════════════════════════════════════════════════════
# Compute reference codon proportions per AA
# ════════════════════════════════════════════════════════════════════════════

def _reference_proportions_per_aa() -> dict[str, dict[str, float]]:
    """
    From the Kazusa per-thousand frequencies, compute the fraction of each AA
    that is encoded by each of its codons. Returns {aa: {codon: proportion}}.
    """
    out = {}
    for aa in MULTI_CODON_AA:
        codons = _AA_TO_CODONS[aa]
        total = sum(HUMAN_CODON_USAGE[c] for c in codons)
        out[aa] = {c: HUMAN_CODON_USAGE[c] / total for c in codons}
    return out


REFERENCE_PROPORTIONS = _reference_proportions_per_aa()


# ════════════════════════════════════════════════════════════════════════════
# Extract codon counts from region_by_id
# ════════════════════════════════════════════════════════════════════════════

def compute_codon_counts(region_by_id: dict) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    For each region, count codons by AA. Returns a long-form DataFrame:
        region_id, group, aa, codon, count
    """
    rows = []
    for rid, region in region_by_id.items():
        dna = region.get("dna", "")
        group = region.get("group")
        if not dna or len(dna) % 3 != 0:
            continue
        codons = [dna[i:i + 3].upper() for i in range(0, len(dna), 3)]
        for codon in codons:
            aa = _CODON_TABLE.get(codon)
            if aa is None or aa == "*":
                continue
            rows.append({"region_id": rid, "group": group,
                         "aa": aa, "codon": codon})
    df = pd.DataFrame(rows)
    counts = (
        df.groupby(["group", "aa", "codon"])
          .size()
          .reset_index(name="count")
    )
    return counts


# ════════════════════════════════════════════════════════════════════════════
# Statistical test per AA
# ════════════════════════════════════════════════════════════════════════════

def test_codon_usage_pos_vs_neg(codon_counts: pd.DataFrame) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    For each multi-codon AA, chi² test on the codon distribution between pos
    and neg. BH-FDR correction across the tested AAs.

    Returns a DataFrame with one row per AA: aa, chi2, dof, p_raw, p_fdr, sig.
    """
    rows = []
    for aa in MULTI_CODON_AA:
        sub = codon_counts[codon_counts["aa"] == aa]
        pivot = sub.pivot_table(
            index="group", columns="codon", values="count",
            aggfunc="sum", fill_value=0,
        )
        # Need both groups to be tested
        if not {"pos", "neg"}.issubset(set(pivot.index)):
            rows.append({"aa": aa, "chi2": np.nan, "dof": np.nan,
                         "p_raw": np.nan, "n_pos": 0, "n_neg": 0})
            continue
        table = pivot.reindex(index=["pos", "neg"]).values
        if table.sum() == 0 or (table.sum(axis=1) == 0).any():
            rows.append({"aa": aa, "chi2": np.nan, "dof": np.nan,
                         "p_raw": np.nan,
                         "n_pos": int(table[0].sum()),
                         "n_neg": int(table[1].sum())})
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        rows.append({
            "aa": aa, "chi2": float(chi2), "dof": int(dof),
            "p_raw": float(p),
            "n_pos": int(table[0].sum()),
            "n_neg": int(table[1].sum()),
        })

    out = pd.DataFrame(rows)
    # FDR correction across tested AAs only
    valid_mask = out["p_raw"].notna()
    if valid_mask.sum() > 0:
        _, corrected, _, _ = multipletests(
            out.loc[valid_mask, "p_raw"].values, method="fdr_bh",
        )
        out.loc[valid_mask, "p_fdr"] = corrected
    else:
        out["p_fdr"] = np.nan
    out["sig"] = out["p_fdr"].apply(
        lambda p: significance_stars(p) if pd.notna(p) else "n.s."
    )
    return out


def test_codon_usage_all_pairs(codon_counts: pd.DataFrame) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    For each multi-codon AA, three pairwise chi² tests:
        pos vs neg
        pos vs reference (Kazusa human)
        neg vs reference

    The reference has no "raw counts" — we scale its proportions to match
    each group's total to make chi² meaningful.

    BH-FDR correction is applied across ALL tests (3 × 18 = 54 tests max).

    Returns a DataFrame with one row per (aa, comparison):
        aa, comparison, chi2, dof, p_raw, p_fdr, sig, n_a, n_b
    """
    rows = []
    for aa in MULTI_CODON_AA:
        sub = codon_counts[codon_counts["aa"] == aa]
        codons = _AA_TO_CODONS[aa]

        # Pivot to group × codon counts
        pivot = sub.pivot_table(
            index="group", columns="codon", values="count",
            aggfunc="sum", fill_value=0,
        ).reindex(columns=codons, fill_value=0)

        pos_counts = pivot.loc["pos"].values if "pos" in pivot.index else np.zeros(len(codons))
        neg_counts = pivot.loc["neg"].values if "neg" in pivot.index else np.zeros(len(codons))

        n_pos = int(pos_counts.sum())
        n_neg = int(neg_counts.sum())

        # Reference: proportions scaled to match the comparison group's total
        ref_proportions = np.array([REFERENCE_PROPORTIONS[aa][c] for c in codons])

        def _pair_chi2(a, b, n_a, n_b):
            table = np.vstack([a, b])
            # Require both rows to sum to >0 and all row sums > 0
            if table.sum() == 0 or (table.sum(axis=1) == 0).any():
                return np.nan, np.nan, np.nan
            try:
                chi2, p, dof, _ = chi2_contingency(table)
                return float(chi2), int(dof), float(p)
            except Exception:
                return np.nan, np.nan, np.nan

        # pos vs neg — straight chi²
        chi2_pn, dof_pn, p_pn = _pair_chi2(pos_counts, neg_counts, n_pos, n_neg)
        rows.append({
            "aa": aa, "comparison": "pos_vs_neg",
            "chi2": chi2_pn, "dof": dof_pn, "p_raw": p_pn,
            "n_a": n_pos, "n_b": n_neg,
        })

        # pos vs reference — scale reference proportions to match pos total
        ref_vs_pos = ref_proportions * n_pos
        chi2_pr, dof_pr, p_pr = _pair_chi2(pos_counts, ref_vs_pos, n_pos, n_pos)
        rows.append({
            "aa": aa, "comparison": "pos_vs_ref",
            "chi2": chi2_pr, "dof": dof_pr, "p_raw": p_pr,
            "n_a": n_pos, "n_b": n_pos,
        })

        # neg vs reference — scale reference proportions to match neg total
        ref_vs_neg = ref_proportions * n_neg
        chi2_nr, dof_nr, p_nr = _pair_chi2(neg_counts, ref_vs_neg, n_neg, n_neg)
        rows.append({
            "aa": aa, "comparison": "neg_vs_ref",
            "chi2": chi2_nr, "dof": dof_nr, "p_raw": p_nr,
            "n_a": n_neg, "n_b": n_neg,
        })

    out = pd.DataFrame(rows)
    # BH-FDR across all tests (54 max)
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
# Plotting: stacked bars per AA with reference comparison
# ════════════════════════════════════════════════════════════════════════════

GROUP_COLORS_PALE = {"pos": "#B8DFAA", "neg": "#E5BEBE"}


def _codon_palette(n: int) -> list[tuple[float, float, float]]:
    """Categorical palette for codons within a single AA."""
    base = sns.color_palette("colorblind", n_colors=max(n, 6))
    return list(base[:n])


def _text_color_for_bg(rgb: tuple[float, float, float]) -> str:
    """Return 'white' or 'black' for readable text on a given RGB color."""
    r, g, b = rgb[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.55 else "black"


def plot_codon_usage(
    codon_counts: pd.DataFrame,
    test_results: pd.DataFrame,
    dataset: str = "gnomad",
    save: bool = True,
    ncols: int = 3,
    source_aas: list[str] | None = None,
    sig_only: bool = False,
    fdr_threshold: float = 0.05,
    show_reference: bool = True,
    title_suffix: str = "",
) -> plt.Figure | None:
    """
    [Dataset-agnostic]
    Stacked-bar codon usage per AA. Supports filtering to a subset of AAs
    via `source_aas` (explicit list) or `sig_only` (auto-select significant).

    Parameters
    ----------
    source_aas : list of AA letters to plot. Overrides sig_only.
    sig_only   : if True and source_aas is None, plot only AAs with p_fdr < threshold.
    show_reference : include the Kazusa reference column.
    """
    # ── Decide which AAs to plot ─────────────────────────────────────────
    if source_aas is not None:
        aas = [aa for aa in source_aas if aa in MULTI_CODON_AA]
        # print(source_aas)
        if not aas:
            print("None of the requested AAs are multi-codon. Nothing to plot.")

            return None
    elif sig_only:
        sig_aas = test_results.loc[
            test_results["p_fdr"] < fdr_threshold, "aa"
        ].tolist()
        if not sig_aas:
            print(f"No AAs significant at FDR < {fdr_threshold}. Nothing to plot.")
            return None
        aas = [aa for aa in MULTI_CODON_AA if aa in sig_aas]
    else:
        aas = MULTI_CODON_AA

    # ── Layout ───────────────────────────────────────────────────────────
    n = len(aas)
    ncols_eff = min(ncols, n)
    nrows = int(np.ceil(n / ncols_eff))
    # Slightly wider per-panel to give room for in-bar codon labels
    fig, axes = plt.subplots(
        nrows, ncols_eff,
        figsize=(ncols_eff * 2.6, nrows * 3),
        squeeze=False,
    )
    axes = axes.flatten()

    # Safe lookup (avoid DataFrame-on-duplicate-index issue)
    results_lookup = (
        test_results.drop_duplicates(subset="aa").set_index("aa")
    )

    for i, aa in enumerate(aas):
        ax = axes[i]
        codons = sorted(_AA_TO_CODONS[aa])  # stable ordering across runs

        # Proportions per group
        sub = codon_counts[codon_counts["aa"] == aa]
        props, totals = {}, {}
        for group in ("pos", "neg"):
            grp = sub[sub["group"] == group]
            total = int(grp["count"].sum())
            totals[group] = total
            if total == 0:
                props[group] = {c: 0.0 for c in codons}
            else:
                props[group] = {
                    c: int(grp.loc[grp["codon"] == c, "count"].sum()) / total
                    for c in codons
                }
        props["reference"] = {
            c: REFERENCE_PROPORTIONS[aa].get(c, 0.0) for c in codons
        }

        # Bar setup
        if show_reference:
            bar_keys   = ["pos", "reference", "neg"]
            bar_labels = [
                f"pos\nn={totals['pos']:,}",
                "ref\n(Kazusa et al.\n(2007))",
                f"neg\nn={totals['neg']:,}",
            ]
            sig_x = (0, 2)
        else:
            bar_keys   = ["pos", "neg"]
            bar_labels = [
                f"pos\nn={totals['pos']:,}",
                f"neg\nn={totals['neg']:,}",
            ]
            sig_x = (0, 1)

        x = np.arange(len(bar_keys))
        palette = _codon_palette(len(codons))

        bottoms = np.zeros(len(bar_keys))
        for c_idx, codon in enumerate(codons):
            vals = np.array([props[k][codon] for k in bar_keys])
            color = palette[c_idx]
            ax.bar(
                x, vals, bottom=bottoms,
                color=color, edgecolor="white", linewidth=0.4,
                label=codon, width=0.72,
            )
            # In-segment label only when segment is tall enough to be readable
            for b_idx, (val, bot) in enumerate(zip(vals, bottoms)):
                if val >= 0.05:
                    ax.text(
                        x[b_idx], bot + val / 2, codon,
                        ha="center", va="center",
                        fontsize=6, color=_text_color_for_bg(color),
                        fontweight="medium",
                    )
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=7)
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "", "0.5", "", "1"], fontsize=6)
        ax.tick_params(axis="y", length=2)

        # Significance bracket — pos vs neg only
        row = results_lookup.loc[aa] if aa in results_lookup.index else None
        if row is not None and pd.notna(row["p_fdr"]):
            sig = row["sig"]
            p = float(row["p_fdr"])
            is_sig = p < fdr_threshold
        else:
            sig = "n.s."
            p = np.nan
            is_sig = False

        y_bar = 1.06
        ax.plot([sig_x[0], sig_x[1]], [y_bar, y_bar],
                color="black", lw=0.6)
        # Small caps at bracket ends
        for xb in sig_x:
            ax.plot([xb, xb], [y_bar - 0.02, y_bar],
                    color="black", lw=0.6)
        ax.text(np.mean(sig_x), y_bar + 0.01, sig,
                ha="center", va="bottom", fontsize=8,
                fontweight="bold" if is_sig else "normal")

        # Panel title
        if pd.notna(p):
            p_text = f"  p={p:.1e}"
        else:
            p_text = ""
        ax.set_title(
            f"{aa}\n{p_text}",
            fontsize=10,
            fontweight="bold" if is_sig else "normal",
            color="black",
        )

        # Y-label only on left column
        if i % ncols_eff == 0:
            ax.set_ylabel("Codon proportion", fontsize=8)
        else:
            ax.set_ylabel("")

        # Per-AA legend in the panel — small and out of the way (below x-labels)
        # Codons that didn't get an in-bar label still need to be identifiable.
        # We add a compact legend strip above the title for AAs with many codons.
        # if len(codons) > 2:
        #     handles = [
        #         plt.Rectangle((0, 0), 1, 1, color=palette[c_idx])
        #         for c_idx in range(len(codons))
        #     ]
        #     ax.legend(
        #         handles, codons,
        #         fontsize=5.5, ncol=min(len(codons), 3),
        #         loc="upper center", bbox_to_anchor=(0.5, -0.22),
        #         frameon=False, handlelength=1.0, handleheight=0.8,
        #         columnspacing=0.8, handletextpad=0.4,
        #     )

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Hide unused subplots
    for j in range(len(aas), len(axes)):
        axes[j].set_visible(False)

    # Title
    sub_title = "all AAs"
    if source_aas is not None:
        sub_title = f"selected AAs: {', '.join(aas)}"
    elif sig_only:
        sub_title = f"significant AAs (FDR < {fdr_threshold})"
    title = (
        f"Codon usage per amino acid ({dataset}) — {sub_title}\n"
        f"pos vs neg χ² with BH-FDR across {len(MULTI_CODON_AA)} multi-codon AAs"
    )
    if show_reference:
        title += "; reference = Kazusa et al. (2007) human"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=11, y=1.00)

    plt.tight_layout()

    if save:
        suffix = "all"
        if source_aas is not None:
            suffix = "selected"
        elif sig_only:
            suffix = "sig"
        save_figure(fig, f"codon_usage_{suffix}", dataset=dataset)

    # ── Printed summary ──────────────────────────────────────────────────
    print(f"\n── Codon usage χ² tests ({dataset}) ──")
    out_table = test_results.sort_values("p_fdr").reset_index(drop=True)
    print(out_table[
        ["aa", "n_pos", "n_neg", "chi2", "p_raw", "p_fdr", "sig"]
    ].to_string(index=False))
    n_sig = int((out_table["p_fdr"] < fdr_threshold).sum())
    print(f"\n  {n_sig} / {len(out_table)} AAs show significantly different "
          f"codon usage between pos and neg after FDR (< {fdr_threshold}).")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_codon_usage_analysis(
    region_by_id: dict,
    dataset: str = "gnomad",
    save: bool = True,
    source_aas: list[str] | None = None,
    sig_only: bool = False,
    fdr_threshold: float = 0.05,
    show_reference: bool = True,
) -> dict:
    """
    [Dataset-agnostic]
    End-to-end codon usage analysis. Returns dict with codon_counts and
    test_results so you can call the plot multiple times with different
    AA selections without recomputing.
    """
    codon_counts = compute_codon_counts(region_by_id)
    test_results = test_codon_usage_pos_vs_neg(codon_counts)

    # plot_codon_usage(
    #     codon_counts, test_results,
    #     dataset=dataset, save=save,
    #     source_aas=source_aas, sig_only=sig_only,
    #     fdr_threshold=fdr_threshold,
    #     show_reference=show_reference,
    # )

    return {"codon_counts": codon_counts, "test_results": test_results}