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
# Plotting: stacked bars per AA with reference line
# ════════════════════════════════════════════════════════════════════════════

GROUP_COLORS_PALE = {"pos": "#B8DFAA", "neg": "#E5BEBE"}


def _codon_palette(n: int) -> list[str]:
    """Return a categorical palette for codons within a single AA."""
    # Use seaborn's colorblind palette extended if needed
    base = sns.color_palette("colorblind", n_colors=max(n, 6))
    return list(base[:n])

def plot_codon_usage(
    codon_counts: pd.DataFrame,
    test_results: pd.DataFrame,
    dataset: str = "gnomad",
    save: bool = True,
    ncols: int = 6,
) -> plt.Figure:
    """
    [Dataset-agnostic]
    One subplot per multi-codon AA. Three stacked bars per subplot:
        pos (regions) | reference (Kazusa human) | neg (regions)
    Only pos vs neg is tested statistically; the reference is visual context.
    Sample sizes (n_pos, n_neg) shown below the bars.
    """
    aas = MULTI_CODON_AA
    nrows = int(np.ceil(len(aas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.3, nrows * 2.4),
                              constrained_layout=True)
    axes = axes.flatten()

    results_lookup = test_results.set_index("aa")

    for i, aa in enumerate(aas):
        ax = axes[i]
        codons = _AA_TO_CODONS[aa]

        # Proportions per group
        sub = codon_counts[codon_counts["aa"] == aa]
        props = {}
        totals = {}
        for group in ("pos", "neg"):
            grp = sub[sub["group"] == group]
            total = grp["count"].sum()
            totals[group] = int(total)
            if total == 0:
                props[group] = {c: 0 for c in codons}
            else:
                props[group] = {
                    c: grp.loc[grp["codon"] == c, "count"].sum() / total
                    for c in codons
                }
        props["reference"] = REFERENCE_PROPORTIONS[aa]

        # Order: pos | ref | neg
        bar_labels = [
            f"pos\nn={totals['pos']:,}",
            "ref",
            f"neg\nn={totals['neg']:,}",
        ]
        bar_proportions = [props["pos"], props["reference"], props["neg"]]
        x = np.arange(len(bar_labels))

        palette = _codon_palette(len(codons))
        bottoms = np.zeros(len(bar_labels))
        for c_idx, codon in enumerate(codons):
            vals = np.array([p[codon] for p in bar_proportions])
            ax.bar(x, vals, bottom=bottoms,
                   color=palette[c_idx], edgecolor="black", linewidth=0.3,
                   label=codon, width=0.75)
            # Label codon inside segment if tall enough
            for b_idx, (val, bot) in enumerate(zip(vals, bottoms)):
                if val >= 0.10:
                    ax.text(
                        x[b_idx], bot + val / 2, codon,
                        ha="center", va="center",
                        fontsize=5.5, color="white" if c_idx % 2 else "black",
                    )
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=6.5)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"], fontsize=6)

        # Significance annotation (pos vs neg only) — bracket above the outer bars
        if aa in results_lookup.index:
            row = results_lookup.loc[aa]
            sig = row["sig"]
            p = row["p_fdr"]
        else:
            sig = "n.s."
            p = np.nan

        # Bracket from bar 0 (pos) to bar 2 (neg)
        y_bar = 1.04
        ax.plot([0, 2], [y_bar, y_bar], color="black", lw=0.5)
        ax.text(1, y_bar + 0.01, sig, ha="center", va="bottom", fontsize=7)

        # Title: AA letter, bold if significant
        weight = "bold" if sig not in ("n.s.", "—", "") else "normal"
        p_text = f" (p={p:.1e})" if pd.notna(p) else ""
        ax.set_title(f"{aa}{p_text}", fontsize=7.5, fontweight=weight)

        if i % ncols != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Codon proportion", fontsize=7)

        ax.legend(fontsize=5, ncol=1, loc="center left",
                  bbox_to_anchor=(1.0, 0.5), frameon=False)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Hide unused subplots
    for j in range(len(aas), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Codon usage per amino acid ({dataset})\n"
        f"pos vs neg chi² with BH-FDR across {len(aas)} AAs; "
        f"reference = Kazusa human",
        fontsize=11,
    )
    if save:
        save_figure(fig, "codon_usage", dataset=dataset)

    # ── Printed summary ────────────────────────────────────────────────────
    print(f"\n── Codon usage chi² tests ({dataset}) ──")
    out_table = test_results.sort_values("p_fdr").reset_index(drop=True)
    print(out_table[["aa", "n_pos", "n_neg", "chi2", "p_raw",
                     "p_fdr", "sig"]].to_string(index=False))
    n_sig = int((out_table["p_fdr"] < 0.05).sum())
    print(f"\n  {n_sig} / {len(out_table)} amino acids show significantly "
          f"different codon usage between pos and neg after FDR.")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_codon_usage_analysis(
    region_by_id: dict,
    dataset: str = "gnomad",
    save: bool = True,
    ) -> dict:
    codon_counts = compute_codon_counts(region_by_id)
    test_results = test_codon_usage_pos_vs_neg(codon_counts)  # ← original
    plot_codon_usage(codon_counts, test_results, dataset=dataset, save=save)
    # ... rest stays the same