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
from scipy.stats import wilcoxon, mannwhitneyu

from src.analysis_visualization.plot_config import (
    GROUP_COLORS, save_figure, significance_stars,
)
# The enumeration function lives in rg_analysis.py
from src.analysis_visualization.rg_analysis import (
    enumerate_single_nt_substitutions,
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



def _seq_metrics(dna: str):
    """GC fraction, CpG fraction, and CpG observed/expected for one sequence.
    Measured on the coding strand; CpG is strand-symmetric so no rev-comp needed.
    expected CpG count = (#C * #G) / length  (standard CpG-island convention)."""
    s = dna.upper(); L = len(s)
    if L < 2:
        return np.nan, np.nan, np.nan
    nC, nG, nCpG = s.count("C"), s.count("G"), s.count("CG")
    gc = (nC + nG) / L
    cpg_frac = nCpG / (L - 1)
    exp = (nC * nG) / L
    oe = (nCpG / exp) if exp > 0 else np.nan
    return gc, cpg_frac, oe


def compute_gc_cpg_by_group(region_by_id: dict, group_col="group",
                            pos_label="pos", neg_label="neg"):
    """Per-region GC%, CpG fraction, CpG O/E + per-group Mann-Whitney (region = unit)."""
    rows = []
    for rid, r in region_by_id.items():
        dna = r.get("dna", "")
        if not dna:
            continue
        gc, cf, oe = _seq_metrics(dna)
        rows.append(dict(region_id=rid, group=r.get(group_col),
                         length=len(dna), gc=gc, cpg_frac=cf, cpg_oe=oe))
    df = pd.DataFrame(rows)
    stats = {}
    for metric in ["gc", "cpg_frac", "cpg_oe"]:
        p = df.loc[df.group == pos_label, metric].dropna()
        n = df.loc[df.group == neg_label, metric].dropna()
        U, pval = mannwhitneyu(p, n, alternative="two-sided")
        stats[metric] = dict(pos_median=p.median(), neg_median=n.median(),
                             pos_mean=p.mean(), neg_mean=n.mean(),
                             n_pos=len(p), n_neg=len(n), U=U, pval=pval)
    return df, stats

# literature reference values (sourced; see notes below)
GC_CPG_REF = {
    "gc":     {"cgi_threshold": 0.50, "genome_avg": 0.41},   # CGI >=50%; human genome ~41%
    "cpg_oe": {"cgi_threshold": 0.60, "genome_bulk": 0.20},   # CGI >=0.6; bulk genome ~0.2
}


def plot_gc_cpg_by_group(df, stats, dataset="gnomad", save=True,
                         pos_label="pos", neg_label="neg"):
    """Three-panel composition figure (GC%, CpG fraction, CpG O/E): per-region
    points + boxplots, Mann-Whitney stars, and sourced reference lines.
    Consumes the (df, stats) returned by compute_gc_cpg_by_group."""
    metrics = [("gc", "GC content", "fraction G+C"),
               ("cpg_frac", "CpG fraction", "CpG / dinucleotide"),
               ("cpg_oe", "CpG observed/expected", "O/E ratio")]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.25))
    pc, nc = GROUP_COLORS["pos"], GROUP_COLORS["neg"]

    for ax, (key, title, ylab) in zip(axes, metrics):
        p = df.loc[df.group == pos_label, key].dropna()
        n = df.loc[df.group == neg_label, key].dropna()
        for i, (vals, col) in enumerate([(p, pc), (n, nc)]):
            x = np.random.normal(i*2, 0.06, len(vals))

            ax.scatter(x, vals, s=10, color=col, alpha=0.35, edgecolor="none", zorder=2)
            ax.boxplot(vals, positions=[i*2], widths=0.5, showfliers=False,


                       medianprops=dict(color="black", lw=1.5), zorder=3)
        ax.set_xticks([0, 2])
        ax.set_xticklabels([f"pos\n(n={len(p)})", f"neg\n(n={len(n)})"])
        ax.set_title(title, fontsize=11, pad=15)
        ax.set_ylabel(ylab, fontsize=9)
        ymax = max(p.max() if len(p) else 0, n.max() if len(n) else 0)
        # ymin = min(p.min() if len(p), n.min() if len(n))

        ax.set_ylim(None, ymax*1.1)



        ax.yaxis.grid(True, alpha=0.3, lw=0.5); ax.set_axisbelow(True)

        if key == "gc":
            ax.axhline(GC_CPG_REF["gc"]["cgi_threshold"], ls="--", color="#666", lw=1)
            ax.text(1, GC_CPG_REF["gc"]["cgi_threshold"], "CGI \u22650.50",
                    fontsize=7, color="#666", va="bottom", ha="center")
            ax.axhline(GC_CPG_REF["gc"]["genome_avg"], ls=":", color="#999", lw=1)
            ax.text(1, GC_CPG_REF["gc"]["genome_avg"], "genome ~0.41",
                    fontsize=7, color="#999", va="bottom", ha="center")
        if key == "cpg_oe":
            ax.axhline(GC_CPG_REF["cpg_oe"]["cgi_threshold"], ls="--", color="#666", lw=1)
            ax.text(1, GC_CPG_REF["cpg_oe"]["cgi_threshold"], "CGI \u22650.60",
                    fontsize=7, color="#666", va="bottom", ha="center")
            ax.axhline(GC_CPG_REF["cpg_oe"]["genome_bulk"], ls=":", color="#999", lw=1)
            ax.text(1, GC_CPG_REF["cpg_oe"]["genome_bulk"], "bulk ~0.20",
                    fontsize=7, color="#999", va="bottom", ha="center")

        pv = stats[key]["pval"]
        star = "***" if pv < 1e-3 else "**" if pv < 1e-2 else "*" if pv < 0.05 else "n.s."

        ax.text(1, ymax*1.05, f"{star}  (p={pv:.1e})", ha="center", va="top", fontsize=8)

    fig.suptitle(f"Sequence composition: pos vs neg regions ({dataset})", fontsize=12, y=1.02)
    plt.tight_layout()
    if save:
        save_figure(fig, "gc_cpg_composition", dataset=dataset)
    return fig




from scipy.stats import mannwhitneyu

# standard genetic code (stops included; excluded from mutability)
_BASES = "ACGT"


def codon_intrinsic_mutability(rates: dict) -> dict:
    """
    Per-codon total mutability = sum over its 3 positions x 3 alternative bases of
    the 1KG mutation rate. The middle base uses the exact codon-internal
    trinucleotide context; the two terminal bases marginalise the unknown flanking
    base over A/C/G/T (the same approximation for every codon, so it cancels in
    pos-vs-neg comparisons). Internal CpGs (e.g. arginine CGN) are captured exactly.
    Returns {codon: total_mutability}.
    """
    out = {}
    for codon, aa in _CODON_TABLE.items():
        if aa == "*":
            continue
        total = 0.0
        for pos in range(3):
            ref = codon[pos]
            for alt in _BASES:
                if alt == ref:
                    continue
                if pos == 1:
                    r = rates.get((codon, alt))            # exact internal context
                    if r:
                        total += r
                elif pos == 0:
                    vals = [rates.get((f + codon[0] + codon[1], alt)) for f in _BASES]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        total += np.mean(vals)
                else:  # pos == 2
                    vals = [rates.get((codon[1] + codon[2] + f, alt)) for f in _BASES]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        total += np.mean(vals)
        out[codon] = total
    return out


def _codons(dna: str) -> list[str]:
    dna = dna.upper()
    return [dna[i:i + 3] for i in range(0, len(dna) - 2, 3)]


def compute_codon_mutability_by_group(region_by_id: dict, rates: dict,
                                      ref_usage: dict | None = None,
                                      group_col="group", pos_label="pos", neg_label="neg"):
    """
    Per region, per amino acid: mean intrinsic mutability of the codons used for
    that AA. Aggregates per group with a per-AA Mann-Whitney (region = unit).
    ref_usage: optional {codon: relative_freq_within_its_AA} (e.g. Kazusa) -> a
    single reference mutability per AA, shown as a line in the plot.
    Returns (df, stats, codon_mutability_dict).
    """
    cm = codon_intrinsic_mutability(rates)
    if ref_usage is None:
        ref_usage = HUMAN_CODON_USAGE
    # normalize ref_usage to within-AA fractions (accepts raw Kazusa per-1000 values)
    if ref_usage:
        aa_tot = {}
        for c, f in ref_usage.items():
            aa = _CODON_TABLE.get(c)
            if aa and aa != "*":
                aa_tot[aa] = aa_tot.get(aa, 0.0) + f
        ref_usage = {c: f / aa_tot[_CODON_TABLE[c]]
                     for c, f in ref_usage.items()
                     if _CODON_TABLE.get(c, "*") != "*" and aa_tot.get(_CODON_TABLE[c], 0) > 0}

    rows = []
    for rid, r in region_by_id.items():
        grp = r.get(group_col)
        dna = r.get("dna", "")
        if not dna:
            continue
        per_aa = {}
        for c in _codons(dna):
            aa = _CODON_TABLE.get(c)
            if aa is None or aa == "*" or c not in cm:
                continue
            per_aa.setdefault(aa, []).append(cm[c])
        for aa, vals in per_aa.items():
            rows.append(dict(region_id=rid, group=grp, aa=aa,
                             mean_mut=np.mean(vals), n=len(vals)))
    df = pd.DataFrame(rows)

    # reference (e.g. Kazusa): usage-weighted mean mutability per AA
    ref_val = {}
    if ref_usage:
        aa_codons = {}
        for c, aa in _CODON_TABLE.items():
            if aa != "*":
                aa_codons.setdefault(aa, []).append(c)
        for aa, codons in aa_codons.items():
            w = np.array([ref_usage.get(c, 0.0) for c in codons])
            m = np.array([cm.get(c, 0.0) for c in codons])
            ref_val[aa] = (w @ m) / w.sum() if w.sum() > 0 else np.nan

    stats = {}
    for aa in df["aa"].unique():
        p = df[(df.group == pos_label) & (df.aa == aa)]["mean_mut"]
        n = df[(df.group == neg_label) & (df.aa == aa)]["mean_mut"]
        if len(p) >= 5 and len(n) >= 5:
            U, pv = mannwhitneyu(p, n, alternative="two-sided")
        else:
            pv = np.nan
        stats[aa] = dict(pos_med=p.median(), neg_med=n.median(),
                         n_pos=len(p), n_neg=len(n), pval=pv,
                         ref=ref_val.get(aa, np.nan))
    return df, stats, cm


def plot_codon_mutability(df, stats, cm, amino_acids, dataset="gnomad", save=True,
                          pos_label="pos", neg_label="neg", seed=0):
    """Per-AA panels: pos/neg per-region mean codon mutability (points + box),
    Mann-Whitney stars, and an optional Kazusa reference line."""
    np.random.seed(seed)  # reproducible jitter
    aas = [a for a in amino_acids if a in df["aa"].unique()]
    n = len(aas)
    fig, axes = plt.subplots(n, 1, figsize=(3,n * 2.6), sharey=False)
    if n == 1:
        axes = [axes]
    pc, nc = GROUP_COLORS["pos"], GROUP_COLORS["neg"]

    for ax, aa in zip(axes, aas):
        p = df[(df.group == pos_label) & (df.aa == aa)]["mean_mut"]
        nn = df[(df.group == neg_label) & (df.aa == aa)]["mean_mut"]
        for i, (vals, col) in enumerate([(p, pc), (nn, nc)]):
            x = np.random.normal(i, 0.06, len(vals))
            ax.scatter(x, vals, s=10, color=col, alpha=0.3, edgecolor="none", zorder=2)
            ax.boxplot(vals, positions=[i], widths=0.5, showfliers=False,
                       medianprops=dict(color="black", lw=1.4), zorder=3)
        ref = stats[aa]["ref"]
        if not np.isnan(ref):
            ax.axhline(ref, ls="--", color="#444", lw=1.1, zorder=4)
            ax.text(0.5, ref, "human\naverage", fontsize=7.5, color="#444", va="bottom", ha="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pos", "neg"], fontsize=9)
        pv = stats[aa]["pval"]
        star = ("***" if pv < 1e-3 else "**" if pv < 1e-2 else "*" if pv < 0.05 else "n.s.") \
            if not np.isnan(pv) else ""
        ax.set_title(f"{aa}\n{star}" if not np.isnan(pv) else aa, fontsize=10) #  (p={pv:.1e})
        ax.yaxis.grid(True, alpha=0.3, lw=0.5); ax.set_axisbelow(True)
        # if ax is axes[0]:
        ax.set_ylabel("mean codon mutability\n(\u03a3 1KG rate / codon)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(f"Per-amino-acid codon mutability: pos vs neg ({dataset})", fontsize=12, y=1.03)
    plt.tight_layout()
    if save:
        save_figure(fig, "codon_mutability", dataset=dataset)
    return fig




def delta_gc(ref: str, alt: str) -> int:
    """Net GC bases gained by a single-nt substitution: +1 / 0 / -1. Strand-invariant."""
    return int(alt in "GC") - int(ref in "GC")


def delta_cpg(context: str | None, alt: str) -> int:
    """Change in CpG dinucleotide count from changing the central base of `context`
    (= [5'flank, ref, 3'flank]) to `alt`. Covers the two dinucleotides the central
    base sits in. Strand-symmetric -> coding-strand context is valid."""
    if context is None or len(context) != 3:
        return 0
    l, r, rt = context[0], context[1], context[2]
    old = int(l == "C" and r == "G") + int(r == "C" and rt == "G")
    new = int(l == "C" and alt == "G") + int(alt == "C" and rt == "G")
    return new - old


def _parse_codons(field):
    """VEP Codons 'Gga/Aga' -> (ref_codon, alt_codon, offset, coding_ref, coding_alt)."""
    if not isinstance(field, str) or "/" not in field:
        return None
    a, b = field.split("/")[:2]
    a, b = a.strip(), b.strip()
    if len(a) != 3 or len(b) != 3:
        return None
    off = next((i for i in range(3) if a[i] != b[i]), None)
    if off is None:
        return None
    return a.upper(), b.upper(), off, a[off].upper(), b[off].upper()


def compute_gc_cpg_flux(df, region_by_id, rates, enumerate_fn,
                        group_col="group", pos_label="pos", neg_label="neg",
                        consequence_col="Consequence", consequence="missense_variant",
                        region_id_col="region_id", ppos_col="protein_position_int",
                        region_start_col="region_start_aa", codons_col="Codons",
                        min_obs=3):
    """
    Per region: observed vs rate-weighted-EXPECTED mean ΔGC and ΔCpG per variant.
      expected = neutral mutational flux (enumerate all possible missense single-nt
                 changes, weight each by its 1KG rate).
      observed = mean over the region's actual variants.
    The tested quantity is per-region (obs - exp): departure from neutral flux.

    Returns (merged_df, stats, match_rate). match_rate is the fraction of observed
    variants whose dna position matched the coding ref base -> MUST be ~1.0 to trust
    the ΔCpG (mapping/off-by-one check). ΔGC does not depend on the mapping.
    """
    # ---------- EXPECTED: rate-weighted over all possible missense single-nt changes ----------
    exp_rows = []
    for rid, r in region_by_id.items():
        dna = r.get("dna", "")
        prot = r.get("prot_seq", "")
        if not dna or not prot or len(dna) != 3 * len(prot):
            continue
        enum = enumerate_fn(dna, prot)
        miss = enum[enum["consequence"] == "missense"]
        w, dgc, dcpg = [], [], []
        for row in miss.itertuples(index=False):
            ctx = getattr(row, "context", None)
            if ctx is None:
                continue
            rate = rates.get((ctx, row.alt_base))
            if rate is None:
                continue
            w.append(rate)
            dgc.append(delta_gc(row.ref_base, row.alt_base))
            dcpg.append(delta_cpg(ctx, row.alt_base))
        if not w:
            continue
        w = np.array(w)
        exp_rows.append(dict(region_id=rid, group=r.get(group_col),
                             exp_dgc=np.average(dgc, weights=w),
                             exp_dcpg=np.average(dcpg, weights=w)))
    exp_df = pd.DataFrame(exp_rows).set_index("region_id")

    # ---------- OBSERVED: per variant, mapped to coding context for ΔCpG ----------
    mis = df[df[consequence_col].fillna("").str.contains(consequence)].copy()
    obs_rows = []
    n_ctx_ok = n_ctx_tot = 0
    for rid, sub in mis.groupby(region_id_col):
        if rid not in region_by_id:
            continue
        dna = region_by_id[rid]["dna"].upper()
        start = sub[region_start_col].iloc[0]
        dgc, dcpg = [], []
        for _, v in sub.iterrows():
            pc = _parse_codons(v.get(codons_col))
            if pc is None:
                continue
            _, _, off, cref, calt = pc
            dgc.append(delta_gc(cref, calt))            # strand-invariant; from Codons directly
            try:
                ci = int(v[ppos_col]) - int(start) - 1
            except Exception:
                continue
            dpos = 3 * ci + off
            n_ctx_tot += 1
            if 0 < dpos < len(dna) - 1:
                ctx = dna[dpos - 1:dpos + 2]
                if dna[dpos] == cref:                    # sanity: dna matches coding ref
                    n_ctx_ok += 1
                    dcpg.append(delta_cpg(ctx, calt))
        if len(dgc) >= min_obs:
            obs_rows.append(dict(region_id=rid,
                                 obs_dgc=np.mean(dgc),
                                 obs_dcpg=np.mean(dcpg) if dcpg else np.nan,
                                 n_obs=len(dgc)))
    obs_df = pd.DataFrame(obs_rows).set_index("region_id")
    match_rate = (n_ctx_ok / n_ctx_tot) if n_ctx_tot else np.nan

    merged = exp_df.join(obs_df, how="inner")
    merged["diff_dgc"] = merged["obs_dgc"] - merged["exp_dgc"]
    merged["diff_dcpg"] = merged["obs_dcpg"] - merged["exp_dcpg"]

    # ---------- tests ----------
    stats = {}
    for metric, col in [("dgc", "diff_dgc"), ("dcpg", "diff_dcpg")]:
        st = {}
        for lab in [pos_label, neg_label]:
            v = merged.loc[merged.group == lab, col].dropna()
            st[lab] = dict(median=v.median(), mean=v.mean(), n=len(v),
                           wilcoxon_p=wilcoxon(v).pvalue if len(v) >= 6 else np.nan)
        p = merged.loc[merged.group == pos_label, col].dropna()
        n = merged.loc[merged.group == neg_label, col].dropna()
        st["between_mwu_p"] = (mannwhitneyu(p, n, alternative="two-sided").pvalue
                               if len(p) >= 5 and len(n) >= 5 else np.nan)
        stats[metric] = st
    return merged, stats, match_rate


def plot_gc_cpg_flux(merged, stats, dataset="gnomad", save=True,
                     pos_label="pos", neg_label="neg", seed=0):
    """Two panels (ΔGC, ΔCpG): per-region (obs - exp) flux, pos vs neg, neutral line at 0.
    Within-group 'vs 0' = Wilcoxon signed-rank; between-group = Mann-Whitney."""
    np.random.seed(seed)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    pc, nc = GROUP_COLORS["pos"], GROUP_COLORS["neg"]
    panels = [("dgc", "diff_dgc", "\u0394GC flux  (obs \u2212 expected)"),
              ("dcpg", "diff_dcpg", "\u0394CpG flux  (obs \u2212 expected)")]
    for ax, (metric, col, title) in zip(axes, panels):
        groupvals = []
        for i, (lab, c) in enumerate([(pos_label, pc), (neg_label, nc)]):
            v = merged.loc[merged.group == lab, col].dropna()
            groupvals.append(v)
            x = np.random.normal(i, 0.06, len(v))
            ax.scatter(x, v, s=10, color=c, alpha=0.35, edgecolor="none", zorder=2)
            ax.boxplot(v, positions=[i], widths=0.5, showfliers=False,
                       medianprops=dict(color="black", lw=1.5), zorder=3)
        ax.axhline(0, color="#444", ls="--", lw=1.1, zorder=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels([pos_label, neg_label])
        ax.set_title(title, fontsize=10, pad=18)
        ax.set_ylabel("per-region (obs \u2212 exp) mean per variant", fontsize=8)
        ax.yaxis.grid(True, alpha=0.3, lw=0.5); ax.set_axisbelow(True)
        # within-group stars under each box
        ymin = ax.get_ylim()[0]
        for i, lab in enumerate([pos_label, neg_label]):
            wp = stats[metric][lab]["wilcoxon_p"]
            star = ("***" if wp < 1e-3 else "**" if wp < 1e-2 else "*" if wp < 0.05 else "n.s.") \
                if not np.isnan(wp) else ""
            ax.text(i, ymin, f"vs0:{star}", ha="center", va="bottom", fontsize=7, color="#555")
        bp = stats[metric]["between_mwu_p"]
        bstar = ("***" if bp < 1e-3 else "**" if bp < 1e-2 else "*" if bp < 0.05 else "n.s.") \
            if not np.isnan(bp) else ""
        ax.text(0.5, ax.get_ylim()[1], f"pos vs neg: {bstar}", ha="center", va="top", fontsize=7)
    fig.suptitle(f"GC / CpG flux: variant gain/loss vs neutral expectation ({dataset})",
                 fontsize=11, y=1.0)
    plt.tight_layout()
    if save:
        save_figure(fig, "gc_cpg_flux", dataset=dataset)
    return fig



 
def _gc3(dna: str) -> float:
    """GC fraction at third codon positions (wobble). NaN if no complete codons."""
    s = dna.upper()
    thirds = s[2::3]                      # 3rd base of each codon
    if len(thirds) == 0:
        return np.nan
    return (thirds.count("G") + thirds.count("C")) / len(thirds)
 
 
def _mean_intrinsic_mutability(dna: str, cm: dict) -> float:
    """Mean intrinsic mutability over the region's codons. NaN if none scorable."""
    s = dna.upper()
    vals = []
    for i in range(0, len(s) - 2, 3):
        c = s[i:i + 3]
        aa = _CODON_TABLE.get(c)
        if aa is None or aa == "*":
            continue
        if c in cm:
            vals.append(cm[c])
    return float(np.mean(vals)) if vals else np.nan
 
 
def compute_gc_codon_indices_per_region(region_by_id: dict,
                                        rates: dict = None) -> pd.DataFrame:
    """
    One row per region (keyed on region_id) with leak-free composition indices:
        gc        : overall GC fraction
        gc3       : GC fraction at 3rd codon positions (wobble bias)
        cpg_frac  : CpG dinucleotide fraction
        cpg_oe    : CpG observed/expected
        codon_mean_mutability : mean intrinsic 1KG mutability over the region's
                    codons (only if `rates` provided; else column omitted)
 
    NaN where a metric is undefined (e.g. sequence too short). No label contrast.
    """
    cm = codon_intrinsic_mutability(rates) if rates is not None else None
 
    rows = []
    for rid, r in region_by_id.items():
        dna = r.get("dna", "")
        if not dna:
            continue
        gc, cpg_frac, cpg_oe = _seq_metrics(dna)
        row = {
            "region_id": rid,
            "gc": gc,
            "gc3": _gc3(dna),
            "cpg_frac": cpg_frac,
            "cpg_oe": cpg_oe,
        }
        if cm is not None:
            row["codon_mean_mutability"] = _mean_intrinsic_mutability(dna, cm)
        rows.append(row)
    return pd.DataFrame(rows)
 
