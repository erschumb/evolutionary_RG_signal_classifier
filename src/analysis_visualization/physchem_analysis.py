"""
Physicochemical feature module for gnomAD variant analysis.

Save as: src/analysis_visualization/physchem_analysis.py

Features computed via localcider:
    ncpr         net charge per residue
    fcr          fraction charged residues
    kappa        charge patterning (Das & Pappu)
    hydropathy   mean Kyte-Doolittle hydropathy
    aromaticity  fraction of F, Y, W
    fraction_proline
    n_pos        count of R + K
    n_neg        count of D + E

For each variant (missense only), computes the change in each metric between
WT and mutant region sequences. Per-region mean deltas are then the classifier
features.

NOTE: localcider can be slow for large variant sets — we parallelise via
multiprocessing. Kappa is the slowest metric.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from multiprocessing import Pool
from tqdm.notebook import tqdm

from localcider.sequenceParameters import SequenceParameters as SeqParams

from src.analysis_visualization.plot_config import (
    FIGSIZE_SINGLE, GROUP_COLORS, save_figure, significance_stars,
)


# ════════════════════════════════════════════════════════════════════════════
# Feature definitions
# ════════════════════════════════════════════════════════════════════════════

PHYSCHEM_FEATURES = [
    "ncpr",
    "fcr",
    "kappa",
    "hydropathy",
    "aromaticity",
    "fraction_proline",
    "n_pos",
    "n_neg",
]


def _compute_features(seq: str) -> dict:
    """
    Compute all physchem features for a single sequence.
    Returns NaN for any feature that fails (e.g. kappa for no-charge sequences).
    """
    out = {f: np.nan for f in PHYSCHEM_FEATURES}
    if not isinstance(seq, str) or len(seq) < 2:
        return out
    if "*" in seq or "-" in seq:
        return out

    try:
        sp = SeqParams(seq)
    except Exception:
        return out

    # Each metric wrapped in its own try/except — kappa commonly fails
    try: out["ncpr"] = sp.get_NCPR()
    except Exception: pass

    try: out["fcr"] = sp.get_FCR()
    except Exception: pass

    try: out["kappa"] = sp.get_kappa()
    except Exception: pass

    try: out["hydropathy"] = sp.get_mean_hydropathy()
    except Exception: pass

    try:
        fracs = sp.get_amino_acid_fractions()
        out["aromaticity"] = fracs.get("F", 0) + fracs.get("Y", 0) + fracs.get("W", 0)
        out["fraction_proline"] = fracs.get("P", 0)
    except Exception: pass

    try: out["n_pos"] = sp.get_countPos()
    except Exception: pass

    try: out["n_neg"] = sp.get_countNeg()
    except Exception: pass

    return out


# ════════════════════════════════════════════════════════════════════════════
# Per-variant computation (parallelised)
# ════════════════════════════════════════════════════════════════════════════

def _delta_worker(args):
    """Worker function for multiprocessing.Pool. Computes deltas per variant."""
    wt_seq, mut_seq = args
    wt = _compute_features(wt_seq)
    mut = _compute_features(mut_seq)
    return {f: mut[f] - wt[f] if pd.notna(wt[f]) and pd.notna(mut[f]) else np.nan
            for f in PHYSCHEM_FEATURES}


def compute_physchem_deltas(
    df_rg: pd.DataFrame,
    region_by_id: dict,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    For each missense variant, compute delta of each physchem feature
    between WT region sequence and mutated region sequence.

    Returns a dataframe with `region_id`, `group`, and one column per feature:
        delta_ncpr, delta_fcr, delta_kappa, delta_hydropathy,
        delta_aromaticity, delta_fraction_proline, delta_n_pos, delta_n_neg
    plus the original variant identifiers retained from df_rg.
    """
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)

    df = df_rg[
        df_rg["Consequence"].fillna("").str.contains("missense_variant")
    ].copy()

    # Build WT and mutant sequences for each variant
    wt_seqs = []
    mut_seqs = []
    for row in df.itertuples(index=False):
        region = region_by_id.get(row.region_id)
        if region is None or pd.isna(row.protein_position_int) or row.after_aa is None:
            wt_seqs.append(None)
            mut_seqs.append(None)
            continue
        seq_wt = region["prot_seq"]
        pos = int(row.protein_position_int) - int(row.region_start_aa)
        if pos < 0 or pos >= len(seq_wt) or len(row.after_aa) != 1:
            wt_seqs.append(None)
            mut_seqs.append(None)
            continue
        seq_mut = seq_wt[:pos] + row.after_aa + seq_wt[pos + 1:]
        wt_seqs.append(seq_wt)
        mut_seqs.append(seq_mut)

    # Drop rows we can't process
    valid = [i for i, (w, m) in enumerate(zip(wt_seqs, mut_seqs))
             if w is not None and m is not None]
    df_valid = df.iloc[valid].copy()
    pairs = [(wt_seqs[i], mut_seqs[i]) for i in valid]

    # Parallel computation
    print(f"Computing physchem deltas for {len(pairs):,} variants "
          f"using {n_workers} workers...")
    with Pool(n_workers) as pool:
        deltas = list(tqdm(
            pool.imap(_delta_worker, pairs, chunksize=100),
            total=len(pairs),
            desc="Physchem",
        ))

    deltas_df = pd.DataFrame(deltas, index=df_valid.index)
    deltas_df.columns = [f"delta_{c}" for c in deltas_df.columns]

    out = pd.concat([df_valid, deltas_df], axis=1)
    return out


def aggregate_per_region(deltas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-region mean of each delta across missense variants in that region.
    Returns one row per region.
    """
    delta_cols = [f"delta_{f}" for f in PHYSCHEM_FEATURES]
    per_region = (
        deltas_df.groupby(["region_id", "group"])[delta_cols]
                 .mean()
                 .reset_index()
    )
    return per_region


# ════════════════════════════════════════════════════════════════════════════
# Static WT features per region (no variant info needed)
# ════════════════════════════════════════════════════════════════════════════

def compute_wt_physchem(region_by_id: dict) -> pd.DataFrame:
    """
    Per-region WT physchem features (from prot_seq alone).
    """
    rows = []
    for rid, r in region_by_id.items():
        feats = _compute_features(r["prot_seq"])
        feats["region_id"] = rid
        feats["group"] = r["group"]
        rows.append(feats)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Plotting (one figure per metric)
# ════════════════════════════════════════════════════════════════════════════

def plot_delta_feature(
    per_region_df: pd.DataFrame,
    feature: str,
    dataset: str = "gnomad",
    save: bool = True,
) -> tuple[plt.Figure, dict]:
    """
    Per-region mean delta for a single physchem feature. Box plot pos vs neg,
    Mann-Whitney U test.
    """
    col = f"delta_{feature}"
    sub = per_region_df[["region_id", "group", col]].dropna(subset=[col])

    pos_vals = sub.loc[sub["group"] == "pos", col]
    neg_vals = sub.loc[sub["group"] == "neg", col]
    _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    sig = significance_stars(p)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    sns.boxplot(
        data=sub, x="group", y=col, order=["neg", "pos"],
        palette=[GROUP_COLORS["neg"], GROUP_COLORS["pos"]],
        width=0.5, fliersize=0, linewidth=0.6, ax=ax,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 4},
    )
    sns.stripplot(
        data=sub, x="group", y=col, order=["neg", "pos"],
        color="black", size=1.5, alpha=0.4, jitter=0.15, ax=ax,
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    vals = sub[col]
    ymax = vals.abs().quantile(0.98)
    if pd.isna(ymax) or ymax == 0:
        ymax = 0.01
    y_bar = ymax * 1.1
    ax.plot([0, 1], [y_bar, y_bar], color="black", lw=0.6)
    ax.text(0.5, y_bar * 1.05, sig, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(-ymax * 1.3, ymax * 1.3)

    ax.set_title(f"Mean Δ {feature} per region (missense)")
    ax.set_ylabel(f"Δ {feature}")
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
        save_figure(fig, f"physchem_delta_{feature}", dataset=dataset)

    results = {
        "feature": feature,
        "p": float(p), "sig": sig,
        "median_pos": float(pos_vals.median()),
        "median_neg": float(neg_vals.median()),
        "mean_pos": float(pos_vals.mean()),
        "mean_neg": float(neg_vals.mean()),
        "n_pos": int(len(pos_vals)),
        "n_neg": int(len(neg_vals)),
    }

    print(f"\n── Δ {feature} per region ({dataset}) ──")
    print(f"  pos: median = {results['median_pos']:+.4g}, "
          f"mean = {results['mean_pos']:+.4g}, n = {results['n_pos']}")
    print(f"  neg: median = {results['median_neg']:+.4g}, "
          f"mean = {results['mean_neg']:+.4g}, n = {results['n_neg']}")
    print(f"  Mann-Whitney p = {p:.2e} {sig}")

    return fig, results


def plot_all_delta_features(
    per_region_df: pd.DataFrame,
    dataset: str = "gnomad",
    save: bool = True,
) -> dict:
    """Run plot_delta_feature for every feature in PHYSCHEM_FEATURES."""
    all_results = {}
    for feature in PHYSCHEM_FEATURES:
        fig, r = plot_delta_feature(per_region_df, feature, dataset=dataset, save=save)
        plt.show()
        all_results[feature] = r
    return all_results