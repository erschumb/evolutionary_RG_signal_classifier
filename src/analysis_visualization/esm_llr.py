"""
ESM1b LLR analysis of missense variants.

Save as: src/analysis_visualization/esm_llr.py

Compares ESM1b log-likelihood ratio (LLR) scores between pos and neg groups.
LLR = log P(variant_aa | sequence) - log P(WT_aa | sequence), masked at the
position of interest. More negative = more disruptive predicted effect.

Source of precomputed scores:
  Brandes et al. 2023, Nature Genetics
  https://huggingface.co/spaces/ntranoslab/esm_variants
  All ~450M human missense variants pre-scored, keyed by UniProt accession.

Per-protein file format:
  One CSV per UniProt accession with columns: variant_name (e.g. "M1A"),
  pos, wt_aa, mt_aa, esm_score (the LLR).

Design choices:
  - Per-variant LLR is the unit of analysis (not per-region averaging).
    Averaging within a region throws away signal — keep variants as
    independent observations and let the test see them.
  - Mann-Whitney U for pos vs neg distributions (same as Grantham analysis).
  - Stratify by AF (rare vs common) for parallelism with substitution_matrix.
  - Cohen's d effect size alongside p-value — significance with thousands
    of variants is easy; the effect size is what tells you if it's real.
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from src.analysis_visualization.plot_config import (
    GROUP_COLORS, save_figure, significance_stars,
)


# ════════════════════════════════════════════════════════════════════════════
# Loading precomputed ESM1b scores
# ════════════════════════════════════════════════════════════════════════════
# import requests
# from io import StringIO

# _ESM_API_BASE = "https://huggingface.co/spaces/ntranoslab/esm_variants/resolve/main/ALL_hum_isoforms_ESM1b_LLR"

# def load_esm_scores_for_protein_api(
#     uniprot_id: str,
#     cache_dir: str | Path | None = None,
# ) -> pd.DataFrame | None:
#     """
#     Fetch one protein's ESM1b LLR scores from the Hugging Face mirror.
#     Caches to cache_dir/<uniprot_id>.csv if provided.

#     Returns DataFrame with columns: pos, wt_aa, mt_aa, esm_score
#     """
#     # Check cache first
#     if cache_dir is not None:
#         cache_path = Path(cache_dir) / f"{uniprot_id}.csv"
#         if cache_path.exists():
#             return pd.read_csv(cache_path)

#     url = f"{_ESM_API_BASE}/{uniprot_id}_LLR.csv"
#     try:
#         r = requests.get(url, timeout=30)
#         if r.status_code == 404:
#             return None
#         r.raise_for_status()
#     except requests.RequestException:
#         return None

#     # The Brandes catalog format: rows = 20 amino acids, columns = positions
#     # First column is the variant_aa, header row is "M 1", "L 2", etc.
#     raw = pd.read_csv(StringIO(r.text), index_col=0)

#     # Pivot to long format
#     records = []
#     for col in raw.columns:
#         # Column header is e.g. "M 1" → wt_aa="M", pos=1
#         parts = col.strip().split()
#         if len(parts) != 2:
#             continue
#         wt_aa, pos_str = parts
#         try:
#             pos = int(pos_str)
#         except ValueError:
#             continue
#         for mt_aa in raw.index:
#             score = raw.loc[mt_aa, col]
#             if pd.notna(score):
#                 records.append((pos, wt_aa, mt_aa, float(score)))

#     df = pd.DataFrame(records, columns=["pos", "wt_aa", "mt_aa", "esm_score"])

#     if cache_dir is not None:
#         Path(cache_dir).mkdir(parents=True, exist_ok=True)
#         df.to_csv(Path(cache_dir) / f"{uniprot_id}.csv", index=False)

#     return df


# def annotate_variants_with_esm(
#     df: pd.DataFrame,
#     esm_dir: str | Path,
#     uniprot_col: str = "uniprot_id",
#     pos_col: str = "protein_position",
#     before_col: str = "before_aa",
#     after_col: str = "after_aa",
#     consequence_col: str = "Consequence",
#     cache: bool = True,
# ) -> pd.DataFrame:
#     """
#     [Dataset-agnostic]
#     Add an `esm_llr` column to a missense variant dataframe by joining
#     against the precomputed Brandes catalog. Caches per-protein lookups
#     in memory to avoid re-reading CSVs.

#     Returns a copy of df with `esm_llr` added (NaN where unavailable).
#     """
#     df = df.copy()
#     is_missense = df[consequence_col].fillna("").str.contains("missense_variant")
#     df["esm_llr"] = np.nan

#     _per_protein: dict[str, pd.DataFrame | None] = {}
#     n_hits, n_missing_protein, n_missing_variant = 0, 0, 0

#     for uid in df.loc[is_missense, uniprot_col].dropna().unique():
#         if cache and uid in _per_protein:
#             esm = _per_protein[uid]
#         else:
#             esm = load_esm_scores_for_protein_api(uid, esm_dir)
#             if cache:
#                 _per_protein[uid] = esm
#         if esm is None:
#             n_missing_protein += 1
#             continue

#         # Build a (pos, mt_aa) -> score lookup once per protein
#         lookup = esm.set_index(["pos", "mt_aa"])["esm_score"].to_dict()

#         mask = is_missense & (df[uniprot_col] == uid)
#         sub = df.loc[mask, [pos_col, after_col]]
#         # Coerce protein position to int — VEP sometimes gives "12-14" ranges
#         # for in-frame indels, but we already filtered to missense so plain
#         # int conversion is safe; non-coercible rows stay NaN.
#         scores = []
#         for _, r in sub.iterrows():
#             try:
#                 p = int(r[pos_col])
#             except (TypeError, ValueError):
#                 scores.append(np.nan); continue
#             mt = r[after_col]
#             s = lookup.get((p, mt), np.nan)
#             scores.append(s)
#             if pd.isna(s):
#                 n_missing_variant += 1
#             else:
#                 n_hits += 1
#         df.loc[mask, "esm_llr"] = scores

#     n_missense = int(is_missense.sum())
#     print(f"── ESM1b LLR annotation ──")
#     print(f"  Missense variants: {n_missense:,}")
#     print(f"  Annotated: {n_hits:,} ({100*n_hits/max(n_missense,1):.1f}%)")
#     print(f"  Proteins not in catalog: {n_missing_protein}")
#     print(f"  Position/AA not found in catalog: {n_missing_variant:,}")
#     return df


# ════════════════════════════════════════════════════════════════════════════
# Effect size
# ════════════════════════════════════════════════════════════════════════════

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled-SD Cohen's d. Sign convention: positive = x > y."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1)*vx + (ny - 1)*vy) / (nx + ny - 2))
    if pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled


# ════════════════════════════════════════════════════════════════════════════
# Core comparison
# ════════════════════════════════════════════════════════════════════════════

def compare_esm_distributions(
    df: pd.DataFrame,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    score_col: str = "esm_llr",
) -> dict:
    """
    [Dataset-agnostic]
    Mann-Whitney U test on ESM LLR distributions between groups.
    """
    pos_vals = df.loc[df[group_col] == pos_label, score_col].dropna().values
    neg_vals = df.loc[df[group_col] == neg_label, score_col].dropna().values

    if len(pos_vals) < 5 or len(neg_vals) < 5:
        return {
            "pos_vals": pos_vals, "neg_vals": neg_vals,
            "n_pos": len(pos_vals), "n_neg": len(neg_vals),
            "u_stat": np.nan, "p_value": np.nan, "cohens_d": np.nan,
            "median_pos": np.nan, "median_neg": np.nan,
            "mean_pos": np.nan, "mean_neg": np.nan,
        }

    u, p = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    return {
        "pos_vals": pos_vals,
        "neg_vals": neg_vals,
        "n_pos": len(pos_vals),
        "n_neg": len(neg_vals),
        "u_stat": u,
        "p_value": p,
        "cohens_d": cohens_d(pos_vals, neg_vals),
        "median_pos": float(np.median(pos_vals)),
        "median_neg": float(np.median(neg_vals)),
        "mean_pos": float(np.mean(pos_vals)),
        "mean_neg": float(np.mean(neg_vals)),
    }


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

# def plot_esm_distributions(
#     result: dict,
#     dataset: str = "gnomad",
#     save: bool = True,
#     title_suffix: str = "",
# ) -> plt.Figure:
#     """
#     Two-panel figure:
#       1. Violin + box of ESM1b LLR distributions
#       2. ECDF of LLR scores
#     """
#     pos_color = GROUP_COLORS.get("pos", "#4daf4a")
#     neg_color = GROUP_COLORS.get("neg", "#e41a1c")

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # ── Panel 1: violin + boxplot ──────────────────────────────────────────
#     plot_df = pd.DataFrame({
#         "esm_llr": np.concatenate([result["pos_vals"], result["neg_vals"]]),
#         "group": (["pos"] * result["n_pos"]) + (["neg"] * result["n_neg"]),
#     })
#     sns.violinplot(
#         data=plot_df, x="group", y="esm_llr", ax=axes[0],
#         order=["pos", "neg"],
#         palette={"pos": pos_color, "neg": neg_color},
#         inner="box", cut=0,
#     )
#     axes[0].set_ylabel("ESM1b LLR  (more negative = more disruptive)")
#     axes[0].set_xlabel("")
#     axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

#     # Annotate with stats
#     p, d = result["p_value"], result["cohens_d"]
#     stars = significance_stars(p) if pd.notna(p) else "n/a"
#     label = (
#         f"n_pos = {result['n_pos']:,}\n"
#         f"n_neg = {result['n_neg']:,}\n"
#         f"U = {result['u_stat']:.0f}\n"
#         f"p = {p:.2e} {stars}\n"
#         f"Cohen's d = {d:.3f}"
#     )
#     axes[0].text(
#         0.98, 0.02, label, transform=axes[0].transAxes,
#         ha="right", va="bottom", fontsize=9,
#         bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.4"),
#     )

#     # ── Panel 2: ECDF ──────────────────────────────────────────────────────
#     for vals, color, label in [
#         (result["pos_vals"], pos_color, "pos"),
#         (result["neg_vals"], neg_color, "neg"),
#     ]:
#         x = np.sort(vals)
#         y = np.arange(1, len(x) + 1) / len(x)
#         axes[1].plot(x, y, color=color, label=label, linewidth=1.5)
#     axes[1].set_xlabel("ESM1b LLR")
#     axes[1].set_ylabel("Cumulative fraction")
#     axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
#     axes[1].legend(loc="lower right")
#     axes[1].grid(alpha=0.2)

#     title = f"ESM1b LLR  pos vs neg ({dataset})"
#     if title_suffix:
#         title += f" — {title_suffix}"
#     fig.suptitle(title, fontsize=13, y=1.02)

#     plt.tight_layout()
#     if save:
#         save_figure(fig, "esm_llr_comparison", dataset=dataset)
#     return fig


# ════════════════════════════════════════════════════════════════════════════
# One-call wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_esm_analysis(
    df: pd.DataFrame,
    zip_path: str | Path | None = None,
    score_col: str = "esm_llr",
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    dataset: str = "gnomad",
    save: bool = True,
    annotate: bool = True,
    **annotate_kwargs,
) -> dict:
    if annotate:
        if zip_path is None:
            raise ValueError("zip_path required when annotate=True")
        df = annotate_variants_with_esm(df, zip_path, **annotate_kwargs)
    elif score_col not in df.columns:
        raise ValueError(f"`{score_col}` not in df and annotate=False.")

    result = compare_esm_distributions(
        df, group_col=group_col, pos_label=pos_label, neg_label=neg_label,
        score_col=score_col,
    )

    print(f"\n── ESM1b LLR comparison ({dataset}) ──")
    print(f"  pos: n={result['n_pos']:,}  median={result['median_pos']:.3f}  mean={result['mean_pos']:.3f}")
    print(f"  neg: n={result['n_neg']:,}  median={result['median_neg']:.3f}  mean={result['mean_neg']:.3f}")
    print(f"  Mann-Whitney U={result['u_stat']:.0f}, p={result['p_value']:.4g}")
    print(f"  Cohen's d = {result['cohens_d']:.3f}")

    plot_esm_distributions(result, dataset=dataset, save=save)
    return result


# ════════════════════════════════════════════════════════════════════════════
# AF-stratified comparison (parallel to substitution_matrix's AF analysis)
# ════════════════════════════════════════════════════════════════════════════

def run_esm_analysis_af_stratified(
    df: pd.DataFrame,
    af_column: str = "AF_joint",
    af_rare_max: float = 1e-4,
    af_common_min: float = 1e-3,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    score_col: str = "esm_llr",
    dataset: str = "gnomad",
    save: bool = True,
) -> dict:
    """
    [gnomAD-specific]
    Compare ESM LLR distributions between rare and common variants,
    parallel to plot_af_comparison_matrices in substitution_matrix.py.

    Rare variants reflect purifying selection cleanly; if pos regions are
    under stronger constraint, the rare-variant LLR shift between pos and
    neg should be larger than the common-variant shift.
    """
    df = df[df[score_col].notna() & df[af_column].notna()].copy()

    rare = df[df[af_column] < af_rare_max]
    common = df[df[af_column] >= af_common_min]

    res_rare = compare_esm_distributions(
        rare, group_col=group_col, pos_label=pos_label,
        neg_label=neg_label, score_col=score_col,
    )
    res_common = compare_esm_distributions(
        common, group_col=group_col, pos_label=pos_label,
        neg_label=neg_label, score_col=score_col,
    )

    # ── Plot side-by-side ──────────────────────────────────────────────────
    pos_color = GROUP_COLORS.get("pos", "#4daf4a")
    neg_color = GROUP_COLORS.get("neg", "#e41a1c")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, res, label in [
        (axes[0], res_rare, f"Rare (AF < {af_rare_max:.0e})"),
        (axes[1], res_common, f"Common (AF ≥ {af_common_min:.0e})"),
    ]:
        plot_df = pd.DataFrame({
            "esm_llr": np.concatenate([res["pos_vals"], res["neg_vals"]]),
            "group": (["pos"]*res["n_pos"]) + (["neg"]*res["n_neg"]),
        })
        sns.violinplot(
            data=plot_df, x="group", y="esm_llr", ax=ax,
            order=["pos", "neg"],
            palette={"pos": pos_color, "neg": neg_color},
            inner="box", cut=0,
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(
            f"{label}\n"
            f"n_pos={res['n_pos']:,}, n_neg={res['n_neg']:,}\n"
            f"p={res['p_value']:.2e} {significance_stars(res['p_value'])}, "
            f"d={res['cohens_d']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("")

    axes[0].set_ylabel("ESM1b LLR")
    axes[1].set_ylabel("")
    fig.suptitle(
        f"ESM1b LLR  pos vs neg, stratified by allele frequency ({dataset})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "esm_llr_af_stratified", dataset=dataset)

    print(f"\n── ESM1b LLR  AF-stratified comparison ({dataset}) ──")
    for label, res in [("Rare", res_rare), ("Common", res_common)]:
        print(f"  {label}: n_pos={res['n_pos']:,}, n_neg={res['n_neg']:,}, "
              f"p={res['p_value']:.4g}, d={res['cohens_d']:.3f}")

    return {"rare": res_rare, "common": res_common}




from zipfile import ZipFile
from urllib.request import urlretrieve
from io import TextIOWrapper

_LLR_ZIP_URL = (
    "https://huggingface.co/spaces/ntranoslab/esm_variants/"
    "resolve/main/ALL_hum_isoforms_ESM1b_LLR.zip"
)

def download_esm_llr_zip(dest: str | Path, force: bool = False) -> Path:
    """Download the Brandes et al. ESM1b LLR zip if not already present (~1.34 GB)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"Already downloaded: {dest} ({dest.stat().st_size/1e9:.2f} GB)")
        return dest
    print(f"Downloading {_LLR_ZIP_URL}\n  → {dest}")
    urlretrieve(_LLR_ZIP_URL, dest)
    print(f"Done: {dest.stat().st_size/1e9:.2f} GB")
    return dest


def load_esm_scores_for_protein(
    uniprot_id: str,
    zip_path: str | Path,
) -> pd.DataFrame | None:
    """
    Load ESM1b LLR scores for one UniProt accession from the Brandes catalog zip.
    Returns long-format DataFrame[pos, wt_aa, mt_aa, esm_score], or None if absent.
    """
    zip_path = Path(zip_path)
    with ZipFile(zip_path) as zf:
        all_names = zf.namelist()
        # Find directory prefix (the one entry ending in "/")
        prefix = next((n for n in all_names if n.endswith("/")), "")
        candidate = f"{prefix}{uniprot_id}_LLR.csv"
        if candidate not in all_names:
            return None
        with zf.open(candidate) as f:
            wide = pd.read_csv(f, index_col=0)

    # Wide → long. Columns are e.g. "M 1", "E 2"; index rows are mutant AAs.
    records = []
    for col in wide.columns:
        parts = col.strip().split()
        if len(parts) != 2:
            continue
        wt_aa, pos_str = parts
        try:
            pos = int(pos_str)
        except ValueError:
            continue
        for mt_aa in wide.index:
            score = wide.loc[mt_aa, col]
            if pd.notna(score):
                records.append((pos, wt_aa, mt_aa, float(score)))
    return pd.DataFrame(records, columns=["pos", "wt_aa", "mt_aa", "esm_score"])


# def annotate_variants_with_esm(
#     df: pd.DataFrame,
#     zip_path: str | Path,
#     uniprot_col: str = "uniprot_accession",
#     pos_col: str = "Protein_position",
#     after_col: str = "after_aa",
#     consequence_col: str = "Consequence",
# ) -> pd.DataFrame:
#     """[Dataset-agnostic] Join ESM1b LLR scores onto a missense variant df."""
#     df = df.copy()
#     is_missense = df[consequence_col].fillna("").str.contains("missense_variant")
#     df["esm_llr"] = np.nan

#     n_hits, n_missing_protein, n_missing_variant = 0, 0, 0
#     n_proteins = df.loc[is_missense, uniprot_col].nunique()
#     print(f"Annotating against {n_proteins} unique proteins...")

#     # Open the zip once and reuse — much faster than re-opening per protein
#     with ZipFile(zip_path) as zf:
#         all_names = zf.namelist()
#         names = set(all_names)
#         # Find directory prefix (the one entry ending in "/")
#         prefix = next((n for n in all_names if n.endswith("/")), "")
#         print(f"  zip prefix: {prefix!r}")
#         names = set(all_names)
#         for i, uid in enumerate(sorted(df.loc[is_missense, uniprot_col].dropna().unique())):
#             target = f"{prefix}{uid}_LLR.csv"
#             if target not in names:
#                 n_missing_protein += 1
#                 continue

#             with zf.open(target) as f:
#                 wide = pd.read_csv(f, index_col=0)

#             # Build (pos, mt_aa) → score lookup
#             lookup = {}
#             for col in wide.columns:
#                 parts = col.strip().split()
#                 if len(parts) != 2:
#                     continue
#                 _, pos_str = parts
#                 try:
#                     pos = int(pos_str)
#                 except ValueError:
#                     continue
#                 for mt_aa in wide.index:
#                     s = wide.loc[mt_aa, col]
#                     if pd.notna(s):
#                         lookup[(pos, mt_aa)] = float(s)

#             mask = is_missense & (df[uniprot_col] == uid)
#             for idx in df.index[mask]:
#                 try:
#                     p = int(df.at[idx, pos_col])
#                 except (TypeError, ValueError):
#                     continue
#                 mt = df.at[idx, after_col]
#                 s = lookup.get((p, mt))
#                 if s is not None:
#                     df.at[idx, "esm_llr"] = s
#                     n_hits += 1
#                 else:
#                     n_missing_variant += 1

#             if (i + 1) % 50 == 0:
#                 print(f"  {i+1}/{n_proteins} proteins")

#     n_missense = int(is_missense.sum())
#     print(f"\n── ESM1b LLR annotation ──")
#     print(f"  Missense variants: {n_missense:,}")
#     print(f"  Annotated: {n_hits:,} ({100*n_hits/max(n_missense,1):.1f}%)")
#     print(f"  Proteins not in catalog: {n_missing_protein}")
#     print(f"  Position/AA not found: {n_missing_variant:,}")
#     return df

def annotate_variants_with_esm(
    df: pd.DataFrame,
    zip_path: str | Path,
    uniprot_col: str = "uniprot_accession",
    pos_col: str = "Protein_position",
    before_col: str = "before_aa",
    after_col: str = "after_aa",
    consequence_col: str = "Consequence",
) -> pd.DataFrame:
    """[Dataset-agnostic] Join ESM1b LLR scores onto a missense variant df.

    Verifies that df's WT residue (`before_col`) matches the ESM catalog's
    WT identity at each position; mismatches indicate position-numbering
    drift (e.g. MANE vs isoform) and are counted and skipped rather than
    assigned a wrong-residue score.
    """
    df = df.copy()
    is_missense = df[consequence_col].fillna("").str.contains("missense_variant")
    df["esm_llr"] = np.nan

    n_hits, n_missing_protein, n_missing_variant, n_wt_mismatch = 0, 0, 0, 0
    n_proteins = df.loc[is_missense, uniprot_col].nunique()
    print(f"Annotating against {n_proteins} unique proteins...")

    with ZipFile(zip_path) as zf:
        all_names = zf.namelist()
        prefix = next((n for n in all_names if n.endswith("/")), "")
        print(f"  zip prefix: {prefix!r}")
        names = set(all_names)

        for i, uid in enumerate(sorted(df.loc[is_missense, uniprot_col].dropna().unique())):
            target = f"{prefix}{uid}_LLR.csv"
            if target not in names:
                n_missing_protein += 1
                continue

            with zf.open(target) as f:
                wide = pd.read_csv(f, index_col=0)

            # Build (pos, mt_aa) → score lookup AND pos → wt_aa map
            lookup = {}
            wt_by_pos = {}
            for col in wide.columns:
                parts = col.strip().split()
                if len(parts) != 2:
                    continue
                wt_aa, pos_str = parts
                try:
                    pos = int(pos_str)
                except ValueError:
                    continue
                wt_by_pos[pos] = wt_aa
                for mt_aa in wide.index:
                    s = wide.loc[mt_aa, col]
                    if pd.notna(s):
                        lookup[(pos, mt_aa)] = float(s)

            mask = is_missense & (df[uniprot_col] == uid)
            for idx in df.index[mask]:
                try:
                    p = int(df.at[idx, pos_col])
                except (TypeError, ValueError):
                    continue

                # WT cross-check: catch position-numbering drift
                expected_wt = wt_by_pos.get(p)
                bef = df.at[idx, before_col]
                if expected_wt is not None and pd.notna(bef) and bef != expected_wt:
                    n_wt_mismatch += 1
                    continue

                mt = df.at[idx, after_col]
                s = lookup.get((p, mt))
                if s is not None:
                    df.at[idx, "esm_llr"] = s
                    n_hits += 1
                else:
                    n_missing_variant += 1

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_proteins} proteins")

    n_missense = int(is_missense.sum())
    print(f"\n── ESM1b LLR annotation ──")
    print(f"  Missense variants: {n_missense:,}")
    print(f"  Annotated: {n_hits:,} ({100*n_hits/max(n_missense,1):.1f}%)")
    print(f"  Proteins not in catalog: {n_missing_protein}")
    print(f"  Position/AA not found: {n_missing_variant:,}")
    print(f"  WT mismatches (numbering drift): {n_wt_mismatch:,}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# Per-substitution LLR difference matrix
# ════════════════════════════════════════════════════════════════════════════

from src.analysis_visualization.substitution_matrix_analysis import (
    ORDERED_AA, AA_GROUPS, GROUP_ORDER, GROUP_SLICES_COL, GROUP_SLICES_ROW,
    _make_diverging_cmap, _add_group_boxes,
)
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def compute_llr_substitution_matrix(
    df: pd.DataFrame,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    before_col: str = "before_aa",
    after_col: str = "after_aa",
    score_col: str = "esm_llr",
    consequence_col: str = "Consequence",
    min_obs: int = 5,
) -> dict:
    """
    [Dataset-agnostic]
    For each (aa_from, aa_to) pair, compare the distribution of ESM1b LLR
    scores between pos and neg groups via Mann-Whitney U, then compute the
    mean LLR difference (neg - pos) as the effect size.

    A positive diff means neg tolerates this substitution more (less negative
    LLR) — i.e. less evolutionary constraint in neg regions for that change.

    Returns dict with:
        mean_pos, mean_neg   — 20×20 mean LLR matrices
        diff                 — mean_neg - mean_pos (20×20)
        pval, fdr            — per-cell Mann-Whitney p and BH-corrected FDR
        n_pos, n_neg         — observation count matrices
        n_tested             — number of cells that passed min_obs filter
    """
    missense = df[
        df[consequence_col].fillna("").str.contains("missense_variant") &
        df[before_col].notna() & df[after_col].notna() &
        df[score_col].notna()
    ].copy()

    # Pre-split for speed
    pos_df = missense[missense[group_col] == pos_label]
    neg_df = missense[missense[group_col] == neg_label]

    # Build per-(aa_from, aa_to) score lists
    def _score_lookup(sub):
        lookup = {}
        for (aa_f, aa_t), grp in sub.groupby([before_col, after_col]):
            if aa_f in ORDERED_AA and aa_t in ORDERED_AA and aa_f != aa_t:
                lookup[(aa_f, aa_t)] = grp[score_col].values
        return lookup

    pos_lookup = _score_lookup(pos_df)
    neg_lookup = _score_lookup(neg_df)

    # Result matrices
    mean_pos = pd.DataFrame(np.nan, index=ORDERED_AA, columns=ORDERED_AA)
    mean_neg = pd.DataFrame(np.nan, index=ORDERED_AA, columns=ORDERED_AA)
    diff     = pd.DataFrame(np.nan, index=ORDERED_AA, columns=ORDERED_AA)
    pval     = pd.DataFrame(np.nan, index=ORDERED_AA, columns=ORDERED_AA)
    n_pos_m  = pd.DataFrame(0,      index=ORDERED_AA, columns=ORDERED_AA)
    n_neg_m  = pd.DataFrame(0,      index=ORDERED_AA, columns=ORDERED_AA)

    tested_cells = []

    for aa_from in ORDERED_AA:
        for aa_to in ORDERED_AA:
            if aa_from == aa_to:
                continue
            pv = pos_lookup.get((aa_from, aa_to), np.array([]))
            nv = neg_lookup.get((aa_from, aa_to), np.array([]))

            n_pos_m.loc[aa_from, aa_to] = len(pv)
            n_neg_m.loc[aa_from, aa_to] = len(nv)

            if len(pv) < min_obs or len(nv) < min_obs:
                continue

            mean_pos.loc[aa_from, aa_to] = pv.mean()
            mean_neg.loc[aa_from, aa_to] = nv.mean()
            diff.loc[aa_from, aa_to]     = nv.mean() - pv.mean()

            _, p = mannwhitneyu(pv, nv, alternative="two-sided")
            pval.loc[aa_from, aa_to] = p
            tested_cells.append((aa_from, aa_to, p))

    # BH FDR correction
    fdr = pd.DataFrame(np.nan, index=ORDERED_AA, columns=ORDERED_AA)
    if tested_cells:
        raw_ps = [c[2] for c in tested_cells]
        _, corrected, _, _ = multipletests(raw_ps, method="fdr_bh")
        for (aa_from, aa_to, _), p_fdr in zip(tested_cells, corrected):
            fdr.loc[aa_from, aa_to] = p_fdr

    return {
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "diff": diff,
        "pval": pval,
        "fdr": fdr,
        "n_pos": n_pos_m,
        "n_neg": n_neg_m,
        "n_tested": len(tested_cells),
    }

def plot_esm_distributions(
    result: dict,
    dataset: str = "gnomad",
    save: bool = True,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Two-panel figure:
      1. Raincloud (half-violin left + box center) of ESM1b LLR distributions
      2. ECDF
    """
    from src.analysis_visualization.plot_config import (
        FIGSIZE_SINGLE, GROUP_COLORS, save_figure, significance_stars,
    )
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe

    pos_color = GROUP_COLORS["pos"]
    neg_color = GROUP_COLORS["neg"]

    p   = result["p_value"]
    d   = result["cohens_d"]
    sig = significance_stars(p) if pd.notna(p) else "n/a"

    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIGSIZE_SINGLE[0] * 2, FIGSIZE_SINGLE[1]),
    )

    # ── Panel 1: raincloud ────────────────────────────────────────────────
    ax = axes[0]

    groups   = [("neg", result["neg_vals"], neg_color, 0),
                ("pos", result["pos_vals"], pos_color, 1)]
    x_positions = [0, 1]

    for label, vals, color, xi in groups:
        # ── Half violin (left side only) ──────────────────────────────
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals, bw_method=0.15)
        y_range = np.linspace(vals.min(), vals.max(), 300)
        kde_vals = kde(y_range)
        # Normalize to half-width of 0.35
        kde_vals = kde_vals / kde_vals.max() * 0.35

        # Fill left of x position
        ax.fill_betweenx(
            y_range,
            xi - kde_vals,   # left edge
            xi,              # center
            alpha=0.55,
            color=color,
            linewidth=0,
        )
        ax.plot(
            xi - kde_vals,
            y_range,
            color=color,
            linewidth=0.8,
            alpha=0.8,
        )

        # ── Box plot (right side, narrow) ─────────────────────────────
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        whisker_lo = max(vals.min(), q1 - 1.5 * iqr)
        whisker_hi = min(vals.max(), q3 + 1.5 * iqr)
        mean_val   = vals.mean()

        box_x     = xi + 0.04   # slight right offset from center
        box_w     = 0.12        # half-width of box

        # IQR box
        ax.add_patch(plt.Rectangle(
            (box_x - box_w, q1), box_w * 2, iqr,
            facecolor=color, alpha=0.85,
            edgecolor="white", linewidth=0.8,
            zorder=3,
        ))
        # Median line
        ax.plot(
            [box_x - box_w, box_x + box_w],
            [median, median],
            color="white", linewidth=1.8, zorder=4,
        )
        # Whiskers
        ax.plot(
            [box_x, box_x], [whisker_lo, q1],
            color=color, linewidth=0.9, zorder=3,
        )
        ax.plot(
            [box_x, box_x], [q3, whisker_hi],
            color=color, linewidth=0.9, zorder=3,
        )
        # Whisker caps
        for wy in [whisker_lo, whisker_hi]:
            ax.plot(
                [box_x - box_w * 0.6, box_x + box_w * 0.6],
                [wy, wy],
                color=color, linewidth=0.9, zorder=3,
            )
        # Mean diamond
        ax.plot(
            box_x, mean_val, marker="D",
            markerfacecolor="white", markeredgecolor="black",
            markersize=4, zorder=5,
        )

    # Reference line at 0
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    # Significance bracket
    all_vals = np.concatenate([result["neg_vals"], result["pos_vals"]])
    ymax = float(np.percentile(np.abs(all_vals), 99))
    if ymax == 0:
        ymax = 10.0
    y_bar = ymax * 1.08
    ax.plot([0, 1], [y_bar, y_bar], color="black", lw=0.8)
    ax.text(0.5, y_bar * 1.01, sig, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(-(ymax * 1.25), ymax * 1.3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["neg", "pos"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("ESM1b LLR  (more negative = more disruptive)")
    ax.set_xlabel("")
    ax.set_title("ESM1b LLR  pos vs neg")

    stats_text = (
        f"p = {p:.1e} {sig}\n"
        f"Cohen's d = {d:.3f}\n"
        f"n_pos = {result['n_pos']:,}\n"
        f"n_neg = {result['n_neg']:,}"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes, fontsize=6.5,
        va="bottom", ha="left",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2),
    )

    sns.despine(ax=ax)

    # ── Panel 2: ECDF (unchanged) ─────────────────────────────────────────
    ax2 = axes[1]
    for vals, color, label in [
        (result["neg_vals"], neg_color, "neg"),
        (result["pos_vals"], pos_color, "pos"),
    ]:
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        ax2.plot(x, y, color=color, label=label, linewidth=1.5)

    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax2.set_xlabel("ESM1b LLR")
    ax2.set_ylabel("Cumulative fraction")
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.2)
    ax2.set_title("ECDF — ESM1b LLR")
    sns.despine(ax=ax2)

    title = f"ESM1b LLR  pos vs neg ({dataset})"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    if save:
        save_figure(fig, "esm_llr_comparison", dataset=dataset)

    return fig

def run_llr_substitution_analysis(
    df: pd.DataFrame,
    group_col: str = "group",
    pos_label: str = "pos",
    neg_label: str = "neg",
    before_col: str = "before_aa",
    after_col: str = "after_aa",
    score_col: str = "esm_llr",
    min_obs: int = 5,
    dataset: str = "gnomad",
    save: bool = True,
    **plot_kwargs,
) -> dict:
    """[Dataset-agnostic] End-to-end LLR substitution matrix."""
    result = compute_llr_substitution_matrix(
        df, group_col=group_col, pos_label=pos_label, neg_label=neg_label,
        before_col=before_col, after_col=after_col, score_col=score_col,
        min_obs=min_obs,
    )
    result["min_obs"] = min_obs
    plot_llr_substitution_matrix(result, dataset=dataset, save=save, **plot_kwargs)
    return result