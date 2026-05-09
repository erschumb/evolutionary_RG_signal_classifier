"""
Save as: src/analysis_visualization/homolog_rg_metrics.py

RG-motif metrics from homolog alignments, mirroring the gnomAD RG analyses.

Analyses:
  1. Change events per motif (no_change / loss / gain / movement) — 
     each (motif, homolog) treated as one "event observation", fractions
     aggregated per motif.
  2. RG count distribution across homologs per motif (stability of count).
  3. RG density delta between query and each homolog.
  4. R-fraction stability — does R vs G ratio persist across homologs?
  5. Per-position RG retention — at query RG positions, what fraction of
     homologs retain R or G at that column?

Approach A (per-homolog event): each homolog hit is one observation,
analyses are per-motif aggregates. Reformats homolog data to the gnomAD
variant schema where feasible to reuse `rg_analysis` code.

[Homolog-specific; gap handling: gap in homolog at RG position = RG loss.]
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.analysis_visualization.plot_config import (
    GROUP_COLORS, save_figure, significance_stars,
)

RG = {"R", "G"}


# ════════════════════════════════════════════════════════════════════════════
# Per-hit RG metrics: compute for each (motif, homolog) pair
# ════════════════════════════════════════════════════════════════════════════

def _classify_event_per_hit(qseq: str, hseq: str) -> str:
    """
    Classify the overall RG-change event for a single (motif, homolog) pair.
    Returns: "no_change" | "loss" | "gain" | "movement".
    Gaps in hseq at RG positions count as losses. Query R/G must be retained
    as R/G to avoid counting as loss. Query non-R/G becoming R/G in hseq
    (any position, gap-free) counts as gain.
    """
    if len(qseq) != len(hseq):
        return "no_change"  # sanity — shouldn't happen with motif-level regions

    loss = False
    gain = False
    for qc, hc in zip(qseq, hseq):
        if qc in RG:
            if hc not in RG:  # gap or other AA both count as loss
                loss = True
        else:
            if hc in RG and hc != "-":
                gain = True

    if loss and gain:
        return "movement"
    if loss:
        return "loss"
    if gain:
        return "gain"
    return "no_change"


def _rg_count(seq: str) -> int:
    """Count R + G in seq (excluding gaps)."""
    return sum(1 for c in seq if c in RG)


def _r_fraction(seq: str) -> float:
    """Fraction of R among R+G positions. NaN if no R/G present."""
    r = sum(1 for c in seq if c == "R")
    g = sum(1 for c in seq if c == "G")
    total = r + g
    return r / total if total > 0 else np.nan


def _rg_density(seq: str) -> float:
    """Fraction of non-gap positions that are R or G."""
    non_gap = [c for c in seq if c != "-"]
    if not non_gap:
        return np.nan
    rg = sum(1 for c in non_gap if c in RG)
    return rg / len(non_gap)


def compute_per_hit_rg_metrics(
    df: pd.DataFrame,
    qseq_col: str = "qseq_rg_region",
    hseq_col: str = "hseq_rg_region",
    motif_col: str = "motif_uid",
    group_col: str = "group",
) -> pd.DataFrame:
    """
    [Homolog-specific]
    Per (motif, homolog) metrics. Returns one row per hit.

    Columns:
      motif_uid, group,
      event_type (no_change/loss/gain/movement),
      qseq_rg_count, hseq_rg_count, delta_rg_count,
      qseq_r_fraction, hseq_r_fraction, delta_r_fraction,
      qseq_rg_density, hseq_rg_density, delta_rg_density.
    """
    df = df[df[qseq_col].notna() & df[hseq_col].notna()].copy()
    # Ensure motif_uid exists
    if motif_col not in df.columns:
        raise KeyError(f"{motif_col} missing; run hr.add_motif_uid first.")

    records = []
    for (motif_uid, group), sub in df.groupby([motif_col, group_col]):
        for q, h in zip(sub[qseq_col].values, sub[hseq_col].values):
            if len(q) != len(h):
                continue
            event = _classify_event_per_hit(q, h)
            q_count = _rg_count(q)
            h_count = _rg_count(h)
            q_r_frac = _r_fraction(q)
            h_r_frac = _r_fraction(h)
            q_density = _rg_density(q)
            h_density = _rg_density(h)

            records.append({
                "motif_uid": motif_uid,
                "group": group,
                "event_type": event,
                "qseq_rg_count": q_count,
                "hseq_rg_count": h_count,
                "delta_rg_count": h_count - q_count,
                "qseq_r_fraction": q_r_frac,
                "hseq_r_fraction": h_r_frac,
                "delta_r_fraction": (h_r_frac - q_r_frac)
                                     if pd.notna(h_r_frac) and pd.notna(q_r_frac) else np.nan,
                "qseq_rg_density": q_density,
                "hseq_rg_density": h_density,
                "delta_rg_density": (h_density - q_density)
                                      if pd.notna(h_density) and pd.notna(q_density) else np.nan,
            })
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# Aggregate per-motif from per-hit rows
# ════════════════════════════════════════════════════════════════════════════

RG_EVENT_TYPES = ["no_change", "loss", "gain", "movement"]


def aggregate_per_motif_rg_metrics(
    per_hit_df: pd.DataFrame,
    min_hits: int = 5,
) -> pd.DataFrame:
    """
    [Homolog-specific]
    Per-motif aggregates derived from per-hit observations.

    Columns:
      motif_uid, group, n_hits,
      event_fraction_<type>              — fraction of hits with each event
      mean_delta_rg_count / std          — RG count change statistics
      mean_delta_r_fraction / std        — R-fraction change statistics
      mean_delta_rg_density              — density change
      fraction_hits_rg_count_preserved   — |delta_rg_count| == 0 fraction
      fraction_hits_r_fraction_preserved — |delta_r_fraction| small fraction
    """
    rows = []
    for motif_uid, sub in per_hit_df.groupby("motif_uid"):
        if len(sub) < min_hits:
            continue
        group = sub["group"].iloc[0]

        event_fractions = {
            f"event_fraction_{etype}": float((sub["event_type"] == etype).mean())
            for etype in RG_EVENT_TYPES
        }
        rows.append({
            "motif_uid": motif_uid,
            "group": group,
            "n_hits": int(len(sub)),
            **event_fractions,
            "mean_delta_rg_count":     float(sub["delta_rg_count"].mean()),
            "std_delta_rg_count":      float(sub["delta_rg_count"].std()),
            "mean_delta_r_fraction":   float(sub["delta_r_fraction"].mean()),
            "std_delta_r_fraction":    float(sub["delta_r_fraction"].std()),
            "mean_delta_rg_density":   float(sub["delta_rg_density"].mean()),
            "fraction_hits_rg_count_preserved":
                float((sub["delta_rg_count"] == 0).mean()),
            "fraction_hits_r_fraction_preserved":
                float((sub["delta_r_fraction"].abs() < 0.05).mean()),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Per-position RG retention
# ════════════════════════════════════════════════════════════════════════════

def compute_per_position_rg_retention(
    df: pd.DataFrame,
    qseq_col: str = "qseq_rg_region",
    hseq_col: str = "hseq_rg_region",
    motif_col: str = "motif_uid",
    group_col: str = "group",
    min_hits: int = 5,
) -> pd.DataFrame:
    """
    [Homolog-specific]
    For each motif, at each query position that is R or G, compute the
    fraction of homolog hits that retain R or G (any of R/G) at that
    alignment column. Returns per-motif aggregates.

    Columns:
      motif_uid, group, n_rg_positions, n_hits,
      mean_rg_retention         — mean across R/G positions
      min_rg_retention          — worst-conserved R/G position
      r_retention, g_retention  — separate for R and G
    """
    rows = []
    for (motif_uid, group), sub in df.groupby([motif_col, group_col]):
        sub = sub[sub[qseq_col].notna() & sub[hseq_col].notna()]
        if len(sub) < min_hits:
            continue
        qseqs = sub[qseq_col].values
        hseqs = sub[hseq_col].values
        qseq = qseqs[0]  # canonical; should be identical across hits
        L = len(qseq)

        # For each query position, collect the column across homologs
        rg_positions = [i for i, c in enumerate(qseq) if c in RG]
        if not rg_positions:
            continue

        retentions_rg_all = []
        retentions_r = []
        retentions_g = []

        valid_hits = [h for h in hseqs if len(h) == L]
        n_valid = len(valid_hits)
        if n_valid < min_hits:
            continue

        for i in rg_positions:
            qc = qseq[i]
            column = [h[i] for h in valid_hits]
            retention = sum(1 for c in column if c in RG) / n_valid
            retentions_rg_all.append(retention)
            if qc == "R":
                retentions_r.append(retention)
            else:
                retentions_g.append(retention)

        rows.append({
            "motif_uid": motif_uid,
            "group": group,
            "n_rg_positions": len(rg_positions),
            "n_hits": n_valid,
            "mean_rg_retention": float(np.mean(retentions_rg_all)),
            "min_rg_retention": float(np.min(retentions_rg_all)),
            "r_retention": float(np.mean(retentions_r)) if retentions_r else np.nan,
            "g_retention": float(np.mean(retentions_g)) if retentions_g else np.nan,
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

def _mwu_box_panel(ax, pos_vals, neg_vals, ylabel, title):
    """Violin + box + strip with Mann-Whitney p shown."""
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
        patch.set_facecolor(color); patch.set_alpha(0.9); patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black"); median.set_linewidth(1.2)

    for x_pos, vals, color in zip([0, 1], data, colors):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.full(len(vals), x_pos) + jitter, vals,
            color=color, alpha=0.4, s=5, edgecolor="none", zorder=2,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["neg", "pos"])
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if len(pos_vals) < 5 or len(neg_vals) < 5:
        ax.text(0.5, 0.96, "n too low", transform=ax.transAxes,
                ha="center", va="top", fontsize=7)
        return {"u": np.nan, "p": np.nan, "sig": "n.s."}

    u, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    sig = significance_stars(p)
    ax.text(
        0.5, 0.96, f"p = {p:.2e} {sig}",
        transform=ax.transAxes, ha="center", va="top", fontsize=7,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
    )
    return {"u": float(u), "p": float(p), "sig": sig}


def plot_rg_change_events_homologs(
    per_motif_df: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    Four panels: fraction of hits per event type (no_change / loss / gain / movement),
    per motif, pos vs neg.
    """
    pos = per_motif_df[per_motif_df["group"] == "pos"]
    neg = per_motif_df[per_motif_df["group"] == "neg"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))

    results = {}
    for ax, etype in zip(axes, RG_EVENT_TYPES):
        col = f"event_fraction_{etype}"
        r = _mwu_box_panel(
            ax,
            pos[col], neg[col],
            ylabel=f"Fraction of homolog hits",
            title=f"Event: {etype}",
        )
        results[etype] = r

    fig.suptitle(f"RG-motif change events per motif ({dataset})",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    if save:
        save_figure(fig, "homolog_rg_change_events", dataset=dataset)

    print(f"\n── RG change event fractions ({dataset}) ──")
    for etype in RG_EVENT_TYPES:
        col = f"event_fraction_{etype}"
        p_med = pos[col].median()
        n_med = neg[col].median()
        r = results[etype]
        print(f"  {etype:12s}: pos median = {p_med:.3f}, neg median = {n_med:.3f}, "
              f"p = {r['p']:.2e} {r['sig']}")

    return results


def plot_rg_count_density_retention(
    per_motif_df: pd.DataFrame,
    retention_df: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    Five panels:
      1. mean_delta_rg_count (RG count preservation)
      2. mean_delta_rg_density
      3. mean_delta_r_fraction (R vs G ratio preservation)
      4. mean_rg_retention (per-position R/G→R/G retention fraction)
      5. min_rg_retention (worst-conserved RG position)
    """
    pos = per_motif_df[per_motif_df["group"] == "pos"]
    neg = per_motif_df[per_motif_df["group"] == "neg"]
    pos_ret = retention_df[retention_df["group"] == "pos"]
    neg_ret = retention_df[retention_df["group"] == "neg"]

    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))

    results = {}
    results["delta_rg_count"] = _mwu_box_panel(
        axes[0],
        pos["mean_delta_rg_count"], neg["mean_delta_rg_count"],
        ylabel="Mean Δ RG count (homolog − query)",
        title="RG count change",
    )
    axes[0].axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)

    results["delta_rg_density"] = _mwu_box_panel(
        axes[1],
        pos["mean_delta_rg_density"], neg["mean_delta_rg_density"],
        ylabel="Mean Δ RG density",
        title="RG density change",
    )
    axes[1].axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)

    results["delta_r_fraction"] = _mwu_box_panel(
        axes[2],
        pos["mean_delta_r_fraction"], neg["mean_delta_r_fraction"],
        ylabel="Mean Δ R-fraction",
        title="R vs G ratio change",
    )
    axes[2].axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)

    results["mean_rg_retention"] = _mwu_box_panel(
        axes[3],
        pos_ret["mean_rg_retention"], neg_ret["mean_rg_retention"],
        ylabel="Mean R/G → R/G retention",
        title="Mean per-position RG retention",
    )

    results["min_rg_retention"] = _mwu_box_panel(
        axes[4],
        pos_ret["min_rg_retention"], neg_ret["min_rg_retention"],
        ylabel="Min R/G → R/G retention",
        title="Weakest-retained RG position",
    )

    fig.suptitle(
        f"RG composition & retention per motif ({dataset})",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "homolog_rg_composition_retention", dataset=dataset)

    print(f"\n── RG composition/retention ({dataset}) ──")
    for metric, r in results.items():
        print(f"  {metric:25s}: p = {r['p']:.2e} {r['sig']}")

    return results


# ════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_homolog_rg_metrics(
    df: pd.DataFrame,
    min_hits: int = 5,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    [Homolog-specific]
    End-to-end: per-hit metrics → per-motif aggregates → plots.
    Expects df to have motif_uid and harmonized group labels (use
    homolog_recruitment.add_motif_uid + harmonize_group_labels first).
    """
    print("  Computing per-hit RG metrics...")
    per_hit = compute_per_hit_rg_metrics(df)
    print(f"  {len(per_hit):,} hit-level observations")

    print("  Aggregating per-motif...")
    per_motif = aggregate_per_motif_rg_metrics(per_hit, min_hits=min_hits)
    print(f"  {len(per_motif)} motifs after min_hits={min_hits} filter")

    print("  Computing per-position RG retention...")
    retention = compute_per_position_rg_retention(df, min_hits=min_hits)
    print(f"  {len(retention)} motifs with RG retention data")

    print("  Plotting change events...")
    events_results = plot_rg_change_events_homologs(
        per_motif, dataset=dataset, save=save,
    )

    print("  Plotting composition/retention...")
    comp_results = plot_rg_count_density_retention(
        per_motif, retention, dataset=dataset, save=save,
    )

    return {
        "per_hit_df": per_hit,
        "per_motif_df": per_motif,
        "retention_df": retention,
        "events_results": events_results,
        "composition_results": comp_results,
    }



 
# ════════════════════════════════════════════════════════════════════════════
# Per-motif expected vs observed retention
# ════════════════════════════════════════════════════════════════════════════
 
def compute_expected_observed_retention(
    df: pd.DataFrame,
    qseq_col: str = "qseq_rg_region",
    hseq_col: str = "hseq_rg_region",
    motif_col: str = "motif_uid",
    group_col: str = "group",
    min_hits: int = 5,
) -> pd.DataFrame:
    """
    [Homolog-specific]
    For each motif:
      - For each query R/G position: fraction of homolog hits with R/G at that
        alignment column → observed retention.
      - For each query non-R/G position: fraction of homolog hits with the
        SAME query AA at that alignment column → expected (background)
        retention for that position. Aggregate to mean per motif.
 
    Observed and expected are directly comparable since both measure
    "fraction of homologs preserving what the query had" at different
    position classes within the same motif.
 
    Also reports per-motif gap rate: mean fraction of gap characters per
    column, across all columns.
    """
    rows = []
    for (motif_uid, group), sub in df.groupby([motif_col, group_col]):
        sub = sub[sub[qseq_col].notna() & sub[hseq_col].notna()]
        if len(sub) < min_hits:
            continue
        qseqs = sub[qseq_col].values
        hseqs = sub[hseq_col].values
        qseq = qseqs[0]
        L = len(qseq)
        valid_hits = [h for h in hseqs if len(h) == L]
        n_valid = len(valid_hits)
        if n_valid < min_hits:
            continue
 
        rg_positions = [i for i, c in enumerate(qseq) if c in RG]
        nonrg_positions = [i for i, c in enumerate(qseq) if c not in RG and c != "-"]
 
        if len(rg_positions) < 1 or len(nonrg_positions) < 1:
            continue
 
        # Observed: RG → R/G retention
        obs_retentions = []
        for i in rg_positions:
            column = [h[i] for h in valid_hits]
            retention = sum(1 for c in column if c in RG) / n_valid
            obs_retentions.append(retention)
        observed_retention_rg = float(np.mean(obs_retentions))
 
        # Expected: non-RG → same-AA retention
        exp_retentions = []
        for i in nonrg_positions:
            qc = qseq[i]
            column = [h[i] for h in valid_hits]
            retention = sum(1 for c in column if c == qc) / n_valid
            exp_retentions.append(retention)
        expected_retention_nonrg = float(np.mean(exp_retentions))
 
        # Per-motif gap rate: average gap fraction across columns
        gap_fracs = []
        for i in range(L):
            column = [h[i] for h in valid_hits]
            gap_fracs.append(sum(1 for c in column if c == "-") / n_valid)
        motif_gap_rate = float(np.mean(gap_fracs))
 
        rows.append({
            "motif_uid": motif_uid,
            "group": group,
            "n_rg_positions": len(rg_positions),
            "n_nonrg_positions": len(nonrg_positions),
            "n_hits": n_valid,
            "observed_retention_rg": observed_retention_rg,
            "expected_retention_nonrg": expected_retention_nonrg,
            "delta_retention": observed_retention_rg - expected_retention_nonrg,
            "motif_gap_rate": motif_gap_rate,
        })
    return pd.DataFrame(rows)
 
 
# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════
 
def plot_expected_observed_retention(
    comp_df: pd.DataFrame,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    Four panels:
      1. Observed RG retention (pos vs neg)
      2. Expected non-RG retention (pos vs neg)
      3. Δ retention (obs - exp), pos vs neg
      4. Motif gap rate (pos vs neg)
 
    Panel 3 is the gene-agnostic signal: if pos delta > neg delta, RG positions
    are specifically preserved beyond flank background in pos more than in neg.
    """
    pos = comp_df[comp_df["group"] == "pos"]
    neg = comp_df[comp_df["group"] == "neg"]
 
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
 
    results = {}
    results["observed"] = _mwu_box_panel(
        axes[0],
        pos["observed_retention_rg"], neg["observed_retention_rg"],
        ylabel="Fraction of homologs with R/G\nat RG position (observed)",
        title="RG position retention",
    )
    results["expected"] = _mwu_box_panel(
        axes[1],
        pos["expected_retention_nonrg"], neg["expected_retention_nonrg"],
        ylabel="Fraction of homologs with same AA\nat non-RG position (expected)",
        title="Non-RG position retention (background)",
    )
    results["delta"] = _mwu_box_panel(
        axes[2],
        pos["delta_retention"], neg["delta_retention"],
        ylabel="Δ retention (observed − expected)",
        title="Within-motif RG-specific conservation",
    )
    axes[2].axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
 
    results["gap_rate"] = _mwu_box_panel(
        axes[3],
        pos["motif_gap_rate"], neg["motif_gap_rate"],
        ylabel="Mean gap fraction per column",
        title="Motif alignment gap rate",
    )
 
    # One-sample test: is delta > 0 within each group?
    # (positive = RG better preserved than non-RG background)
    pos_delta = pos["delta_retention"].dropna()
    neg_delta = neg["delta_retention"].dropna()
    _, p_pos = stats.wilcoxon(pos_delta, alternative="greater")
    _, p_neg = stats.wilcoxon(neg_delta, alternative="greater")
 
    fig.suptitle(
        f"RG retention — observed vs expected (within-motif background) "
        f"({dataset})\n(Δ > 0 = RG specifically preserved beyond flanking)",
        fontsize=10, y=1.04,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "homolog_rg_expected_observed_retention", dataset=dataset)
 
    # Summary print
    print(f"\n── Expected vs observed RG retention ({dataset}) ──")
    print(f"  Motifs analyzed: pos = {len(pos)}, neg = {len(neg)}")
    print(f"\n  Observed RG retention:")
    print(f"    pos median = {pos['observed_retention_rg'].median():.3f}, "
          f"neg median = {neg['observed_retention_rg'].median():.3f}")
    print(f"    Mann-Whitney p = {results['observed']['p']:.2e} "
          f"{results['observed']['sig']}")
 
    print(f"\n  Expected (non-RG) retention:")
    print(f"    pos median = {pos['expected_retention_nonrg'].median():.3f}, "
          f"neg median = {neg['expected_retention_nonrg'].median():.3f}")
    print(f"    Mann-Whitney p = {results['expected']['p']:.2e} "
          f"{results['expected']['sig']}")
 
    print(f"\n  Δ retention (obs − exp):")
    print(f"    pos median = {pos['delta_retention'].median():.3f}, "
          f"neg median = {neg['delta_retention'].median():.3f}")
    print(f"    Pos vs neg: Mann-Whitney p = {results['delta']['p']:.2e} "
          f"{results['delta']['sig']}")
 
    print(f"\n  Within-group Wilcoxon (is delta > 0? i.e., RG specifically preserved):")
    print(f"    pos: p = {p_pos:.2e}")
    print(f"    neg: p = {p_neg:.2e}")
 
    print(f"\n  Motif gap rate:")
    print(f"    pos median = {pos['motif_gap_rate'].median():.3f}, "
          f"neg median = {neg['motif_gap_rate'].median():.3f}")
    print(f"    Mann-Whitney p = {results['gap_rate']['p']:.2e} "
          f"{results['gap_rate']['sig']}")
 
    return {
        "comp_df": comp_df,
        "observed_test": results["observed"],
        "expected_test": results["expected"],
        "delta_test": results["delta"],
        "gap_test": results["gap_rate"],
        "within_pos_wilcoxon_p": float(p_pos),
        "within_neg_wilcoxon_p": float(p_neg),
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ════════════════════════════════════════════════════════════════════════════
 
def run_rg_expected_vs_observed(
    df: pd.DataFrame,
    min_hits: int = 5,
    dataset: str = "homologs",
    save: bool = True,
) -> dict:
    """
    [Homolog-specific]
    End-to-end expected-vs-observed RG retention analysis.
    Requires df with motif_uid and harmonized group labels.
    """
    print("  Computing expected-vs-observed RG retention per motif...")
    comp_df = compute_expected_observed_retention(df, min_hits=min_hits)
    print(f"  {len(comp_df)} motifs analyzed")
 
    print("  Plotting...")
    results = plot_expected_observed_retention(
        comp_df, dataset=dataset, save=save,
    )
    return results
 

    