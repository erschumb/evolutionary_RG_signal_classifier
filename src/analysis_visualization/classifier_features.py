"""
Save as: src/analysis_visualization/classifier_features.py

Consolidated feature extraction for the false-negative RG motif classifier.

Aggregates all per-region features we've developed into a single dataframe,
one row per region, with:
  - label:                  pos (1) / neg (0)
  - sequence composition:   length, RG density, codon usage
  - variant-derived:        density, consequence breakdown, RG disruption,
                             physchem deltas, AlphaMissense, substitution classes
  - static WT physchem:     NCPR, FCR, kappa, hydropathy, aromaticity, etc.

Usage:
    features_df = build_classifier_features(
        df_rg=df_rg,              # variant df with hits_rg, is_rg_disrupting
        df_events=df_events,      # from compute_rg_change_events
        region_by_id=region_by_id,
    )
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import re

# Reuse existing module functions
from src.analysis_visualization.region_analysis import (
    compute_variant_density, collapse_consequence, CONSEQUENCE_ORDER,
)
from src.analysis_visualization.rg_analysis import (
    compute_region_rg_stats, compute_per_rg_burden,
    compute_per_region_burden_stats, VARIANT_TYPES,
    compute_rg_change_events, RG_EVENT_TYPES,
    compute_delta_rg_ratio,
)
from src.analysis_visualization.physchem_analysis import (
    compute_physchem_deltas, aggregate_per_region, compute_wt_physchem,
    PHYSCHEM_FEATURES,
)
from src.analysis_visualization.codon_usage import (
    compute_codon_counts, _AA_TO_CODONS, MULTI_CODON_AA, _CODON_TABLE,
)
from src.analysis_visualization.substitution_matrix_analysis import AA_GROUPS


# ════════════════════════════════════════════════════════════════════════════
# Substitution class definitions (Option B: biochemically meaningful classes)
# ════════════════════════════════════════════════════════════════════════════

POSITIVE_AAS = {"R", "K"}
NEGATIVE_AAS = {"D", "E"}
CHARGED_AAS = POSITIVE_AAS | NEGATIVE_AAS
AROMATIC_AAS = {"F", "Y", "W"}
HYDROPHOBIC_AAS = {"A", "V", "I", "L", "M"}
POLAR_AAS = {"S", "T", "N", "Q"}
FLEX_AAS = {"G"}
PROLINE = {"P"}


def _classify_substitution(before: str, after: str) -> dict:
    """
    Binary flags for each substitution class. For a single missense variant.
    """
    return {
        "is_charge_alter": (
            (before in POSITIVE_AAS) != (after in POSITIVE_AAS)
            or (before in NEGATIVE_AAS) != (after in NEGATIVE_AAS)
        ),
        "is_pos_to_anything": before in POSITIVE_AAS and after not in POSITIVE_AAS,
        "is_aromatic_alter": before in AROMATIC_AAS and after not in AROMATIC_AAS,
        "is_hydrophobic_alter": before in HYDROPHOBIC_AAS and after not in HYDROPHOBIC_AAS,
        "is_polar_alter": before in POLAR_AAS and after not in POLAR_AAS,
        "is_proline_intro": after in PROLINE and before not in PROLINE,
        "is_glycine_intro": after in FLEX_AAS and before not in FLEX_AAS,
        "is_conservative": (
            AA_GROUPS.get(before) == AA_GROUPS.get(after)
            if before in AA_GROUPS and after in AA_GROUPS
            else False
        ),
    }


SUBSTITUTION_CLASSES = [
    "charge_alter", "pos_to_anything", "aromatic_alter",
    "hydrophobic_alter", "polar_alter", "proline_intro",
    "glycine_intro", "conservative",
]


def compute_substitution_class_rates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-region rate of each substitution class, normalized by region length
    (requires region_length column or we compute it separately).
    """
    df = df[
        df["Consequence"].fillna("").str.contains("missense_variant") &
        df["before_aa"].notna() & df["after_aa"].notna() &
        (df["before_aa"].str.len() == 1) & (df["after_aa"].str.len() == 1)
    ].copy()

    # Classify each variant
    flags = df.apply(
        lambda r: _classify_substitution(r["before_aa"], r["after_aa"]),
        axis=1, result_type="expand",
    )
    df = pd.concat([df, flags], axis=1)

    # Per-region sums of each class
    agg_cols = [f"is_{c}" for c in SUBSTITUTION_CLASSES]
    per_region = (
        df.groupby(["region_id", "group"])[agg_cols]
          .sum()
          .reset_index()
    )
    per_region.columns = [
        c.replace("is_", "n_sub_") if c.startswith("is_") else c
        for c in per_region.columns
    ]
    return per_region


# ════════════════════════════════════════════════════════════════════════════
# AlphaMissense aggregations
# ════════════════════════════════════════════════════════════════════════════

def compute_alphamissense_per_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-statistic AlphaMissense aggregation per region.
    """
    sub = df[
        df["Consequence"].fillna("").str.contains("missense_variant") &
        df["am_pathogenicity"].notna()
    ]

    def _agg(g):
        return pd.Series({
            "am_median": g["am_pathogenicity"].median(),
            "am_mean":   g["am_pathogenicity"].mean(),
            "am_max":    g["am_pathogenicity"].max(),
            "am_std":    g["am_pathogenicity"].std(ddof=0),
            "am_fraction_pathogenic": (g["am_pathogenicity"] > 0.5).mean(),
        })

    return sub.groupby(["region_id", "group"]).apply(_agg).reset_index()


# ════════════════════════════════════════════════════════════════════════════
# Per-region consequence density + proportions
# ════════════════════════════════════════════════════════════════════════════

def compute_consequence_per_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts, densities, and proportions of each consequence class per region.
    """
    df = df.copy()
    df["consequence_class"] = df["Consequence"].apply(collapse_consequence)
    # Region length
    region_meta = df.groupby(["region_id", "group"])[
        ["region_start_aa", "region_end_aa"]
    ].first()
    region_meta["region_length"] = (
        region_meta["region_end_aa"] - region_meta["region_start_aa"] + 1
    )
    region_meta = region_meta.reset_index()[
        ["region_id", "group", "region_length"]
    ]

    # Per-region counts per consequence class
    counts = (
        df.groupby(["region_id", "group", "consequence_class"])
          .size()
          .unstack("consequence_class", fill_value=0)
          .reset_index()
    )
    for c in CONSEQUENCE_ORDER:
        if c not in counts.columns:
            counts[c] = 0

    merged = counts.merge(region_meta, on=["region_id", "group"])
    merged["n_variants_total"] = merged[CONSEQUENCE_ORDER].sum(axis=1)
    merged["variant_density"] = merged["n_variants_total"] / merged["region_length"]

    for c in CONSEQUENCE_ORDER:
        merged[f"density_{c}"] = merged[c] / merged["region_length"]
        merged[f"fraction_{c}"] = np.where(
            merged["n_variants_total"] > 0,
            merged[c] / merged["n_variants_total"],
            0.0,
        )
        # Rename the raw count column with a clearer prefix
        merged = merged.rename(columns={c: f"n_{c}"})

    return merged


# ════════════════════════════════════════════════════════════════════════════
# RG burden features per region
# ════════════════════════════════════════════════════════════════════════════

def compute_rg_features_per_region(
    df_rg: pd.DataFrame, region_by_id: dict,
) -> pd.DataFrame:
    """
    Wraps compute_per_rg_burden + compute_per_region_burden_stats and pivots
    into wide form: one row per region, columns per (variant_type, metric).
    Also adds RG density + R vs G asymmetry.
    """
    # RG density per region
    region_stats = compute_region_rg_stats(region_by_id)[
        ["region_id", "region_length", "n_rg_motifs", "rg_fraction"]
    ]

    # Per-RG burden → per-region burden stats
    per_rg = compute_per_rg_burden(df_rg, region_by_id)
    per_region_burden = compute_per_region_burden_stats(per_rg)

    # Pivot to wide
    wide = per_region_burden.pivot_table(
        index=["region_id", "group"],
        columns="variant_type",
        values=["fraction_rgs_hit", "mean_burden_on_hit"],
        aggfunc="first",
    )
    # Flatten column multi-index
    wide.columns = [f"rg_{metric}_{vt}" for metric, vt in wide.columns]
    wide = wide.reset_index()

    # R vs G asymmetry (missense only)
    r_g_sub = df_rg[
        df_rg["is_rg_disrupting"].fillna(False) &
        df_rg["Consequence"].fillna("").str.contains("missense_variant") &
        df_rg["rg_role"].isin(["R", "G"])
    ]
    asym = (
        r_g_sub.groupby(["region_id", "group", "rg_role"])
               .size()
               .unstack("rg_role", fill_value=0)
               .reset_index()
    )
    if "R" not in asym.columns: asym["R"] = 0
    if "G" not in asym.columns: asym["G"] = 0
    asym["rg_r_fraction"] = np.where(
        (asym["R"] + asym["G"]) > 0,
        asym["R"] / (asym["R"] + asym["G"]),
        np.nan,
    )
    asym = asym.rename(columns={"R": "n_r_hits_disrupting", "G": "n_g_hits_disrupting"})

    # Merge
    out = region_stats.merge(wide, on="region_id", how="left")
    out = out.merge(asym, on=["region_id"], how="left")
    # Fill NaN burden-related columns with 0 (no observation)
    rg_cols = [c for c in out.columns if c.startswith("rg_fraction_rgs_hit_")
               or c.startswith("rg_mean_burden_on_hit_")]
    for c in rg_cols:
        # Only zero-fill "no observation" cases; keep NaN for regions with no RG
        pass  # actually leave NaN where it is; classifier will handle
    return out


# ════════════════════════════════════════════════════════════════════════════
# RG event features per region (loss/gain/movement proportions)
# ════════════════════════════════════════════════════════════════════════════

def compute_rg_event_features(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Per-region fractions of each RG change event (no_change, loss, gain,
    movement) among missense variants.
    """
    events = df_events[df_events["rg_change_event"].notna()]
    counts = (
        events.groupby(["region_id", "group", "rg_change_event"])
              .size()
              .unstack("rg_change_event", fill_value=0)
              .reset_index()
    )
    for ev in RG_EVENT_TYPES:
        if ev not in counts.columns:
            counts[ev] = 0
    total = counts[RG_EVENT_TYPES].sum(axis=1).replace(0, np.nan)
    for ev in RG_EVENT_TYPES:
        counts[f"rg_event_fraction_{ev}"] = counts[ev] / total
    # Keep only proportions + metadata (drop raw counts here to reduce redundancy)
    keep = ["region_id", "group"] + [f"rg_event_fraction_{ev}" for ev in RG_EVENT_TYPES]
    return counts[keep]


# ════════════════════════════════════════════════════════════════════════════
# Delta RG ratio per region
# ════════════════════════════════════════════════════════════════════════════

def compute_delta_rg_ratio_per_region(
    df_rg: pd.DataFrame, region_by_id: dict,
) -> pd.DataFrame:
    """
    Per-region mean relative delta RG ratio across all missense variants.
    """
    df = compute_delta_rg_ratio(df_rg, region_by_id)
    df = df[df["delta_rg_ratio_rel"].notna()]
    per_region = (
        df.groupby(["region_id", "group"])["delta_rg_ratio_rel"]
          .mean()
          .reset_index(name="delta_rg_ratio_rel_mean")
    )
    return per_region


# ════════════════════════════════════════════════════════════════════════════
# Codon usage features per region
# ════════════════════════════════════════════════════════════════════════════

def compute_codon_usage_features(region_by_id: dict) -> pd.DataFrame:
    """
    For each region and each multi-codon AA, compute fraction of that AA's
    codons that are each specific codon. E.g. fraction of G codons that are
    GGC, GGA, GGG, GGT.

    Returns: one row per region with columns like 'codon_G_GGC', 'codon_G_GGA', ...
    """
    rows = []
    for rid, r in region_by_id.items():
        dna = r.get("dna", "")
        if not dna or len(dna) % 3 != 0:
            continue
        row = {"region_id": rid, "group": r["group"]}
        codons = [dna[i:i + 3].upper() for i in range(0, len(dna), 3)]

        for aa in MULTI_CODON_AA:
            aa_codons = _AA_TO_CODONS[aa]
            counts = {c: 0 for c in aa_codons}
            for codon in codons:
                if codon in counts:
                    counts[codon] += 1
            total = sum(counts.values())
            for c in aa_codons:
                row[f"codon_{aa}_{c}"] = (
                    counts[c] / total if total > 0 else np.nan
                )
        rows.append(row)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Substitution class rates per region (normalized by region length)
# ════════════════════════════════════════════════════════════════════════════

def compute_substitution_class_features(
    df_rg: pd.DataFrame, region_by_id: dict,
) -> pd.DataFrame:
    """
    Per-region rate of each substitution class, normalized by region length.
    """
    # Get per-region substitution counts
    counts = compute_substitution_class_rates(df_rg)

    # Add region length
    lengths = pd.DataFrame([
        {"region_id": rid, "region_length": len(r["prot_seq"])}
        for rid, r in region_by_id.items()
    ])
    merged = counts.merge(lengths, on="region_id", how="left")

    # Convert counts to rates per residue
    for cls in SUBSTITUTION_CLASSES:
        merged[f"sub_rate_{cls}"] = merged[f"n_sub_{cls}"] / merged["region_length"]

    keep = ["region_id", "group"] + [f"sub_rate_{cls}" for cls in SUBSTITUTION_CLASSES]
    return merged[keep]


# ════════════════════════════════════════════════════════════════════════════
# Physchem WT features
# ════════════════════════════════════════════════════════════════════════════

def compute_wt_physchem_features(region_by_id: dict) -> pd.DataFrame:
    """
    WT physchem per region (baseline sequence properties).
    """
    wt = compute_wt_physchem(region_by_id)
    wt = wt.rename(
        columns={f: f"wt_{f}" for f in PHYSCHEM_FEATURES}
    )
    return wt


# ════════════════════════════════════════════════════════════════════════════
# MAIN: build the full feature dataframe
# ════════════════════════════════════════════════════════════════════════════

def build_classifier_features(
    df_rg: pd.DataFrame,
    df_events: pd.DataFrame,
    region_by_id: dict,
    physchem_deltas_df: pd.DataFrame | None = None,
    dataset: str = "gnomad",
) -> pd.DataFrame:
    """
    End-to-end: build one feature dataframe for the classifier.

    Parameters
    ----------
    df_rg
        Variant dataframe with RG disruption columns (hits_rg, is_rg_disrupting,
        rg_role, before_aa, after_aa, am_pathogenicity, etc.)
    df_events
        From compute_rg_change_events(df_rg, region_by_id).
    region_by_id
        Patched region dict (with half-open regions fixed, broken regions removed).
    physchem_deltas_df
        Optional. If provided, the pre-computed per-variant physchem deltas
        (output of compute_physchem_deltas). Saves recomputation. If None,
        skips per-variant delta features.
    dataset
        Label, currently just for logging.

    Returns
    -------
    features_df
        One row per region with all features + `group` (label).
    """
    print(f"Building classifier features for dataset: {dataset}")

    # Start with consequence features (includes region_length, variant_density)
    print("  1/8 consequence + variant density...")
    cons_df = compute_consequence_per_region(df_rg)
    # Keep per-consequence counts + densities + fractions, drop raw consequence class column
    features = cons_df.copy()

    # AlphaMissense
    print("  2/8 AlphaMissense...")
    am_df = compute_alphamissense_per_region(df_rg)
    features = features.merge(am_df, on=["region_id", "group"], how="left")

    # RG density + burden
    print("  3/8 RG density + burden...")
    rg_feats = compute_rg_features_per_region(df_rg, region_by_id)
    # Drop columns already in `features`
    dup = [c for c in rg_feats.columns if c in features.columns and c != "region_id"]
    features = features.merge(
        rg_feats.drop(columns=dup, errors="ignore"),
        on="region_id", how="left",
    )

    # RG event features
    print("  4/8 RG change events...")
    rg_ev = compute_rg_event_features(df_events)
    features = features.merge(rg_ev, on=["region_id", "group"], how="left")

    # Delta RG ratio per region
    print("  5/8 Δ RG ratio...")
    drg = compute_delta_rg_ratio_per_region(df_rg, region_by_id)
    features = features.merge(drg, on=["region_id", "group"], how="left")

    # Substitution class rates
    print("  6/8 substitution class rates...")
    sub_feats = compute_substitution_class_features(df_rg, region_by_id)
    features = features.merge(sub_feats, on=["region_id", "group"], how="left")

    # Physchem deltas per region (if available) + WT physchem
    if physchem_deltas_df is not None:
        print("  7/8 physchem deltas (per-region means)...")
        pc_mean = aggregate_per_region(physchem_deltas_df)
        features = features.merge(pc_mean, on=["region_id", "group"], how="left")
    else:
        print("  7/8 physchem deltas SKIPPED (no physchem_deltas_df provided)")

    print("  7/8 WT physchem...")
    wt_pc = compute_wt_physchem_features(region_by_id)
    features = features.merge(wt_pc, on=["region_id", "group"], how="left")

    # Codon usage
    print("  8/8 codon usage...")
    codon_feats = compute_codon_usage_features(region_by_id)
    features = features.merge(codon_feats, on=["region_id", "group"], how="left")

    # Encode label
    features["label"] = (features["group"] == "pos").astype(int)

    # Reorder: region_id, group, label first
    front = ["region_id", "group", "label"]
    rest = [c for c in features.columns if c not in front]
    features = features[front + rest]

    print(f"\nDone. Final shape: {features.shape[0]} regions × "
          f"{features.shape[1] - 3} features (+ region_id, group, label)")
    print(f"Label balance: pos = {(features['label'] == 1).sum()}, "
          f"neg = {(features['label'] == 0).sum()}")

    # Report NaN summary
    n_nan = features.drop(columns=front).isna().sum()
    n_nan_cols = (n_nan > 0).sum()
    print(f"\nColumns with NaN values: {n_nan_cols}")
    if n_nan_cols > 0:
        print("Top 10 NaN-heavy columns:")
        print(n_nan[n_nan > 0].sort_values(ascending=False).head(10))

    return features


# ════════════════════════════════════════════════════════════════════════════
# Utility: save + quick summary
# ════════════════════════════════════════════════════════════════════════════

def save_features(
    features_df: pd.DataFrame,
    output_path: str = "/mnt/d/phd/scripts/16_ev_signature_predictor/"
                       "data/processed/classifier_features_gnomad.parquet",
) -> None:
    """Save the features dataframe as parquet."""
    features_df.to_parquet(output_path)
    print(f"Saved: {output_path}")
    print(f"  Shape: {features_df.shape}")


def feature_summary(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick descriptive stats per feature, by group.
    Useful sanity check before training.
    """
    feat_cols = [c for c in features_df.columns
                 if c not in ("region_id", "group", "label")]
    summary = (
        features_df.groupby("group")[feat_cols]
                   .describe()
                   .stack(level=0)
                   .reset_index()
                   .rename(columns={"level_1": "feature"})
    )
    return summary