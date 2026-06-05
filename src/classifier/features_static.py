"""
features_static.py  —  REGISTRY of FOLD-STATIC features.

Contract for everything in this file:
  * A feature is fold-static iff swapping which regions are train vs test does
    NOT change the feature's value for a given region. (No label/group contrast
    inside the computation.)
  * Therefore everything here is safe to compute ONCE, up front, on the full
    dataset. Nothing in this file may be label-derived.
  * This module does NOT own feature logic. It IMPORTS the real implementations
    from their home modules and only orchestrates + joins them on region_id.

If a feature needs to be refit per CV fold (anything that contrasts pos vs neg),
it does NOT belong here — it goes in features_folded.py as a transformer.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Import feature implementations from their home modules.
# >>> POINT THESE AT THE REAL MODULES <<<
# from region_analysis import compute_consequence_per_region
from src.classifier.classifier_features import compute_alphamissense_per_region   # <-- adjust path
from src.classifier.classifier_features import compute_consequence_per_region   # <-- adjust path
from src.analysis_visualization.af_spectrum import compute_af_features_per_region
from src.classifier.classifier_features import compute_esm_per_region
from src.analysis_visualization.rg_analysis import compute_rg_features_per_region
from src.analysis_visualization.physchem_analysis import (compute_physchem_deltas,            # <-- adjust path
                               aggregate_per_region,
                               compute_wt_physchem_features)
from src.classifier.classifier_features import compute_codon_usage_features              # <-- adjust path
from src.analysis_visualization.codon_usage import compute_gc_codon_indices_per_region
# ... future static features get imported here from their own modules ...
 
#  Cache path for the expensive physchem-delta computation.
_PHYSCHEM_DELTA_CACHE = (
    "/mnt/d/phd/scripts/16_ev_signature_predictor/data/processed/physchem_deltas.parquet"
)
# ---------------------------------------------------------------------------
 
JOIN_KEY = "region_id"
 
# Columns that must never enter the feature matrix (label / grouping leakage).
_LABEL_COLS = {"group"}
 
 
def _strip_label_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive: drop any label/grouping columns that rode along in a builder."""
    drop = [c for c in df.columns if c in _LABEL_COLS]
    return df.drop(columns=drop) if drop else df
 
 
def _set_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a unique region_id index for clean joins."""
    if JOIN_KEY not in df.columns and df.index.name != JOIN_KEY:
        raise ValueError(f"feature frame has no '{JOIN_KEY}' to key on")
    if df.index.name != JOIN_KEY:
        df = df.set_index(JOIN_KEY)
    if not df.index.is_unique:
        raise ValueError(f"'{JOIN_KEY}' is not unique after indexing "
                         f"(duplicates: {df.index[df.index.duplicated()].unique()[:5]})")
    return df
 
 
def _load_or_compute_physchem_deltas(df_rg, region_by_id, cache_path):
    """Load cached physchem deltas if present, else compute and save them.
    compute_physchem_deltas is expensive, so we avoid recomputing per call."""
    if cache_path:
        try:
            return pd.read_parquet(cache_path)
        except (FileNotFoundError, OSError):
            pass
    deltas = compute_physchem_deltas(df_rg, region_by_id)
    if cache_path:
        try:
            deltas.to_parquet(cache_path)
        except OSError:
            pass  # caching is best-effort; don't fail the build if write fails
    return deltas
 
 
def build_static_features(df_rg: pd.DataFrame, region_by_id: dict = None,
                          df_esm: pd.DataFrame = None,
                          include_wt_physchem: bool = True,
                          codon_source_aas: list = None,
                          include_gc_codon_indices: bool = True,
                          codon_rates: dict = None,
                          physchem_delta_cache: str = _PHYSCHEM_DELTA_CACHE) -> pd.DataFrame:
    """
    Assemble ALL fold-static features into one matrix keyed on region_id.
 
    Parameters
    ----------
    df_rg : variant-level dataframe (one row per variant).
    region_by_id : dict of region metadata; required for the RG, physchem-delta,
            WT-physchem, and codon-usage groups. If None, those groups are skipped.
    df_esm : optional variant-level dataframe with esm_llr. If None, ESM skipped.
    include_wt_physchem : if False, the WT (baseline-sequence) physchem group is
            omitted. Flip to False for the with/without-WT ablation, since WT
            physchem is the strongest potential baseline-composition label proxy.
    codon_source_aas : list of amino-acid letters whose codon-usage fractions to
            include (e.g. list("RG") or list("ADEGLPRS")). None = all multi-codon
            AAs (~50+ collinear columns; usually too many). Restrict to the
            biologically-relevant set to keep dimensionality and importance sane.
    include_gc_codon_indices : if False, omit the GC/GC3/CpG/mutability scalar
            index group. (ENC and CAI are intentionally excluded entirely.)
    codon_rates : optional 1KG mutation-rate dict; if provided, the GC-codon
            group adds a mean intrinsic codon-mutability column.
    physchem_delta_cache : parquet path for the expensive physchem-delta result;
            loaded if present, otherwise recomputed and saved.
 
    Returns
    -------
    DataFrame indexed by region_id, fold-static features only, no label columns.
    """
    parts = []
 
    # 1) consequence + variant-density features (imported, not reimplemented)
    cons = compute_consequence_per_region(df_rg)
    cons = _strip_label_cols(cons)
    cons = _set_index(cons)
    parts.append(cons)
 
    # 2) AlphaMissense per-region aggregations (imported, not reimplemented)
    am = compute_alphamissense_per_region(df_rg)
    am = _strip_label_cols(am)
    am = _set_index(am)
    parts.append(am)
 
    # 3) AF-derived per-region features (imported, not reimplemented)
    #    NOTE: af_am_score_weighted_by_rarity & af_n_likely_path_rare carry the
    #    AlphaMissense circularity caveat — exclude from orthogonal-selection claims.
    af = compute_af_features_per_region(df_rg)
    af = _strip_label_cols(af)
    af = _set_index(af)
    parts.append(af)
 
    # 4) ESM1b LLR per-region aggregations (imported, not reimplemented).
    #    Computed from df_esm, not df_rg; skipped if df_esm is None (mirrors
    #    the orchestrator). ESM is sequence-only -> orthogonal to the AF signal.
    if df_esm is not None:
        esm = compute_esm_per_region(df_esm)
        esm = _strip_label_cols(esm)
        esm = _set_index(esm)
        parts.append(esm)
 
    # 5) RG-specific features: density, burden, R/G asymmetry, change events,
    #    delta RG ratio (imported, not reimplemented). Derives RG change events
    #    internally from df_rg + region_by_id. region_length is dropped inside
    #    the builder to avoid colliding with the consequence group's copy.
    if region_by_id is not None:
        rg = compute_rg_features_per_region(df_rg, region_by_id)
        rg = _strip_label_cols(rg)
        rg = _set_index(rg)
        parts.append(rg)
 
    # 6) physchem DELTAS: per-region mean shift in biophysical properties across
    #    missense variants (variant-driven; on-narrative for selection). Derived
    #    from df_rg + region_by_id, cached to parquet (expensive to recompute).
    if region_by_id is not None:
        deltas_df = _load_or_compute_physchem_deltas(df_rg, region_by_id,
                                                     physchem_delta_cache)
        pc_delta = aggregate_per_region(deltas_df)
        pc_delta = _strip_label_cols(pc_delta)
        pc_delta = _set_index(pc_delta)
        parts.append(pc_delta)
 
    # 7) WT physchem: baseline sequence properties (no variants). OPTIONAL —
    #    this is the strongest potential baseline-composition label proxy, so it
    #    is toggleable for the with/without-WT ablation. Also note wt_* charge
    #    features correlate with the RG-density group (both track R/G content).
    if include_wt_physchem and region_by_id is not None:
        wt = compute_wt_physchem_features(region_by_id)
        wt = _strip_label_cols(wt)
        wt = _set_index(wt)
        parts.append(wt)
 
    # 8) codon-usage fractions per region (leak-free: pure-sequence, no label
    #    contrast). Restricted to codon_source_aas to control dimensionality;
    #    within each AA the fractions sum to 1 (one column redundant by design).
    #    Same baseline-composition label-proxy caveat as WT physchem.
    if region_by_id is not None:
        codon = compute_codon_usage_features(region_by_id, source_aas=codon_source_aas)
        codon = _strip_label_cols(codon)
        codon = _set_index(codon)
        parts.append(codon)
 
    # 9) GC / codon composition indices: gc, gc3 (wobble), cpg_frac, cpg_oe, and
    #    (if codon_rates given) mean intrinsic codon mutability. Compact scalar
    #    summaries complementing the codon fractions; leak-free, fold-static.
    #    Toggleable. ENC/CAI intentionally excluded (noisy at short lengths /
    #    off-narrative).
    if include_gc_codon_indices and region_by_id is not None:
        gci = compute_gc_codon_indices_per_region(region_by_id, rates=codon_rates)
        gci = _strip_label_cols(gci)
        gci = _set_index(gci)
        parts.append(gci)
 
    # 10) future static feature groups append here, each keyed on region_id:
    # codon = compute_codon_features_per_region(df_rg); parts.append(_set_index(_strip_label_cols(codon)))
    # phys  = compute_physchem_per_region(df_rg);       parts.append(_set_index(_strip_label_cols(phys)))
 
    # join on region_id; 'outer' so a region missing from one builder isn't dropped
    X = pd.concat(parts, axis=1, join="outer")
 
    # guard against accidental duplicate column names across builders
    dupes = X.columns[X.columns.duplicated()].unique()
    if len(dupes):
        raise ValueError(f"duplicate feature columns across builders: {list(dupes)}")
    return X
 
