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
# ... future static features get imported here from their own modules ...
# from codon_analysis import compute_codon_features_per_region
# from physchem import compute_physchem_per_region
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
 
 
def build_static_features(df_rg: pd.DataFrame, df_esm: pd.DataFrame = None) -> pd.DataFrame:
    """
    Assemble ALL fold-static features into one matrix keyed on region_id.
 
    Parameters
    ----------
    df_rg : variant-level dataframe (one row per variant), the same input the
            individual builders expect.
    df_esm : optional variant-level dataframe with the esm_llr column. If None,
            the ESM feature group is skipped (matching the orchestrator's
            conditional behavior).
 
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
 
    # 5) future static feature groups append here, each keyed on region_id:
    # codon = compute_codon_features_per_region(df_rg); parts.append(_set_index(_strip_label_cols(codon)))
    # phys  = compute_physchem_per_region(df_rg);       parts.append(_set_index(_strip_label_cols(phys)))
 
    # join on region_id; 'outer' so a region missing from one builder isn't dropped
    X = pd.concat(parts, axis=1, join="outer")
 
    # guard against accidental duplicate column names across builders
    dupes = X.columns[X.columns.duplicated()].unique()
    if len(dupes):
        raise ValueError(f"duplicate feature columns across builders: {list(dupes)}")
    return X
 
