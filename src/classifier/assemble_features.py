"""
assemble_features.py — the glue between raw inputs and the CV harness.

Single responsibility: turn the raw project inputs (df_rg, region_by_id, df_esm,
+ the mutation-rate path) into the four objects every downstream analysis needs:

    X_static : pd.DataFrame   fold-static feature matrix, indexed by region_id
    y        : pd.Series       binary label (1=functional/pos), indexed by region_id
    groups   : pd.Series       accession per region (CV grouping), indexed by region_id
    factory  : callable        builds a FRESH SubstitutionScoreTransformer per fold

Holds NO model logic and NO CV logic — those live in nested_cv.py. The notebook
calls assemble_everything() once, then hands the four outputs to the harness.

All three of X_static, y, groups share the SAME region_id index, so the harness
can slice them consistently per fold. The label NEVER enters X_static.
"""

import pandas as pd

from src.classifier.features_static import build_static_features
from src.classifier.features_folded import SubstitutionScoreTransformer

def make_label_and_groups(region_by_id, group_col="group",
                          protein_key="protein", pos_label="pos"):
    """
    Build y (binary label) and groups (accession) as Series indexed by region_id,
    taken from region_by_id — the AUTHORITATIVE, COMPLETE region list.
 
    This deliberately uses region_by_id rather than the variant table, because
    regions whose variants are all non-PASS (filtered out by QC) are absent from
    the variant table but ARE real regions. They are kept here with proper labels;
    their variant-derived features will be NaN (imputed at model time) while their
    pure-sequence features still apply.
 
    y      : 1 where region's group == pos_label, else 0
    groups : protein/accession per region (used for StratifiedGroupKFold)
    """
    rows = {}
    for rid, r in region_by_id.items():
        rows[rid] = (1 if r.get(group_col) == pos_label else 0,
                     r.get(protein_key))
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["label", "accession"])
    df.index.name = "region_id"
    y = df["label"]; y.name = "label"
    groups = df["accession"]; groups.name = "accession"
    return y, groups
 
 
def make_substitution_factory(df_rg, region_by_id, rates_path,
                              alpha=0.5, min_total=0, split_source=False,
                              from_col="aa_from", to_col="aa_to"):
    """
    Return a zero-arg callable that builds a FRESH SubstitutionScoreTransformer.
    The harness calls this once per fold so each fold gets an unfitted instance
    (avoids reusing a fitted transformer across folds).
 
    from_col/to_col name the single-letter source/destination AA columns in the
    VARIANT table (df_rg). Default 'aa_from'/'aa_to' matches the parsed columns.
    """
    def factory():
        return SubstitutionScoreTransformer(
            df_for_rg=df_rg,
            region_by_id=region_by_id,
            rates_path=rates_path,
            alpha=alpha,
            min_total=min_total,
            split_source=split_source,
            from_col=from_col,
            to_col=to_col,
        )
    return factory
 
 
def assemble_everything(
    df_rg,
    region_by_id,
    # df_esm=None,
    rates_path=None,
    *,
    # static-feature toggles (passed through to build_static_features)
    include_wt_physchem=True,
    include_gc_codon_indices=True,
    codon_source_aas=None,
    codon_rates=None,
    physchem_delta_cache=None,
    # substitution-transformer hyperparameters
    alpha=0.5,
    min_total=0,
    split_source=False,
    sub_from_col="aa_from",
    sub_to_col="aa_to",
    # column names
    region_id_col="region_id",
    group_col="group",
    protein_key="protein",
    pos_label="pos",
):
    """
    One-call assembly. Returns (X_static, y, groups, factory).
 
    Example
    -------
        X_static, y, groups, factory = assemble_everything(
            df_rg, region_by_id, df_esm=df_esm,
            rates_path=RATES_PATH, codon_source_aas=list("RG"))
        full = run_nested_cv(X_static, y, groups,
                             include_groups=list(FEATURE_GROUPS),
                             folded_transformer_factory=factory)
    """
    # fold-static feature matrix (computed once; safe — no label contrast)
    X_static = build_static_features(
        df_rg,
        region_by_id=region_by_id,
        # df_esm=df_esm,
        include_wt_physchem=include_wt_physchem,
        codon_source_aas=codon_source_aas,
        include_gc_codon_indices=include_gc_codon_indices,
        codon_rates=codon_rates,
        physchem_delta_cache=physchem_delta_cache,
    )
 
    # label + grouping, indexed by region_id — derived from region_by_id (the
    # complete region list), so variant-free regions are kept with proper labels
    y, groups = make_label_and_groups(
        region_by_id, group_col=group_col,
        protein_key=protein_key, pos_label=pos_label)
 
    # align y/groups to the static matrix's region_id index
    y = y.reindex(X_static.index)
    groups = groups.reindex(X_static.index)
 
    # sanity: no region without a label or accession after alignment
    if y.isna().any():
        missing = y.index[y.isna()].tolist()[:5]
        raise ValueError(f"{int(y.isna().sum())} regions in X_static have no label "
                         f"(e.g. {missing}); check region_id alignment")
    if groups.isna().any():
        missing = groups.index[groups.isna()].tolist()[:5]
        raise ValueError(f"{int(groups.isna().sum())} regions have no accession "
                         f"(e.g. {missing}); check {accession_col}")
 
    # folded-transformer factory (fresh instance per fold)
    factory = make_substitution_factory(
        df_rg, region_by_id, rates_path,
        alpha=alpha, min_total=min_total, split_source=split_source,
        from_col=sub_from_col, to_col=sub_to_col)
 
    return X_static, y, groups, factory
 
