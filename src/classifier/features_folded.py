"""
features_folded.py  —  REGISTRY of FOLD-DERIVED features (label-contrast).

Contract for everything in this file:
  * A feature is fold-derived iff its value for a region depends on a quantity
    estimated by contrasting the pos/neg groups (e.g. the substitution score
    table). Swapping train/test CHANGES the value -> it must be refit per fold.
  * Therefore each fold-derived feature is a scikit-learn-style transformer:
      - fit(X, y)  : sees ONLY the training fold; builds whatever table it needs
      - transform(X): applies the fitted table to any rows (train or held-out)
    Dropped into the Pipeline before the classifier, these refit per fold for
    free, so held-out regions never inform their own encoding.
  * Like features_static.py, this module ORCHESTRATES; it imports the real
    scoring logic from its home module.

NOTE: this is a SCAFFOLD. The substitution transformer below is stubbed at the
fit/transform boundary because it needs VARIANT-LEVEL data (region_id, from, to,
group) + the expected-count model to recompute counts per fold. We fill it in
once that input is wired.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
 
# --- import the REAL functions from their home modules (do not reimplement) ---
# from src.analysis_visualization.substitution_matrix_analysis import (
#     compute_composition_normalized_enrichment, result_to_table, load_mutation_rates)
# from src.analysis_visualization.substitution_matrix_analysis import build_score_table
# For standalone testing we import the stubs:
from src.analysis_visualization.substitution_matrix_analysis import (
    compute_composition_normalized_enrichment, result_to_table,
    load_mutation_rates, build_score_table)
 
def _as_region_ids(X):
    """Accept a Series, 1-col DataFrame, index, or array of region_ids."""
    if isinstance(X, pd.DataFrame):
        if "region_id" in X.columns:
            return X["region_id"].tolist()
        if X.shape[1] == 1:
            return X.iloc[:, 0].tolist()
        return X.index.tolist()
    if isinstance(X, pd.Series):
        return X.tolist()
    return list(X)
 
 
class SubstitutionScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, df_for_rg, region_by_id, rates_path=None,
                 alpha=0.5, min_count=0, min_total=0,
                 split_source=False, source_residues=("R", "G", "A", "D", "L", "E", "P", "S"),
                 missense_only=True, consequence_col="Consequence",
                 region_col="region_id", group_col="group",
                 from_col="from", to_col="to", key_sep="->"):
        # data + config held whole; fit slices per fold
        self.df_for_rg = df_for_rg
        self.region_by_id = region_by_id
        self.rates_path = rates_path
        self.alpha = alpha
        self.min_count = min_count
        self.min_total = min_total
        self.split_source = split_source
        self.source_residues = source_residues
        self.missense_only = missense_only
        self.consequence_col = consequence_col
        self.region_col = region_col
        self.group_col = group_col
        self.from_col = from_col
        self.to_col = to_col
        self.key_sep = key_sep
 
    # ---- helpers -----------------------------------------------------------
    def _variants_for(self, region_ids):
        """Rows of df_for_rg belonging to region_ids (optionally missense-only)."""
        d = self.df_for_rg[self.df_for_rg[self.region_col].isin(set(region_ids))]
        if self.missense_only and self.consequence_col in d.columns:
            d = d[d[self.consequence_col].astype(str).str.contains("missense", na=False)]
        return d
 
    def _key(self, f, t):
        return f"{f}{self.key_sep}{t}"
 
    # ---- sklearn API -------------------------------------------------------
    def fit(self, X, y=None):
        train_ids = set(_as_region_ids(X))
 
        # 1) slice variant table to TRAINING regions only
        df_train = self._variants_for(train_ids)
 
        # 2) slice region_by_id (the expected/opportunity null) to TRAINING regions
        rbi_train = {rid: self.region_by_id[rid]
                     for rid in train_ids if rid in self.region_by_id}
 
        # 3) load rates once (cached on the instance)
        if not hasattr(self, "rates_"):
            self.rates_ = load_mutation_rates(self.rates_path) if self.rates_path else load_mutation_rates()
 
        # 4) counting core -> tidy table -> score table  (training data only)
        result = compute_composition_normalized_enrichment(
            df_train, rbi_train, min_total=self.min_total, rates=self.rates_,
            group_col=self.group_col)
        table = result_to_table(result)
 
        if table is None or table.empty:
            # degenerate fold: no scorable substitutions
            self.lookup_ = {}
            self.global_ = 0.0
            self.table_ = table
            return self
 
        self.table_, self.lookup_ = build_score_table(
            table, alpha=self.alpha, min_count=self.min_count, key_sep=self.key_sep)
        # fallback for substitutions unseen in this fold: neutral (no preference)
        self.global_ = 0.0
        return self
 
    def _region_score(self, variants):
        """Average fitted score over one region's variants. Optionally split by source AA."""
        if len(variants) == 0:
            if self.split_source:
                d = {f"sub_score_mean_{r}": np.nan for r in self.source_residues}
                d["sub_score_mean_other"] = np.nan
                return d
            return {"sub_score_mean": np.nan}
 
        scores_all, by_src = [], {r: [] for r in self.source_residues}
        other = []
        for f, t in zip(variants[self.from_col], variants[self.to_col]):
            s = self.lookup_.get(self._key(f, t), self.global_)
            scores_all.append(s)
            if f in by_src:
                by_src[f].append(s)
            else:
                other.append(s)
 
        if not self.split_source:
            return {"sub_score_mean": float(np.mean(scores_all)) if scores_all else np.nan}
 
        out = {}
        for r in self.source_residues:
            out[f"sub_score_mean_{r}"] = float(np.mean(by_src[r])) if by_src[r] else np.nan
        out["sub_score_mean_other"] = float(np.mean(other)) if other else np.nan
        return out
 
    def transform(self, X):
        region_ids = _as_region_ids(X)
        # a region's OWN variants are not leakage; look up from the full table
        rows = {}
        for rid in region_ids:
            v = self._variants_for([rid])
            rows[rid] = self._region_score(v)
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = self.region_col
        return out
 
    def get_feature_names_out(self, input_features=None):
        if self.split_source:
            return np.array([f"sub_score_mean_{r}" for r in self.source_residues] +
                            ["sub_score_mean_other"])
        return np.array(["sub_score_mean"])
 
