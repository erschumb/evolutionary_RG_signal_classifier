"""
assemble_features.py  —  thin orchestrator showing how the two registries compose.

Fold-static features are computed once and passed as a ready matrix.
Fold-derived features live in a Pipeline (refit per fold) BEFORE the classifier.
The label y is kept SEPARATE, keyed on region_id, and never enters either
feature frame.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from features_static import build_static_features, JOIN_KEY
from features_folded import SubstitutionScoreTransformer


def make_label_vector(df_rg: pd.DataFrame) -> pd.Series:
    """y keyed on region_id, taken from the variant table's group column."""
    y = (df_rg.groupby(JOIN_KEY)["group"].first()
              .map({"neg": 0, "pos": 1}))   # adjust mapping to your label values
    y.name = "label"
    return y


def build_pipeline(alpha=0.5, split_source=False, **rf_kwargs) -> Pipeline:
    """
    Fold-derived transformer(s) -> classifier.
    Fold-static features are merged in separately (computed once); they do not
    need to live in the pipeline. param grid example: {'subscore__alpha': [...]}.
    """
    return Pipeline([
        ("subscore", SubstitutionScoreTransformer(alpha=alpha,
                                                  split_source=split_source)),
        ("rf", RandomForestClassifier(**rf_kwargs)),
    ])


# Sketch of intended use (not executed here):
#   X_static = build_static_features(df_rg)          # once, full data, safe
#   y        = make_label_vector(df_rg)              # separate, keyed on region_id
#   # within CV: pipeline refits fold-derived features per training fold,
#   # X_static is sliced by the same region_id index and concatenated.