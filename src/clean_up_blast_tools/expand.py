# src/blast/expand.py

import pandas as pd


def expand_hits_to_motifs(df_blast: pd.DataFrame,
                         df_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Expand protein-level BLAST hits to motif-level rows.

    Removes motif coordinates from BLAST header and replaces them
    with motif-resolved coordinates from df_windows.
    """

    df_blast = df_blast.copy()

    # Extract UniqueID from FASTA header
    df_blast["UniqueID"] = df_blast["query_title"].str.split("_").str[0]

    # Drop misleading motif coords from BLAST header
    df_blast = df_blast.drop(
        columns=["motif_start", "motif_end"],
        errors="ignore"
    )

    df_expanded = df_windows.merge(
        df_blast,
        on="UniqueID",
        how="left"
    )

    return df_expanded