# src/blast/assignment.py
import pandas as pd
import numpy as np


def assign_shared_accessions(df: pd.DataFrame, random_seed: int = 42):
    """
    For each hit_accession that appears under more than one UniqueID, assign it
    to the UniqueID with the highest mean bit_score for that accession. If two
    or more UniqueIDs tie on mean bit_score, one is chosen at random.

    Accessions appearing under multiple orig_motif_index values of the SAME
    UniqueID are always kept (not considered redundant).

    Parameters
    ----------
    df          : dataframe after compute_rg_window_columns(), filtered to
                  whatever coverage/gap step you want to apply this to
    random_seed : for reproducibility of random tie-breaking

    Returns
    -------
    df_out      : filtered dataframe where each hit_accession appears under
                  at most one UniqueID (but may appear under multiple motifs
                  within that UniqueID)
    report      : dict with summary statistics
    """
    rng = np.random.default_rng(random_seed)

    # --- step 1: identify accessions shared across multiple UniqueIDs ---
    acc_uid = df.groupby("hit_accession")["UniqueID"].nunique()
    shared = acc_uid.index[acc_uid > 1]
    n_total = len(acc_uid)
    n_shared = len(shared)

    if n_shared == 0:
        return df.copy(), {
            "n_total_accessions":    n_total,
            "n_shared_accessions":   0,
            "n_randomly_assigned":   0,
            "pct_randomly_assigned": 0.0,
            "n_rows_before":         len(df),
            "n_rows_after":          len(df),
            "n_rows_removed":        0,
        }

    # --- step 2: mean bit_score per (accession, UniqueID) for shared accessions ---
    is_shared = df["hit_accession"].isin(shared)
    mean_score = (
        df[is_shared]
        .groupby(["hit_accession", "UniqueID"], as_index=False)["bit_score"]
        .mean()
    )

    # --- step 3: count how many accessions had a tie at the max (for reporting) ---
    max_per_acc = mean_score.groupby("hit_accession")["bit_score"].transform("max")
    at_max = mean_score["bit_score"] == max_per_acc
    n_random = int((at_max.groupby(mean_score["hit_accession"]).sum() > 1).sum())

    # --- step 4: vectorised assignment with random tie-breaking ---
    # add a random column, then sort by (accession asc, score desc, tiebreak desc)
    # and take the first row per accession — ties are broken uniformly at random
    mean_score["_tiebreak"] = rng.random(len(mean_score))
    mean_score = mean_score.sort_values(
        ["hit_accession", "bit_score", "_tiebreak"],
        ascending=[True, False, False],
    )
    assigned = (
        mean_score.drop_duplicates("hit_accession")
        .set_index("hit_accession")["UniqueID"]
    )

    # --- step 5: build the filtered dataframe ---
    df_shared = df[is_shared].copy()
    df_shared = df_shared[df_shared["UniqueID"] == df_shared["hit_accession"].map(assigned)]
    df_out = pd.concat([df[~is_shared], df_shared], ignore_index=True)

    # --- step 6: compile report ---
    report = {
        "n_total_accessions":    n_total,
        "n_shared_accessions":   n_shared,
        "n_randomly_assigned":   n_random,
        "pct_randomly_assigned": round(n_random / n_shared * 100, 2),
        "n_rows_before":         len(df),
        "n_rows_after":          len(df_out),
        "n_rows_removed":        len(df) - len(df_out),
    }
    return df_out, report