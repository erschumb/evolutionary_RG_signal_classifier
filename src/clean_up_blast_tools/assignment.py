# src/blast/assignment.py

import pandas as pd
import numpy as np


def assign_shared_accessions(df: pd.DataFrame,
                             random_seed: int = 42):

    rng = np.random.default_rng(random_seed)

    acc_uid = (
        df.groupby("hit_accession")["UniqueID"]
        .nunique()
    )

    shared = acc_uid[acc_uid > 1].index

    if len(shared) == 0:
        return df.copy(), {"n_shared_accessions": 0}

    shared_df = df[df["hit_accession"].isin(shared)]

    mean_score = (
        shared_df.groupby(["hit_accession", "UniqueID"])["bit_score"]
        .mean()
        .reset_index()
    )

    assigned = {}
    n_random = 0

    for acc, grp in mean_score.groupby("hit_accession"):

        max_score = grp["bit_score"].max()
        candidates = grp[grp["bit_score"] == max_score]["UniqueID"].tolist()

        if len(candidates) == 1:
            assigned[acc] = candidates[0]
        else:
            assigned[acc] = rng.choice(candidates)
            n_random += 1

    df_shared = df[df["hit_accession"].isin(shared)].copy()
    df_shared["_assigned"] = df_shared["hit_accession"].map(assigned)

    df_shared = df_shared[df_shared["UniqueID"] == df_shared["_assigned"]]
    df_shared = df_shared.drop(columns="_assigned")

    df_unshared = df[~df["hit_accession"].isin(shared)]

    df_out = pd.concat([df_unshared, df_shared], ignore_index=True)

    report = {
        "n_shared_accessions": len(shared),
        "n_random_assignments": n_random,
        "rows_removed": len(df) - len(df_out),
    }

    return df_out, report