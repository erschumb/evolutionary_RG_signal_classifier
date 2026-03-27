# src/blast/coverage.py

import pandas as pd
import numpy as np


def compute_rg_window_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate BLAST alignments with:
      - rg_window_coverage: full / partial / none
      - rg_window_gap_fraction_bin: quartile bin
    """

    df = df.copy()

    coverage = []
    gap_bins = []

    for _, row in df.iterrows():

        win_start = row["win_start_x"]
        win_end   = row["win_end_x"]

        aln_start = int(row["query_from"]) - 1
        aln_end   = int(row["query_to"])

        overlap_start = max(win_start, aln_start)
        overlap_end   = min(win_end, aln_end)
        overlap_len   = overlap_end - overlap_start

        if overlap_len <= 0:
            coverage.append("none")
            gap_bins.append(np.nan)
            continue

        if overlap_start <= win_start and overlap_end >= win_end:
            cov = "full"
        else:
            cov = "partial"

        coverage.append(cov)

        # --- map alignment columns ---
        qseq = row["qseq"]
        hseq = row["hseq"]

        protein_pos = aln_start
        hseq_window = []

        for q_char, h_char in zip(qseq, hseq):

            if protein_pos >= overlap_end:
                break

            if q_char != "-":
                if protein_pos >= overlap_start:
                    hseq_window.append(h_char)
                protein_pos += 1

        if not hseq_window:
            gap_bins.append(np.nan)
            continue

        gap_fraction = sum(c == "-" for c in hseq_window) / len(hseq_window)

        if gap_fraction <= 0.25:
            gap_bins.append("0-25%")
        elif gap_fraction <= 0.50:
            gap_bins.append("25-50%")
        elif gap_fraction <= 0.75:
            gap_bins.append("50-75%")
        else:
            gap_bins.append("75-100%")

    df["rg_window_coverage"] = coverage

    df["rg_window_gap_fraction_bin"] = pd.Categorical(
        gap_bins,
        categories=["0-25%", "25-50%", "50-75%", "75-100%"],
        ordered=True
    )

    return df