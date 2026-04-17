# src/blast/sequence_extraction.py
"""
Extract aligned query/hit subsequences corresponding to a defined window
in query protein coordinates.
"""
import numpy as np
import pandas as pd


def extract_rg_region_seqs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract qseq and hseq for the RG window region into two new columns:
      - qseq_rg_region
      - hseq_rg_region

    For rows where rg_window_coverage != 'full', both columns are NaN.

    Window coordinates are interpreted in query protein space (0-based,
    half-open): gaps in the query consume no protein position; gaps in the
    hit are retained within the window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: rg_window_coverage, qseq, hseq, win_start_x,
        win_end_x, query_from.

    Returns
    -------
    pd.DataFrame
        Copy of the input with qseq_rg_region and hseq_rg_region columns added.
    """
    df = df.copy()
    qseq_rg = np.full(len(df), np.nan, dtype=object)
    hseq_rg = np.full(len(df), np.nan, dtype=object)

    full_mask = (df["rg_window_coverage"] == "full").to_numpy()
    idx_full = np.where(full_mask)[0]

    qseqs      = df["qseq"].to_numpy()
    hseqs      = df["hseq"].to_numpy()
    win_starts = df["win_start_x"].to_numpy()
    win_ends   = df["win_end_x"].to_numpy()
    aln_starts = df["query_from"].astype(int).to_numpy() - 1  # 0-based

    for i in idx_full:
        qseq = qseqs[i]
        hseq = hseqs[i]
        win_start = win_starts[i]
        win_end   = win_ends[i]
        protein_pos = aln_starts[i]

        q_window = []
        h_window = []
        for q_char, h_char in zip(qseq, hseq):
            if protein_pos >= win_end:
                break
            if q_char != "-":
                if protein_pos >= win_start:
                    q_window.append(q_char)
                    h_window.append(h_char)
                protein_pos += 1

        qseq_rg[i] = "".join(q_window)
        hseq_rg[i] = "".join(h_window)

    df["qseq_rg_region"] = qseq_rg
    df["hseq_rg_region"] = hseq_rg
    return df