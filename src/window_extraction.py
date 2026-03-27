# window_extraction.py

import pandas as pd

# --- your existing helpers (unchanged) ---

def merge_motif_idr(df_motif, df_idr):
    return df_motif.merge(
        df_idr,
        on="UniqueID",
        how="left",
        validate="many_to_one"
    )

def idr_fraction_interval(idr_array, start, end):
    try:
        window = idr_array[start:end]
    except Exception:
        return None
    if len(window) == 0:
        return None
    return sum(window) / len(window)

def effective_flank_limits(row, flank):
    motif_start = row["start"]
    motif_end = row["end"]
    prot_len = len(row["full_seq"])

    max_left = motif_start
    max_right = prot_len - motif_end

    return min(flank, max_left), min(flank, max_right)


def adaptive_flank_trimming(row,
                             flank=20,
                             min_idr_fraction=0.7,
                             min_flank_len=5,
                             trim_step=1):

    seq = row["full_seq"]
    idr = row["prediction-disorder-mobidb_lite"]

    motif_start = row["start"]
    motif_end = row["end"]

    left_flank, right_flank = effective_flank_limits(row, flank)

    # --- left ---
    while left_flank >= min_flank_len:
        frac = idr_fraction_interval(idr, motif_start - left_flank, motif_start)
        if frac is not None and frac >= min_idr_fraction:
            break
        left_flank -= trim_step

    # --- right ---
    while right_flank >= min_flank_len:
        frac = idr_fraction_interval(idr, motif_end, motif_end + right_flank)
        if frac is not None and frac >= min_idr_fraction:
            break
        right_flank -= trim_step

    return left_flank, right_flank


def compute_windows(df,
                    flank=20,
                    mode="adaptive",
                    min_idr_fraction=0.7,
                    min_flank_len=5,
                    trim_step=1):

    records = []

    for _, row in df.iterrows():

        seq = row["full_seq"]
        motif_start = row["start"]
        motif_end = row["end"]
        prot_len = len(seq)

        if mode == "adaptive":
            lf, rf = adaptive_flank_trimming(
                row,
                flank,
                min_idr_fraction,
                min_flank_len,
                trim_step
            )
        elif mode == "control":
            lf = min(flank, motif_start)
            rf = min(flank, prot_len - motif_end)
        elif mode == "full":
            lf, rf = adaptive_flank_trimming(
                row,
                flank,
                min_idr_fraction,
                min_flank_len,
                trim_step
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        win_start = max(0, motif_start - lf)
        win_end = min(prot_len, motif_end + rf)

        region_seq = seq[win_start:win_end]

        if mode == "full":
            region_seq = seq
            win_start, win_end = 0, len(seq)

        records.append({
            "UniqueID": row["UniqueID"],
            "orig_motif_index": row["orig_motif_index"],
            "mode": mode,
            "motif_start": motif_start,
            "motif_end": motif_end,
            "win_start": win_start,
            "win_end": win_end,
            "left_flank": lf,
            "right_flank": rf,
            "window_length": win_end - win_start,
            "motif_length": motif_end - motif_start,
            "full_seq": seq,
            "region_seq": region_seq
        })

    return pd.DataFrame(records)


def prepare_window_dataframe(df_motif,
                             df_idr,
                             flank=20,
                             mode="adaptive",
                             min_idr_fraction=0.7,
                             min_flank_len=5,
                             trim_step=1):

    df = merge_motif_idr(df_motif, df_idr)

    return compute_windows(
        df,
        flank=flank,
        mode=mode,
        min_idr_fraction=min_idr_fraction,
        min_flank_len=min_flank_len,
        trim_step=trim_step
    )