"""
Save as: src/analysis_visualization/homolog_region_classification.py

Step 1: Fetch MobiDB-lite disorder annotations per protein and classify every
protein residue into one of:
  - RG_motif    : inside a known RG motif from the input data
  - mIDR        : in an IDR that contains (overlaps) at least one RG motif
  - oIDR        : in an IDR with no RG-motif overlap
  - structured  : not in any predicted IDR

Merges consecutive same-class residues into contiguous regions, filters out
regions shorter than min_length.
"""

from __future__ import annotations
import time
import json
import requests
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# MobiDB REST fetching
# ════════════════════════════════════════════════════════════════════════════

MOBIDB_API = "https://mobidb.bio.unipd.it/api/download"
MOBIDB_LITE_KEY = "prediction-disorder-mobidb_lite"


def fetch_mobidb_disorder(
    uniprot_id: str,
    timeout: int = 15,
    retry: int = 2,
) -> Optional[Dict]:
    """
    Query the MobiDB REST API for a single UniProt accession.
    Returns the raw JSON dict, or None on failure.
    """
    params = {"acc": uniprot_id, "format": "json"}
    for attempt in range(retry + 1):
        try:
            r = requests.get(MOBIDB_API, params=params, timeout=timeout)
            if r.status_code == 200:
                text = r.text.strip()
                if not text:
                    return None
                # MobiDB returns JSONL (one JSON per line). Take the first.
                first_line = text.split("\n", 1)[0]
                return json.loads(first_line)
            if r.status_code == 404:
                return None
        except (requests.RequestException, json.JSONDecodeError):
            if attempt < retry:
                time.sleep(1.0)
    return None


def extract_mobidb_lite_regions(
    mobidb_json: Dict,
) -> Optional[List[Tuple[int, int]]]:
    """
    Extract MobiDB-lite disorder regions from the MobiDB JSON record.
    Returns list of (start, end) tuples, 1-indexed inclusive (UniProt convention).
    Returns None if record lacks MobiDB-lite predictions.
    """
    if mobidb_json is None:
        return None
    pred = mobidb_json.get(MOBIDB_LITE_KEY, {})
    regions = pred.get("regions", [])
    if regions is None:
        return None
    return [(int(r[0]), int(r[1])) for r in regions]


def fetch_all_mobidb_disorder(
    uniprot_ids: List[str],
    verbose: bool = True,
    sleep_between: float = 0.15,
) -> Dict[str, Optional[List[Tuple[int, int]]]]:
    """
    Fetch MobiDB-lite for every accession. Returns dict keyed by UniProt ID
    with list of (start, end) disorder regions, or None on failure.
    """
    out = {}
    failed = []
    n = len(uniprot_ids)
    for i, uid in enumerate(uniprot_ids, 1):
        raw = fetch_mobidb_disorder(uid)
        regions = extract_mobidb_lite_regions(raw) if raw is not None else None
        out[uid] = regions
        if regions is None:
            failed.append(uid)
        if verbose and (i % 25 == 0 or i == n):
            print(f"  MobiDB fetch: {i}/{n}  (failed so far: {len(failed)})")
        time.sleep(sleep_between)
    if verbose:
        print(f"  Done. {n - len(failed)}/{n} succeeded.")
    return out


# ════════════════════════════════════════════════════════════════════════════
# Region classification per protein
# ════════════════════════════════════════════════════════════════════════════

def _build_residue_classification(
    protein_length: int,
    disorder_regions: List[Tuple[int, int]],
    rg_motifs: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Assign every residue a class string. Uses 1-indexed coordinates (UniProt).
    Returns a numpy array of length protein_length with dtype=object.

    Classification precedence:
      RG_motif    : any residue inside any RG motif
      mIDR        : IDR residue, and the IDR overlaps ≥1 RG motif (but not the motif itself)
      oIDR        : IDR residue, and the IDR has no RG motif overlap
      structured  : not in any disorder region
    """
    classification = np.array(["structured"] * protein_length, dtype=object)

    # Mark IDR residues first (tentative as oIDR or mIDR — we'll resolve below)
    for start, end in disorder_regions:
        s = max(1, start) - 1       # convert to 0-indexed
        e = min(protein_length, end) # end inclusive in 1-indexed = exclusive in 0-indexed
        if e < s:
            continue
        classification[s:e] = "oIDR"

    # For each IDR, check if it overlaps any RG motif; if yes, tag as mIDR
    for start, end in disorder_regions:
        s = max(1, start) - 1
        e = min(protein_length, end)
        overlaps_rg = False
        for m_start, m_end in rg_motifs:
            ms = max(1, m_start) - 1
            me = min(protein_length, m_end)
            # Overlap test (inclusive/exclusive math)
            if ms < e and me > s:
                overlaps_rg = True
                break
        if overlaps_rg:
            classification[s:e] = "mIDR"

    # Finally, overwrite RG-motif residues themselves as RG_motif
    for m_start, m_end in rg_motifs:
        ms = max(1, m_start) - 1
        me = min(protein_length, m_end)
        if me < ms:
            continue
        classification[ms:me] = "RG_motif"

    return classification


def _merge_contiguous(
    classification: np.ndarray,
    min_length: int,
) -> List[Tuple[int, int, str]]:
    """
    Merge consecutive same-class residues into contiguous regions.
    Returns list of (start, end, class) tuples, 1-indexed inclusive.
    Filters out regions shorter than min_length.
    """
    regions = []
    if len(classification) == 0:
        return regions
    start = 0
    current_cls = classification[0]
    for i in range(1, len(classification)):
        if classification[i] != current_cls:
            length = i - start
            if length >= min_length:
                regions.append((start + 1, i, current_cls))  # 1-indexed inclusive
            start = i
            current_cls = classification[i]
    # Close final run
    length = len(classification) - start
    if length >= min_length:
        regions.append((start + 1, len(classification), current_cls))
    return regions


def classify_protein_regions(
    uniprot_id: str,
    protein_length: int,
    disorder_regions: List[Tuple[int, int]],
    rg_motifs: List[Tuple[int, int]],
    min_length: int = 20,
) -> pd.DataFrame:
    """
    [Dataset-agnostic]
    Produce a per-region classification DataFrame for one protein.

    Columns: UniqueID, region_start, region_end, region_length, region_class
    """
    classification = _build_residue_classification(
        protein_length, disorder_regions, rg_motifs,
    )
    merged = _merge_contiguous(classification, min_length=min_length)
    df = pd.DataFrame(merged, columns=["region_start", "region_end", "region_class"])
    df["UniqueID"] = uniprot_id
    df["region_length"] = df["region_end"] - df["region_start"] + 1
    return df[["UniqueID", "region_start", "region_end", "region_length", "region_class"]]


# ════════════════════════════════════════════════════════════════════════════
# Build RG motif list per protein from homolog dataframe
# ════════════════════════════════════════════════════════════════════════════

def extract_rg_motifs_per_protein(
    df_combined: pd.DataFrame,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    For each protein (UniqueID), collect all RG motif coordinates as
    list of (motif_start, motif_end) tuples. Deduplicates by motif position.
    """
    out = {}
    for uid, sub in df_combined.groupby("UniqueID"):
        # Dedupe on (motif_start, motif_end) — same motif can appear in many hits
        motifs = sub[["motif_start", "motif_end"]].drop_duplicates()
        out[uid] = [
            (int(row.motif_start), int(row.motif_end))
            for row in motifs.itertuples(index=False)
        ]
    return out


def extract_protein_lengths(df_combined: pd.DataFrame) -> Dict[str, int]:
    """
    Extract protein length per UniqueID. Uses query_len column.
    """
    return (
        df_combined.groupby("UniqueID")["query_len"]
        .first()
        .astype(int)
        .to_dict()
    )


# ════════════════════════════════════════════════════════════════════════════
# End-to-end convenience function
# ════════════════════════════════════════════════════════════════════════════

def build_region_classification_table(
    df_combined: pd.DataFrame,
    min_length: int = 20,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    End-to-end pipeline:
      1. Extract unique UniProt IDs from df_combined
      2. Fetch MobiDB-lite disorder regions (or load from cache)
      3. For each protein, classify residues and merge into regions
      4. Concatenate into one DataFrame

    cache_path: if provided, saves/loads the fetched MobiDB dict to/from JSON.
    """
    uniprot_ids = sorted(df_combined["UniqueID"].unique().tolist())
    if verbose:
        print(f"Unique proteins: {len(uniprot_ids)}")

    # Fetch or load disorder annotations
    mobidb_data = None
    if cache_path is not None:
        try:
            with open(cache_path, "r") as f:
                mobidb_data = json.load(f)
            if verbose:
                print(f"Loaded MobiDB cache from {cache_path}")
        except FileNotFoundError:
            mobidb_data = None

    if mobidb_data is None:
        if verbose:
            print("Fetching MobiDB-lite annotations...")
        mobidb_data = fetch_all_mobidb_disorder(uniprot_ids, verbose=verbose)
        if cache_path is not None:
            with open(cache_path, "w") as f:
                json.dump(mobidb_data, f)
            if verbose:
                print(f"Saved MobiDB cache to {cache_path}")

    # Normalize keys (cache json turns tuple keys into strings)
    mobidb_clean = {}
    for uid, regs in mobidb_data.items():
        if regs is None:
            mobidb_clean[uid] = None
        else:
            mobidb_clean[uid] = [tuple(r) for r in regs]

    rg_motifs_by_protein = extract_rg_motifs_per_protein(df_combined)
    protein_lengths = extract_protein_lengths(df_combined)

    # Classify each protein
    all_regions = []
    proteins_no_mobidb = 0
    for uid in uniprot_ids:
        if mobidb_clean.get(uid) is None:
            proteins_no_mobidb += 1
            continue
        regs = classify_protein_regions(
            uniprot_id=uid,
            protein_length=protein_lengths[uid],
            disorder_regions=mobidb_clean[uid],
            rg_motifs=rg_motifs_by_protein.get(uid, []),
            min_length=min_length,
        )
        all_regions.append(regs)

    if not all_regions:
        raise ValueError("No proteins classified. MobiDB fetch may have failed.")

    classification_df = pd.concat(all_regions, ignore_index=True)

    if verbose:
        print(f"\nClassification summary:")
        print(f"  Proteins with MobiDB annotation: "
              f"{len(uniprot_ids) - proteins_no_mobidb}/{len(uniprot_ids)}")
        print(f"  Proteins with no MobiDB data: {proteins_no_mobidb}")
        print(f"  Total regions (≥{min_length} aa): {len(classification_df):,}")
        print(f"\n  Region count by class:")
        print(classification_df["region_class"].value_counts().to_string())
        print(f"\n  Total residues by class:")
        print(classification_df.groupby("region_class")["region_length"].sum().to_string())

    return classification_df, mobidb_clean