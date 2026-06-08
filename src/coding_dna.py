#!/usr/bin/env python3
"""
coding_dna_v2.py

Map UniProt protein regions -> genomic coordinates using a prebuilt MANE
Select index + local GRCh38 FASTA.

Coordinate convention for INPUT (start, end):
    0-based half-open (i.e. Python-slice).
    AA 0..3 means residues 0, 1, 2 -> 3 residues total, 9 nt.
    Matches your df where full_seq[start:end] gives the region.

Output 'intervals' use 1-based inclusive coords (gnomAD/bcftools convention).
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pysam
from Bio.Seq import Seq

# Import dataclasses from build script
from src.build_mane_index import CDSExon, TranscriptCDS  # noqa: F401


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
REFS_DIR        = Path("/mnt/d/phd/scripts/16_ev_signature_predictor/data/refs")
DEFAULT_INDEX   = REFS_DIR / "mane_cds_index.pkl"
DEFAULT_FASTA   = REFS_DIR / "GRCh38.primary_assembly.genome.fa"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("coding_dna_v2")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)


# ---------------------------------------------------------------------------
# Load index + FASTA
# ---------------------------------------------------------------------------
def load_index(path: Path = DEFAULT_INDEX) -> Dict[str, TranscriptCDS]:
    with open(path, "rb") as fh:
        index = pickle.load(fh)
    logger.info(f"Loaded index: {len(index)} UniProt accessions")
    return index


def open_fasta(path: Path = DEFAULT_FASTA) -> pysam.FastaFile:
    fasta = pysam.FastaFile(str(path))
    logger.info(f"Opened FASTA: {path.name} ({len(fasta.references)} contigs)")
    return fasta


def sanity_check_chromosomes(index: Dict[str, TranscriptCDS],
                             fasta: pysam.FastaFile) -> None:
    """Make sure MANE chromosomes exist in the FASTA."""
    PRIMARY = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
    mane_chroms = {tx.chrom for tx in index.values()}
    fasta_refs = set(fasta.references)
    missing = mane_chroms - fasta_refs

    missing_primary = missing & PRIMARY
    missing_alt = missing - PRIMARY

    if missing_primary:
        raise RuntimeError(
            f"Primary chromosomes missing from FASTA: {sorted(missing_primary)}. "
            f"Check chr-prefix conventions."
        )
    if missing_alt:
        # Count how many transcripts live on these contigs
        alt_txs = [upr for upr, tx in index.items() if tx.chrom in missing_alt]
        logger.warning(
            f"{len(missing_alt)} alt/fix contigs not in FASTA "
            f"(affects {len(alt_txs)} transcripts: {alt_txs[:5]}...). "
            f"These will fail with a clear error if queried."
        )
    logger.info(
        f"Chromosome naming check: {len(mane_chroms & fasta_refs)} "
        f"of {len(mane_chroms)} MANE chroms present in FASTA"
    )

# ---------------------------------------------------------------------------
# CDS fetch + AA->interval math
# ---------------------------------------------------------------------------

def _translate_full_cds(tx, fasta, cds_cache=None):
    """Translate the entire CDS into protein (without trailing stop).
    Returns None if the CDS could not be fetched (e.g. contig not in FASTA)."""
    if cds_cache is not None and tx.transcript_id in cds_cache:
        cds_concat = cds_cache[tx.transcript_id]
    else:
        cds_concat = _fetch_cds_concat(tx, fasta)
        if cds_cache is not None:
            cds_cache[tx.transcript_id] = cds_concat

    # _fetch_cds_concat returns (None, reason) on failure (e.g. alt/fix contig)
    if not isinstance(cds_concat, str):
        return None

    prot = str(Seq(cds_concat).translate())   # cds_concat is now guaranteed str
    return prot.rstrip("*")

def _fetch_cds_concat(tx: TranscriptCDS, fasta: pysam.FastaFile) -> str:
    """Concatenate CDS exons in transcription order."""
    parts = []
    for ex in tx.exons:
        # pysam uses 0-based half-open; GFF is 1-based inclusive
        if ex.chrom not in fasta.references:          # contig not in primary assembly
            return None, f"contig_not_in_fasta:{ex.chrom}"
        seq = fasta.fetch(ex.chrom, ex.g_start - 1, ex.g_end).upper()
        if tx.strand == "-":
            seq = str(Seq(seq).reverse_complement())
        parts.append(seq)
    return "".join(parts)


def _aa_to_intervals(
    tx: TranscriptCDS, aa_start: int, aa_end: int
) -> List[Dict]:
    """
    Map an AA window (0-based half-open) to genomic intervals,
    returned in transcription order.
    """
    nt_start = aa_start * 3       # 0-based inclusive in concatenated CDS
    nt_end   = aa_end * 3 - 1     # 0-based inclusive (last nt of window)

    intervals = []
    cds_cursor = 0
    merged_cursor = 0

    for ex in tx.exons:
        ex_len = ex.g_end - ex.g_start + 1
        ex_cds_start = cds_cursor
        ex_cds_end = cds_cursor + ex_len - 1
        cds_cursor += ex_len

        ov_s = max(nt_start, ex_cds_start)
        ov_e = min(nt_end, ex_cds_end)
        if ov_s > ov_e:
            continue

        local_s = ov_s - ex_cds_start   # offset into exon, 5'->3'
        local_e = ov_e - ex_cds_start
        length = local_e - local_s + 1

        if tx.strand == "+":
            g_s = ex.g_start + local_s
            g_e = ex.g_start + local_e
        else:
            g_e = ex.g_end - local_s
            g_s = ex.g_end - local_e

        intervals.append({
            "chrom": ex.chrom,
            "start": min(g_s, g_e),   # 1-based inclusive
            "end":   max(g_s, g_e),
            "strand": tx.strand,
            "merged_start": merged_cursor,
            "merged_end": merged_cursor + length - 1,
        })
        merged_cursor += length

    return intervals


# ---------------------------------------------------------------------------
# Per-region API (with optional per-transcript CDS cache)
# ---------------------------------------------------------------------------

def get_exact_dna(
    protein_id: str,
    aa_start: int,
    aa_end: int,
    cds_index: Dict[str, TranscriptCDS],
    fasta: pysam.FastaFile,
    prot_seq: Optional[str] = None,
    cds_cache: Optional[Dict[str, str]] = None,
    realign_to_mane: bool = True,
    region_seq: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Map a UniProt protein region to genomic coordinates.

    If the user-provided full_seq disagrees with the MANE-encoded protein
    (different isoform), and realign_to_mane=True, the function searches
    for region_seq within the MANE protein and uses those coordinates
    instead. The 'warning' field will be 'realigned' in that case.

    Coordinates: aa_start/aa_end are 0-based half-open.
    """
    tx = cds_index.get(protein_id)
    if tx is None:
        return None, "no_mane_transcript_for_uniprot"

    # Derive region_seq from prot_seq if not given
    if region_seq is None and prot_seq is not None:
        region_seq = prot_seq[aa_start:aa_end]

    # MANE protein length (CDS includes stop codon)
    mane_n_aa = tx.cds_length // 3 - 1

    def _do_extraction(s: int, e: int, expected: Optional[str]):
        """Inner helper that does the actual extraction at given AA coords."""
        if s < 0 or e > mane_n_aa or s >= e:
            return None, f"aa_range_out_of_bounds (MANE protein length {mane_n_aa})"

        if cds_cache is not None and tx.transcript_id in cds_cache:
            cds_concat = cds_cache[tx.transcript_id]
        else:
            cds_concat = _fetch_cds_concat(tx, fasta)
            if cds_cache is not None:
                cds_cache[tx.transcript_id] = cds_concat

        nt_s, nt_e = s * 3, e * 3
        dna = cds_concat[nt_s:nt_e]

        expected_nt = (e - s) * 3
        if len(dna) != expected_nt:
            return None, f"dna_length_mismatch (expected {expected_nt}, got {len(dna)})"

        intervals = _aa_to_intervals(tx, s, e)
        trans = str(Seq(dna).translate())

        warning = None
        if expected is not None and trans != expected:
            warning = "translation_mismatch"

        return {
            "intervals": intervals,
            "dna": dna,
            "prot_seq": trans,
            "warning": warning,
        }, None

    # First attempt: use the coordinates as given
    out, err = _do_extraction(aa_start, aa_end, region_seq)

    needs_realign = (
        realign_to_mane
        and region_seq is not None
        and (err is not None or (out and out["warning"] == "translation_mismatch"))
    )

    realigned_from = None
    if needs_realign:
        mane_prot = _translate_full_cds(tx, fasta, cds_cache)
        if mane_prot is None:
            # CDS unfetchable (alt/fix contig) — can't realign; treat as failure
            return None, "realign_cds_unfetchable"
        positions = []
        idx = mane_prot.find(region_seq)
        while idx != -1:
            positions.append(idx)
            idx = mane_prot.find(region_seq, idx + 1)

        if len(positions) == 1:
            new_start = positions[0]
            new_end = new_start + len(region_seq)
            realigned_from = (aa_start, aa_end)
            out, err = _do_extraction(new_start, new_end, region_seq)
            if out is not None:
                out["warning"] = "realigned"
                aa_start, aa_end = new_start, new_end
        elif len(positions) == 0:
            return None, "realign_not_found"
        else:
            return None, f"realign_ambiguous (found {len(positions)} matches)"

    if err is not None:
        return None, err

    result = {
        "protein": protein_id,
        "prot_region": (aa_start, aa_end),
        "prot_seq": out["prot_seq"],
        "intervals": out["intervals"],
        "dna": out["dna"],
    }
    if out["warning"]:
        result["warning"] = out["warning"]
    if realigned_from is not None:
        result["original_region"] = realigned_from
    return result, out["warning"]

# ---------------------------------------------------------------------------
# Batch wrapper
# ---------------------------------------------------------------------------

def process_multiple_regions(
    region_list: List[Tuple],
    cds_index: Dict[str, TranscriptCDS],
    fasta: pysam.FastaFile,
    realign_to_mane: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    region_list entries: (uniprot_acc, aa_start, aa_end, full_prot_seq[, region_seq])
    """
    cds_cache: Dict[str, str] = {}
    results, failed = [], []

    for entry in region_list:
        region_seq = None
        if len(entry) == 5:
            protein_id, s, e, prot_seq, region_seq = entry
        elif len(entry) == 4:
            protein_id, s, e, prot_seq = entry
        else:
            protein_id, s, e = entry
            prot_seq = None

        res, reason = get_exact_dna(
            protein_id, s, e, cds_index, fasta, prot_seq, cds_cache,
            realign_to_mane=realign_to_mane, region_seq=region_seq,
        )
        if res is None:
            failed.append({"protein": protein_id, "region": (s, e), "reason": reason})
        else:
            results.append(res)

    n_realigned = sum(1 for r in results if r.get("warning") == "realigned")
    logger.info(
        f"Processed {len(region_list)} regions: "
        f"{len(results)} ok ({n_realigned} realigned), {len(failed)} failed"
    )
    return results, failed

# ---------------------------------------------------------------------------
# DataFrame helper
# ---------------------------------------------------------------------------
def df_to_regions(df, uid_col="UniqueID", start_col="win_start",
                  end_col="win_end", seq_col="full_seq",
                  region_seq_col="region_seq") -> List[Tuple]:
    """Convert your motif df into the tuple list expected by process_multiple_regions."""
    return [
        (row[uid_col], int(row[start_col]), int(row[end_col]),
         row[seq_col], row[region_seq_col])
        for _, row in df.iterrows()
    ]