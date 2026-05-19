#!/usr/bin/env python3
"""
build_mane_index.py

One-time script: parse MANE Select GFF3 + UniProt ID-mapping into a pickled
{uniprot_acc: TranscriptCDS} index.

Usage:
    python build_mane_index.py
"""
from __future__ import annotations

import gzip
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REFS_DIR     = Path("/mnt/d/phd/scripts/16_ev_signature_predictor/data/refs")
MANE_GFF     = REFS_DIR / "MANE.GRCh38.v1.5.ensembl_genomic.gff.gz"
IDMAPPING    = REFS_DIR / "HUMAN_9606_idmapping.dat.gz"
PICKLE_OUT   = REFS_DIR / "mane_cds_index.pkl"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CDSExon:
    """One CDS exon in genomic coords (GFF: 1-based inclusive)."""
    chrom: str
    g_start: int       # 1-based inclusive
    g_end: int         # 1-based inclusive
    strand: str        # '+' or '-'
    phase: int         # 0/1/2


@dataclass
class TranscriptCDS:
    """All CDS exons for one MANE Select transcript."""
    transcript_id: str        # ENST...
    protein_id: str           # ENSP...
    uniprot_acc: str
    chrom: str
    strand: str
    exons: List[CDSExon] = field(default_factory=list)   # transcription order

    @property
    def cds_length(self) -> int:
        return sum(e.g_end - e.g_start + 1 for e in self.exons)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("build_mane_index")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_gff3_attrs(s: str) -> Dict[str, str]:
    return dict(kv.split("=", 1) for kv in s.rstrip(";").split(";") if "=" in kv)


# ---------------------------------------------------------------------------
# Step 1: UniProt -> ENST mapping (filtered to MANE)
# ---------------------------------------------------------------------------

def load_uniprot_to_enst(
    idmapping_path: Path,
    mane_enst_ids: set,
    enst_to_ensg: Dict[str, str],
) -> Dict[str, str]:
    """
    Map UniProt accessions to MANE Select ENSTs via gene IDs (ENSG).
    Strategy:
      1. Read all UniProt -> Ensembl (gene) mappings from idmapping.
      2. For each ENSG, find the MANE ENST (1:1 in MANE).
      3. Also accept direct Ensembl_TRS hits as a fallback / cross-check.
    """
    # Invert: ENSG -> MANE ENST
    ensg_to_mane_enst: Dict[str, str] = {}
    for enst, ensg in enst_to_ensg.items():
        if enst in mane_enst_ids:
            if ensg in ensg_to_mane_enst:
                log.warning(f"{ensg} has multiple MANE ENSTs "
                            f"({ensg_to_mane_enst[ensg]}, {enst})")
                continue
            ensg_to_mane_enst[ensg] = enst
    log.info(f"ENSG -> MANE ENST: {len(ensg_to_mane_enst)} mappings")

    # Now walk idmapping
    upr_to_enst: Dict[str, str] = {}
    n_via_gene = 0
    n_via_trs = 0

    with gzip.open(idmapping_path, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            acc, db, val = parts

            if db == "Ensembl":
                ensg = val.split(".")[0]
                if ensg in ensg_to_mane_enst and acc not in upr_to_enst:
                    upr_to_enst[acc] = ensg_to_mane_enst[ensg]
                    n_via_gene += 1

            elif db == "Ensembl_TRS":
                enst = val.split(".")[0]
                if enst in mane_enst_ids and acc not in upr_to_enst:
                    upr_to_enst[acc] = enst
                    n_via_trs += 1

    log.info(f"UniProt->MANE ENST: {len(upr_to_enst)} mappings "
             f"({n_via_gene} via gene, {n_via_trs} via transcript)")
    return upr_to_enst



# ---------------------------------------------------------------------------
# Step 2: Parse MANE GFF3
# ---------------------------------------------------------------------------

def parse_mane_gff(gff_path: Path):
    """
    Parse MANE GFF3. Since this file contains ONLY MANE Select transcripts,
    every transcript with CDS entries is by definition MANE Select.
    """
    enst_to_cds: Dict[str, List[CDSExon]] = {}
    enst_to_ensp: Dict[str, str] = {}
    enst_to_ensg: Dict[str, str] = {}     # NEW

    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            feature = f[2]
            attrs = _parse_gff3_attrs(f[8])

            # NEW: capture transcript -> gene from 'transcript' lines
            if feature == "transcript":
                tid = attrs.get("ID", "")
                if "ENST" in tid:
                    enst = tid.split(":")[-1].split(".")[0]
                    parent = attrs.get("Parent", "")
                    for p in parent.split(","):
                        if "ENSG" in p:
                            ensg = p.split(":")[-1].split(".")[0]
                            enst_to_ensg[enst] = ensg
                            break
                continue

            if feature != "CDS":
                continue

            chrom = f[0]
            start, end = int(f[3]), int(f[4])
            strand = f[6]
            phase = int(f[7]) if f[7].isdigit() else 0

            parent = attrs.get("Parent", "")
            enst = None
            for p in parent.split(","):
                if "ENST" in p:
                    enst = p.split(":")[-1].split(".")[0]
                    break
            if enst is None:
                continue

            enst_to_cds.setdefault(enst, []).append(
                CDSExon(chrom=chrom, g_start=start, g_end=end,
                        strand=strand, phase=phase)
            )
            if "protein_id" in attrs:
                enst_to_ensp[enst] = attrs["protein_id"].split(".")[0]

    mane_enst_ids = set(enst_to_cds.keys())
    log.info(f"MANE Select transcripts (from CDS records): {len(mane_enst_ids)}")
    log.info(f"ENST -> ENSG mappings: {len(enst_to_ensg)}")
    return mane_enst_ids, enst_to_cds, enst_to_ensp, enst_to_ensg

# ---------------------------------------------------------------------------
# Step 3: Assemble final index
# ---------------------------------------------------------------------------
def build_index(
    upr_to_enst: Dict[str, str],
    enst_to_cds: Dict[str, List[CDSExon]],
    enst_to_ensp: Dict[str, str],
) -> Dict[str, TranscriptCDS]:
    """Sort exons in transcription order and assemble per-UniProt index."""
    index: Dict[str, TranscriptCDS] = {}

    for upr, enst in upr_to_enst.items():
        exons = enst_to_cds.get(enst)
        if not exons:
            continue
        strand = exons[0].strand
        chrom = exons[0].chrom
        # Sanity: all exons same chrom & strand
        if not all(e.chrom == chrom and e.strand == strand for e in exons):
            log.warning(f"{enst}: mixed chrom/strand in CDS; skipping")
            continue

        # Sort: ascending genomic for +, descending for -
        exons_sorted = sorted(
            exons, key=lambda e: e.g_start, reverse=(strand == "-")
        )

        tx = TranscriptCDS(
            transcript_id=enst,
            protein_id=enst_to_ensp.get(enst, ""),
            uniprot_acc=upr,
            chrom=chrom,
            strand=strand,
            exons=exons_sorted,
        )

        # Sanity: CDS length divisible by 3
        if tx.cds_length % 3 != 0:
            log.warning(f"{enst} ({upr}): CDS length {tx.cds_length} "
                        f"not divisible by 3; skipping")
            continue

        # Sanity: first exon phase 0 (canonical start)
        if exons_sorted[0].phase != 0:
            log.warning(f"{enst} ({upr}): first exon phase = "
                        f"{exons_sorted[0].phase}, expected 0")
            # don't skip — could still be valid for partial coverage

        index[upr] = tx

    log.info(f"Final index: {len(index)} UniProt accessions")
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("Step 1/3: parsing MANE GFF3...")
    mane_enst_ids, enst_to_cds, enst_to_ensp, enst_to_ensg = parse_mane_gff(MANE_GFF)

    log.info("Step 2/3: building UniProt -> MANE ENST mapping...")
    upr_to_enst = load_uniprot_to_enst(IDMAPPING, mane_enst_ids, enst_to_ensg)

    log.info("Step 3/3: assembling final index...")
    index = build_index(upr_to_enst, enst_to_cds, enst_to_ensp)

    log.info(f"Writing pickle: {PICKLE_OUT}")
    with open(PICKLE_OUT, "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Done.")

if __name__ == "__main__":
    main()