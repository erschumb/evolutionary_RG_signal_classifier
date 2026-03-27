#!/usr/bin/env python3
"""
coding_dna.py

Core API for extracting exact coding DNA for protein regions (Uniprot -> genomic).
Includes a small CLI to run on a file of regions.

Usage (CLI):
    python coding_dna.py --input regions.json --output out.json
    python coding_dna.py --input regions.csv --output out.json --no-parallel

Input formats:
- JSON: a list of tuples/lists like [["P12345", 10, 20], ["Q9Y4Z0", 1, 5]]
  or list of objects [{"protein":"P12345","start":10,"end":20}, ...]
- CSV: header with columns protein,start,end
"""
import argparse
import csv
import json
import logging
import time
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional
from Bio.Seq import Seq
from polars import corr
import requests
from torch import concat

# ============================================================
#   LOGGING SETUP
# ============================================================
logger = logging.getLogger("coding_dna")
logger.setLevel(logging.INFO)

# Clear old handlers that accumulate due to Jupyter re-imports
logger.handlers.clear()

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ============================================================
#   NETWORK HELPERS (RETRY LOGIC)
# ============================================================
def _safe_get(url: str, headers: Optional[dict] = None, max_retries: int = 4, sleep: float = 1):
    """Robust GET request with retries."""
    flag_not_found =  False
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return r
            if not flag_not_found:
                logger.warning(f"Status {r.status_code} for {url} (attempt {attempt+1})")
            flag_not_found = True
        except Exception as e:
            logger.warning(f"Request failed ({attempt + 1}): {e}")

        time.sleep(sleep)

    logger.error(f"FAILED after {max_retries} attempts: {url}")
    return None


# ============================================================
#   PROTEIN LOCATION PARSING
# ============================================================
def _parse_protein_location(pl: dict) -> Tuple[int, int]:
    """
    Accepts UniProt proteinLocation shapes:
      {"position": 57}
      {"position": {"position": 57}}
      {"begin": {"position": 12}, "end": {"position": 35}}
    Returns (start, end) as ints.
    """
    if "position" in pl:
        pos = pl["position"]
        return (int(pos["position"]), int(pos["position"])) if isinstance(pos, dict) else (int(pos), int(pos))

    if "begin" in pl and "end" in pl:
        return int(pl["begin"]["position"]), int(pl["end"]["position"])

    raise ValueError(f"Unexpected proteinLocation structure: {pl}")


# ============================================================
#   FETCH SEQUENCE (ENSEMBL)
# ============================================================
def _fetch_seq(chrom: str, start: int, end: int, species: str = "human") -> Optional[str]:
    url = f"https://rest.ensembl.org/sequence/region/{species}/{chrom}:{start}..{end}:1"
    r = _safe_get(url, headers={"Content-Type": "text/plain"})
    if r is None:
        logger.error(f"Failed fetching genomic region {chrom}:{start}-{end}")
        return None
    return r.text.strip()


# ============================================================
#   EXON EXTRACTION
# ============================================================
def _extract_exons(entry: dict):
    """
    Normalize UniProt exon entries into list of:
      {p_start, p_end, g_start, g_end, len}
    and return (chrom, sorted_exons).
    """
    exons = []
    chrom = "chr" + str(entry["genomicLocation"]["chromosome"])

    for ex in entry["genomicLocation"]["exon"]:
        p_start, p_end = _parse_protein_location(ex["proteinLocation"])
        gl = ex["genomeLocation"]
        if "begin" in gl:
            g1 = gl["begin"]["position"]
            g2 = gl["end"]["position"]
        else:
            g1, g2 = _parse_protein_location(gl)

        exons.append({
            "p_start": p_start,
            "p_end": p_end,
            "g_start": min(g1, g2),
            "g_end": max(g1, g2),
            "len": abs(g2 - g1) + 1,
        })

    return chrom, sorted(exons, key=lambda e: e["p_start"])


# ============================================================
#   TRANSLATION CHECK
# ============================================================
def _translation_check(dna: str, prot_seq: Optional[str], aa_start: int, aa_end: int, protein_id: str) -> Optional[str]:
    """
    Translate DNA and compare to prot_seq slice if available.
    Returns the translated sequence (or None if prot_seq not provided).
    """
    if not prot_seq:
        return None

    trans = str(Seq(dna).translate())
    expected = prot_seq[aa_start - 1: aa_end]

    if trans != expected:
        logger.warning(f"Translation mismatch for {protein_id}  (expected:{expected}, got:{trans}) ({aa_start}-{aa_end})")

    return trans


# ============================================================
#   FORWARD STRAND EXTRACTION
# ============================================================
def _extract_forward(entry: dict, aa_start: int, aa_end: int, prot_seq: Optional[str], protein_id: str, species: str):
    chrom, exons = _extract_exons(entry)
    # chrom = "chr" + str(chrom)
    first_aa = exons[0]["p_start"]

    exon_seqs, exon_cum, cum = [], [], 0
    for ex in exons:
        seq = _fetch_seq(chrom, ex["g_start"], ex["g_end"], species)
        if seq is None:
            return None, None, None
        exon_seqs.append(seq)
        exon_cum.append(cum)
        cum += len(seq)

    concatenated = "".join(exon_seqs)
    total_nt = len(concatenated)

    nt_start = max(0, (aa_start - first_aa) * 3)
    nt_end = min(total_nt - 1, (aa_end - first_aa) * 3 + 2)

    intervals = []
    for ex, cum_start in zip(exons, exon_cum):
        ov_s = max(nt_start, cum_start)
        ov_e = min(nt_end, cum_start + ex["len"] - 1)
        if ov_s > ov_e:
            continue

        local_s = ov_s - cum_start
        local_e = ov_e - cum_start

        intervals.append({
            "chrom": chrom,
            "start": ex["g_start"] + local_s,
            "end": ex["g_start"] + local_e,
            "strand": "+"
        })

    # add merged coordinates
    offset = 0
    for iv in intervals:
        length = iv["end"] - iv["start"] + 1
        iv["merged_start"] = offset
        iv["merged_end"] = offset + length - 1
        offset += length

    dna = concatenated[nt_start:nt_end + 1]
    # print(dna)
    trans = _translation_check(dna, prot_seq, aa_start, aa_end, protein_id)
    return intervals, dna, trans


# ============================================================
#   REVERSE STRAND EXTRACTION
# ============================================================
def _extract_reverse(entry: dict, aa_start: int, aa_end: int, prot_seq: Optional[str], protein_id: str, species: str):
    chrom, exons = _extract_exons(entry)
    # chrom = "chr" + str(chrom)
    # print(exons)
    first_aa = exons[0]["p_start"]

    exon_seqs_rc, exon_cum, cum = [], [], 0
    for ex in exons:
        seq = _fetch_seq(chrom, ex["g_start"], ex["g_end"], species)
        # print(seq)

        if seq is None:
            return None, None, None
        rc = str(Seq(seq).reverse_complement())
        exon_seqs_rc.append(rc)
        exon_cum.append(cum)
        cum += len(rc)

    concatenated = "".join(exon_seqs_rc)
    # print(concatenated)
    # print(exon_cum)

    total_nt = len(concatenated)
    # print(total_nt)


    nt_start = max(0, (aa_start - first_aa) * 3)
    nt_end = min(total_nt - 1, (aa_end - first_aa) * 3 + 2)

    intervals = []
    for ex, cum_start in zip(exons, exon_cum):
        ov_s = max(nt_start, cum_start)
        ov_e = min(nt_end, cum_start + ex["len"] - 1)
        # print(ov_s, ov_e)

        if ov_s > ov_e:
            continue

        local_s = ov_s - cum_start
        local_e = ov_e - cum_start

        g_s = ex["g_start"] + (ex["len"] - 1 - local_s)
        g_e = ex["g_start"] + (ex["len"] - 1 - local_e)

        intervals.append({
            "chrom": chrom,
            "start": min(g_s, g_e),
            "end": max(g_s, g_e),
            "strand": "-"
        })
    # print(intervals)
    offset = 0
    for iv in intervals:
        length = iv["end"] - iv["start"] + 1
        iv["merged_start"] = offset
        iv["merged_end"] = offset + length - 1
        offset += length
    dna = concatenated[nt_start:nt_end + 1]
        # print(dna)

    trans = _translation_check(dna, prot_seq, aa_start, aa_end, protein_id)
    return intervals, dna, trans

def fallback_ensembl_coordinates(
        protein_id: str,
        aa_start: int,
        aa_end: int,
        species: str = "human"
    ):
    """
    Fallback strategy when UniProt does not provide gnCoordinate.
    Uses Ensembl REST API directly:
      1. Map UniProt protein -> Ensembl protein translation.
      2. Map translation positions to genomic coordinates.

    Returns:
        {
            "intervals": [...],
            "dna": "...",
            "prot_seq": translated,
            "warning": "ensembl_fallback"
        }
        or (None, reason)
    """

    base = "https://rest.ensembl.org"
    headers_json = {"Content-Type": "application/json"}

    # Step 1. Map UniProt accession to Ensembl translation ID
    url_xref = f"{base}/xrefs/id/{protein_id}?external_db=UniProtKB/Swiss-Prot"
    r = _safe_get(url_xref, headers=headers_json)
    if r is None:
        return None, "ensembl_xref_failed"

    try:
        data = r.json()
    except:
        return None, "ensembl_xref_parse_failed"

    # Filter Ensembl peptide xrefs
    peptides = [x for x in data if x.get("type") == "translation"]
    if not peptides:
        return None, "ensembl_no_translation_id"

    translation_id = peptides[0]["id"]  # choose the first one

    # Step 2. Map AA positions to genomic coordinates
    url_map = f"{base}/map/translation/{translation_id}/{aa_start}..{aa_end}"
    r = _safe_get(url_map, headers=headers_json)
    if r is None:
        return None, "ensembl_map_failed"

    try:
        mapping = r.json()
    except:
        return None, "ensembl_map_parse_failed"

    if "mappings" not in mapping or not mapping["mappings"]:
        return None, "ensembl_map_empty"

    intervals = []
    chrom_seen = None

    for m in mapping["mappings"]:
        chrom = m["seq_region_name"]
        if chrom_seen is None:
            chrom_seen = chrom

        # Normalize naming similar to UniProt path
        chrom = "chr" + str(chrom)

        intervals.append({
            "chrom": chrom,
            "start": m["start"],
            "end": m["end"],
            "strand": "+" if m["strand"] == 1 else "-"
        })

    # Step 3. Fetch the actual DNA sequence
    dna = []
    for iv in intervals:
        seq = _fetch_seq(iv["chrom"], iv["start"], iv["end"], species)
        if seq is None:
            return None, "ensembl_dna_fetch_failed"
        dna.append(seq if iv["strand"] == "+" else str(Seq(seq).reverse_complement()))

    dna_concat = "".join(dna)

    # Translate
    prot_seq = str(Seq(dna_concat).translate())

    return {
        "intervals": intervals,
        "dna": dna_concat,
        "prot_seq": prot_seq,
        "warning": "ensembl_fallback"
    }, None



# ============================================================
#   MAIN API WRAPPER WITH FAILURE REASON
# ============================================================
def _get_exact_dna(protein_id: str, aa_start: int, aa_end: int, species: str = "human"):
    url = f"https://www.ebi.ac.uk/proteins/api/coordinates/{protein_id}"
    r = _safe_get(url, headers={"Accept": "application/json"})
    warning_msg = None
    # print(r.json())
    if r is None:
        # Try Ensembl fallback
        fallback, reason2 = fallback_ensembl_coordinates(protein_id, aa_start, aa_end, species)
        if fallback is None:
            return None, f"uniprot_lookup_failed_and_{reason2}"
        return (fallback["intervals"], fallback["dna"], fallback["prot_seq"]), fallback.get("warning")
        # return None, "uniprot_lookup_failed"

    try:
        data_json = r.json()
        data = data_json[0] if isinstance(data_json, list) else data_json
    except Exception:
        return None, "uniprot_parse_failed"
    # print(data)
    # print(data["gnCoordinate"])
    # print(len(data["gnCoordinate"]))

    if len(data["gnCoordinate"]) == 0:
        return None, "uniprot_no_gnCoordinate"
    elif len(data["gnCoordinate"]) == 1:
        entry = data["gnCoordinate"][0]
    else:
        temp_entries = data["gnCoordinate"]
        chrom_list_possible = []
        for ikj in temp_entries:
            chrom_list_possible.append(ikj["genomicLocation"]["chromosome"])
        corr_chrom = [(i, x) for i, x in enumerate(chrom_list_possible) if "_" not in x]
        # print(corr_chrom)

        if len(corr_chrom) > 1:
            warning_msg = "multiple_possible_gnCoordinates"
        elif len(corr_chrom) == 0:
            return None, "only_ALT_chromosomes_found" 
        entry = data["gnCoordinate"][corr_chrom[0][0]] 
    
    # print(entry)


    prot_seq = data.get("sequence")
    reverse = entry["genomicLocation"]["reverseStrand"]

    extractor = _extract_reverse if reverse else _extract_forward
    intervals, dna, trans = extractor(entry, aa_start, aa_end, prot_seq, protein_id, species)

    if intervals is None:
        return None, "genomic_fetch_failed"

    # translation mismatch is NOT a failure; store as soft warning
    if trans is not None:
        expected_len = aa_end - aa_start + 1
        if len(trans) != expected_len:
            return (intervals, dna, trans), "translation_mismatch"

    return (intervals, dna, trans), warning_msg

# ============================================================
#   MULTIPROCESSING WRAPPER
# ============================================================
def _process_single(region: Tuple[str, int, int, str]):
    protein_id, aa_start, aa_end, species = region
    # print(protein_id)
    

    output, reason = _get_exact_dna(protein_id, aa_start, aa_end, species)

    # Hard failure
    if output is None:
        return None, {
            "protein": protein_id,
            "region": (aa_start, aa_end),
            "reason": reason
        }

    intervals, dna, trans = output

    result = {
        "protein": protein_id,
        "prot_region": (aa_start, aa_end),
        "prot_seq": trans,
        "intervals": intervals,
        "dna": dna
    }

    if reason == "translation_mismatch":
        result["warning"] = "translation_mismatch"

    return result, None


def process_multiple_regions(region_list: List[Tuple[str, int, int]],
                             species: str = "human",
                             parallel: bool = True):

    tasks = [(pid, s, e, species) for pid, s, e in region_list]

    if parallel:
        cores = min(cpu_count(), 8)
        logger.info(f"Processing {len(tasks)} regions using {cores} CPUs")
        with Pool(cores) as pool:
            raw = pool.map(_process_single, tasks)
    else:
        logger.info(f"Processing {len(tasks)} regions sequentially")
        raw = list(map(_process_single, tasks))

    results = []
    failed = []

    for r, f in raw:
        if r is not None:
            results.append(r)
        if f is not None:
            failed.append(f)

    return results, failed


# ============================================================
#   CLI / I/O HELPERS
# ============================================================
def _read_input(path: str) -> List[Tuple[str, int, int]]:
    """Supports JSON (list of tuples or dicts) and CSV with header UniqueID,start,end."""
    if path.endswith(".json"):
        with open(path, "r") as fh:
            data = json.load(fh)
        out = []
        if isinstance(data, list):
            if len(data) == 0:
                return []
            first = data[0]
            if isinstance(first, (list, tuple)) and len(first) >= 3:
                out = [(str(x[0]), int(x[1]), int(x[2])) for x in data]
            elif isinstance(first, dict):
                for d in data:
                    out.append((str(d["UniqueID"]), int(d["start"]), int(d["end"])))
            else:
                raise ValueError("Unsupported JSON input format")
        else:
            raise ValueError("Unsupported JSON input - expected list")
        return out

    elif path.endswith(".csv"):
        out = []
        with open(path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                out.append((row["UniqueID"], int(row["start"]), int(row["end"])))
        return out

    else:
        raise ValueError("Unsupported input format: use .json or .csv")


def _write_output(path: str, content):
    with open(path, "w") as fh:
        json.dump(content, fh, indent=4)


# ============================================================
#   CLI ENTRYPOINT
# ============================================================ 
def _build_parser():
    p = argparse.ArgumentParser(prog="coding_dna")
    p.add_argument("--input", "-i", required=True, help="Input JSON or CSV file with regions")
    p.add_argument("--output", "-o", required=True, help="Output JSON path")
    p.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable multiprocessing")
    p.add_argument("--species", default="human", help="Species name for Ensembl (default 'human')")
    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    regions = _read_input(args.input)
    if not regions:
        logger.info("No regions found in input.")
        _write_output(args.output, [])
        return 0

    results, failed = process_multiple_regions(regions, species=args.species, parallel=args.parallel)

    # Write original results (unchanged format)
    _write_output(args.output, results)

    # Write failures separately
    fail_path = args.output.replace(".json", "_failed.json")
    _write_output(fail_path, failed)

    logger.info(f"Wrote {len(results)} successful regions to {args.output}")
    logger.info(f"Wrote {len(failed)} failed regions to {fail_path}")
    return 0


if __name__ == "__main__":
    main()
