"""
Load and filter gnomAD VEP TSV output.
"""
from __future__ import annotations
import pandas as pd
import json


def load_vep_tsv(path: str) -> pd.DataFrame:
    """
    Load the combined VEP TSV produced by the bcftools +split-vep pipeline.

    Keeps all rows — no filtering yet. Use filter_vep() next.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False, na_values=["."])
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def filter_vep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter VEP rows down to the ones relevant for protein-region analysis:
      - FILTER == PASS           (drops AC0 and other QC-failed variants)
      - MANE_SELECT populated    (one canonical transcript per gene)
      - BIOTYPE == protein_coding
      - Protein_position populated (needed for region assignment)
    """
    n0 = len(df)

    mask_pass    = df["FILTER"] == "PASS"
    mask_mane    = df["MANE_SELECT"].notna()
    mask_coding  = df["BIOTYPE"] == "protein_coding"
    mask_pos     = df["Protein_position"].notna()
    mask_ensembl = df["Feature"].str.startswith("ENST", na=False)

    df_filt = df[mask_pass & mask_mane & mask_coding & mask_pos & mask_ensembl].copy()

    print(f"Filter summary (starting from {n0:,} rows):")
    print(f"  PASS               : {mask_pass.sum():,} kept")
    print(f"  MANE_SELECT set    : {mask_mane.sum():,} kept")
    print(f"  protein_coding     : {mask_coding.sum():,} kept")
    print(f"  Protein_position   : {mask_pos.sum():,} kept")
    print(f"  Ensembl-sourced    : {mask_ensembl.sum():,} kept")
    print(f"  After all filters  : {len(df_filt):,} rows")

    return df_filt


import requests

def fetch_uniprot_to_symbols(uniprot_ids: list[str], batch_size: int = 50) -> dict[str, list[str]]:
    """
    Fetch primary gene symbol + synonyms for a list of UniProt accessions.
    Returns {'Q13151': ['HNRNPA0', 'synonym1', ...], ...}

    Batches requests to stay within UniProt's URL length limits.
    """
    mapping: dict[str, list[str]] = {}
    url = "https://rest.uniprot.org/uniprotkb/search"

    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i + batch_size]
        query = " OR ".join(f"accession:{acc}" for acc in batch)
        params = {
            "query": query,
            "fields": "accession,gene_names",
            "format": "tsv",
            "size": 500,
        }

        r = requests.get(url, params=params)
        r.raise_for_status()

        lines = r.text.strip().split("\n")
        for line in lines[1:]:  # skip header
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            acc = parts[0]
            gene_names = parts[1] if len(parts) > 1 else ""
            mapping[acc] = gene_names.split() if gene_names else []

    print(f"Fetched {len(mapping)} / {len(uniprot_ids)} UniProt → gene symbol mappings")
    return mapping

def build_region_lookups(
    regions_json_path: str,
    uniprot_to_symbols: dict[str, list[str]],
) -> tuple[dict, dict]:
    """
    Build two lookups so we can match variants by UniProt (primary) or
    gene symbol (fallback).

    Returns:
        uniprot_lookup: {uniprot_id: [(region_id, group, start_aa, end_aa), ...]}
        symbol_lookup:  {gene_symbol: [(region_id, group, start_aa, end_aa, uniprot_id), ...]}
    """
    with open(regions_json_path) as f:
        regions = json.load(f)

    uniprot_lookup: dict[str, list[tuple]] = {}
    symbol_lookup: dict[str, list[tuple]] = {}

    for r in regions:
        uniprot_id = r["protein"]
        start_aa, end_aa = r["prot_region"]
        entry = (r["region_id"], r["group"], start_aa, end_aa)

        uniprot_lookup.setdefault(uniprot_id, []).append(entry)

        # Add to symbol lookup for every gene name associated with this UniProt
        for symbol in uniprot_to_symbols.get(uniprot_id, []):
            symbol_entry = entry + (uniprot_id,)
            symbol_lookup.setdefault(symbol, []).append(symbol_entry)

    n_regions = sum(len(v) for v in uniprot_lookup.values())
    print(f"Built lookups: {len(uniprot_lookup)} proteins, "
          f"{len(symbol_lookup)} gene symbols, {n_regions} regions")

    return uniprot_lookup, symbol_lookup


def assign_variants_to_regions(
    df_vep: pd.DataFrame,
    uniprot_lookup: dict,
    symbol_lookup: dict,
) -> pd.DataFrame:
    """
    For each VEP row, assign it to matching region(s) based on:
      1. UniProt accession (primary)
      2. Gene symbol (fallback when UniProt is missing)

    Returns a new DataFrame with added columns:
        uniprot_accession, protein_position_int, region_id, group,
        region_start_aa, region_end_aa, matched_by
    """
    df = df_vep.copy()

    # Parse columns we'll need
    df["uniprot_accession"] = df["UNIPROT_ISOFORM"].apply(_parse_uniprot_accession)
    df["protein_position_int"] = df["Protein_position"].apply(_parse_protein_position)

    # Must have protein position at minimum
    df = df[df["protein_position_int"].notna()].copy()

    matched_rows = []
    for row in df.itertuples(index=False):
        pos = row.protein_position_int
        matched = False

        # Try UniProt first
        if row.uniprot_accession:
            regions = uniprot_lookup.get(row.uniprot_accession, [])
            for region_id, group, start_aa, end_aa in regions:
                if start_aa <= pos <= end_aa:
                    d = row._asdict()
                    d.update({
                        "region_id": region_id,
                        "group": group,
                        "region_start_aa": start_aa,
                        "region_end_aa": end_aa,
                        "matched_by": "uniprot",
                    })
                    matched_rows.append(d)
                    matched = True

        # Fall back to gene symbol
        if not matched and row.SYMBOL:
            regions = symbol_lookup.get(row.SYMBOL, [])
            for region_id, group, start_aa, end_aa, uniprot_id in regions:
                if start_aa <= pos <= end_aa:
                    d = row._asdict()
                    d.update({
                        "region_id": region_id,
                        "group": group,
                        "region_start_aa": start_aa,
                        "region_end_aa": end_aa,
                        "uniprot_accession": uniprot_id,  # backfill from lookup
                        "matched_by": "symbol",
                    })
                    matched_rows.append(d)

    result = pd.DataFrame(matched_rows)
    print(f"Matched {len(result)} variant-region assignments")
    print(f"  by UniProt: {(result['matched_by']=='uniprot').sum()}")
    print(f"  by symbol : {(result['matched_by']=='symbol').sum()}")
    return result


# Helper functions (private)

def _parse_uniprot_accession(isoform_str) -> str | None:
    if pd.isna(isoform_str) or isoform_str == ".":
        return None
    return str(isoform_str).split("-")[0]


def _parse_protein_position(pos_str) -> int | None:
    if pd.isna(pos_str):
        return None
    s = str(pos_str).split("-")[0]
    if s in ("?", ""):
        return None
    try:
        return int(s)
    except ValueError:
        return None
    
def load_and_merge_frequencies(
    df_assigned: pd.DataFrame,
    pos_tsv_path: str,
    neg_tsv_path: str,
) -> pd.DataFrame:
    """
    Load allele frequency TSVs from your earlier joint VCF extraction and
    merge them onto df_assigned by CHROM+POS+REF+ALT.

    Note: these TSVs may be incomplete relative to the current df_assigned
    (different BED files used). Variants without a match get NaN for AF fields.
    """
    af_cols_to_keep = [
        "CHROM", "POS", "REF", "ALT",
        "AC_joint", "AN_joint", "AF_joint",
        "AC_genomes", "AN_genomes", "AF_genomes",
        "AC_exomes", "AN_exomes", "AF_exomes",
        "grpmax_joint",
        "AF_joint_afr", "AF_joint_amr", "AF_joint_asj", "AF_joint_eas",
        "AF_joint_fin", "AF_joint_nfe", "AF_joint_mid", "AF_joint_sas",
        "AF_joint_remaining",
    ]

    pos_df = pd.read_csv(pos_tsv_path, sep="\t", low_memory=False,
                         usecols=af_cols_to_keep)
    neg_df = pd.read_csv(neg_tsv_path, sep="\t", low_memory=False,
                         usecols=af_cols_to_keep)

    print(f"Loaded {len(pos_df):,} pos rows, {len(neg_df):,} neg rows")

    # Combine and deduplicate (a variant could appear in both if regions overlap)
    af_df = pd.concat([pos_df, neg_df], ignore_index=True)
    af_df = af_df.drop_duplicates(subset=["CHROM", "POS", "REF", "ALT"])
    print(f"After dedup: {len(af_df):,} unique variants with AF data")

    # Merge onto df_assigned
    merged = df_assigned.merge(
        af_df,
        on=["CHROM", "POS", "REF", "ALT"],
        how="left",
    )

    n_with_af = merged["AF_joint"].notna().sum()
    n_total   = len(merged)
    print(f"Merge coverage: {n_with_af:,} / {n_total:,} rows have AF "
          f"({n_with_af/n_total:.1%})")

    return merged

def parse_amino_acids_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split VEP's Amino_acids column into before_aa and after_aa.

    VEP conventions:
        'R'     -> ('R', 'R')    synonymous
        'R/H'   -> ('R', 'H')    missense
        'R/*'   -> ('R', '*')    stop_gained
        '*/Y'   -> ('*', 'Y')    stop_lost
        'X/-'   -> ('X', '-')    deletion
        '-/X'   -> ('-', 'X')    insertion
        NaN     -> (None, None)  non-coding / no amino acid change
    """
    df = df.copy()
    parsed = df["Amino_acids"].apply(lambda x: pd.Series(_parse_aa(x)))
    df["before_aa"] = parsed[0]
    df["after_aa"]  = parsed[1]
    return df


def _parse_aa(aa_str):
    if pd.isna(aa_str):
        return (None, None)
    s = str(aa_str)
    if "/" in s:
        parts = s.split("/", 1)
        return (parts[0], parts[1])
    return (s, s)

def load_alphamissense_for_proteins(
    am_tsv_path: str,
    uniprot_ids: list[str],
) -> pd.DataFrame:
    """
    Load AlphaMissense amino-acid-substitution scores, filtered to the
    UniProt IDs of interest.

    AlphaMissense file format (AlphaMissense_aa_substitutions.tsv.gz):
        uniprot_id  protein_variant  am_pathogenicity  am_class
        Q6UWZ7      A2G              0.1289           benign
        Q6UWZ7      A2C              0.2451           ambiguous
        ...

    `protein_variant` is like 'R446H' — [original][position][new].
    """
    am = pd.read_csv(
        am_tsv_path,
        sep="\t",
        comment="#",
        usecols=["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"],
        dtype={
            "uniprot_id": "string",
            "protein_variant": "string",
            "am_pathogenicity": "float32",
            "am_class": "string",
        },
    )
    print(f"Loaded {len(am):,} total AlphaMissense rows")

    # Filter to our proteins
    am = am[am["uniprot_id"].isin(uniprot_ids)].copy()
    print(f"After UniProt filter: {len(am):,} rows for {am['uniprot_id'].nunique()} proteins")

    # Parse protein_variant into pos + before_aa + after_aa
    import re
    pattern = re.compile(r"^([A-Z])(\d+)([A-Z])$")

    def _parse(pv):
        m = pattern.match(pv)
        if m:
            return pd.Series([m.group(1), int(m.group(2)), m.group(3)])
        return pd.Series([None, None, None])

    am[["before_aa", "protein_position_int", "after_aa"]] = (
        am["protein_variant"].apply(_parse)
    )
    am["protein_position_int"] = am["protein_position_int"].astype("Int64")

    return am


def merge_alphamissense(
    df: pd.DataFrame,
    am: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge AlphaMissense scores onto the variant dataframe.
    Match key: (uniprot_accession, protein_position_int, before_aa, after_aa).
    AlphaMissense only covers missense variants, so non-missense rows get NaN.
    """
    key_cols = ["uniprot_accession", "protein_position_int", "before_aa", "after_aa"]

    am_renamed = am.rename(columns={"uniprot_id": "uniprot_accession"})[
        key_cols + ["am_pathogenicity", "am_class"]
    ]

    merged = df.merge(am_renamed, on=key_cols, how="left")

    # Coverage stats
    missense_mask = merged["Consequence"].str.contains("missense_variant", na=False)
    missense_total = missense_mask.sum()
    missense_with_am = (missense_mask & merged["am_pathogenicity"].notna()).sum()
    print(f"AlphaMissense merge coverage (missense only): "
          f"{missense_with_am:,} / {missense_total:,} "
          f"({missense_with_am/missense_total:.1%})")

    return merged