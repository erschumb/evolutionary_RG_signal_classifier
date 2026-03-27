import json
import pandas as pd
from pathlib import Path
import argparse


def blast_json_to_dataframe(json_path):
    """
    Parse NCBI BLAST+ JSON output (BlastOutput2 format)
    and return a pandas DataFrame with query-anchored alignments.
    """

    with open(json_path) as f:
        data = json.load(f)

    rows = []

    for entry in data.get("BlastOutput2", []):
        search = entry["report"]["results"]["search"]

        query_id = search.get("query_id")
        query_title_full = search.get("query_title", "")
        query_title = query_title_full.split("|")[0]

        # --- parse FASTA header metadata ---
        header_fields = {}
        for field in query_title_full.split("|")[1:]:
            if "=" in field:
                k, v = field.split("=", 1)
                header_fields[k] = v

        motif_range = header_fields.get("motif_range", "0-0")
        motif_start, motif_end = map(int, motif_range.split("-"))

        mode = header_fields.get("mode", "unknown")

        prot_range = header_fields.get("prot_range", "0-0")
        win_start, win_end = map(int, prot_range.split("-"))

        query_len = int(search.get("query_len", 0))

        for hit in search.get("hits", []):

            desc = hit.get("description", [{}])[0]

            # Some hits may not have HSPs
            hsps = hit.get("hsps", [])
            if not hsps:
                continue

            hsp = hsps[0]  # top HSP only

            rows.append({
                "query_id":       query_id,
                "query_title":    query_title,
                "query_len":      query_len,
                "mode":           mode,
                "win_start":      win_start,
                "win_end":        win_end,
                "motif_start":    motif_start,
                "motif_end":      motif_end,
                "hit_accession":  desc.get("accession"),
                "hit_title":      desc.get("title"),
                "species":        desc.get("sciname"),
                "taxid":          desc.get("taxid"),
                "evalue":         hsp.get("evalue"),
                "bit_score":      hsp.get("bit_score"),
                "identity":       hsp.get("identity"),
                "align_len":      hsp.get("align_len"),
                "query_from":     hsp.get("query_from"),
                "query_to":       hsp.get("query_to"),
                "qseq":           hsp.get("qseq"),
                "hseq":           hsp.get("hseq"),
            })

    return pd.DataFrame(rows)


def load_blast_group(folder_path, *filenames):
    """
    Load multiple BLAST JSON files and concatenate into one DataFrame.
    """

    folder = Path(folder_path)

    dfs = []
    for fname in filenames:
        path = folder / fname
        df = blast_json_to_dataframe(path)
        df["source_file"] = fname  # useful for traceability
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)




# ---------------- CLI ---------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse BLAST JSON files into a single DataFrame"
    )

    parser.add_argument(
        "--input_folder",
        required=True,
        help="Folder containing BLAST JSON files"
    )

    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of JSON files to parse"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output file (CSV or Parquet)"
    )

    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = load_blast_group(args.input_folder, *args.files)

    output_path = Path(args.output)

    if args.format == "csv":
        df.to_csv(output_path, index=False)
    elif args.format == "parquet":
        df.to_parquet(output_path, index=False)

    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()