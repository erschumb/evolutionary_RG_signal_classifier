from __future__ import annotations
import argparse
import json
from pathlib import Path



def write_merged_bed(
    pos_results: list[dict],
    neg_results: list[dict],
    output_path: str,
) -> None:
    """
    Write a merged BED file from pos and neg genomic coordinate results.

    One BED line per (region, interval). Multi-interval regions produce
    multiple lines. Output is sorted by chromosome and start position,
    which is required by bcftools -R.

    BED columns:
        1. chrom
        2. start (0-based, per BED convention)
        3. end   (exclusive)
        4. region_id
        5. group   ("pos" or "neg")
    """
    rows = []
    for results, group in [(pos_results, "pos"), (neg_results, "neg")]:
        for region in results:
            rid = region["region_id"]
            for iv in region["intervals"]:
                rows.append((iv["chrom"], iv["start"], iv["end"], rid, group))

    # Natural chromosome sort: chr1, chr2, ... chr10, ... chrX, chrY
    def chrom_sort_key(row):
        chrom = row[0].replace("chr", "")
        if chrom == "X":
            return (23, row[1])
        if chrom == "Y":
            return (24, row[1])
        if chrom == "M" or chrom == "MT":
            return (25, row[1])
        try:
            return (int(chrom), row[1])
        except ValueError:
            return (99, row[1])  # anything unexpected goes last

    rows.sort(key=chrom_sort_key)

    with open(output_path, "w") as f:
        for chrom, start, end, rid, group in rows:
            f.write(f"{chrom}\t{start}\t{end}\t{rid}\t{group}\n")

    n_pos = sum(1 for r in rows if r[4] == "pos")
    n_neg = sum(1 for r in rows if r[4] == "neg")
    print(f"Wrote {len(rows)} intervals to {output_path}")
    print(f"  pos: {n_pos}, neg: {n_neg}")

def write_bed_from_results(results, output_file):
    with open(output_file, "w") as f:
        for res in results:

            protein = res.get("protein")
            prot_region = res.get("prot_region")
            intervals = res.get("intervals", [])

            if protein is None or prot_region is None:
                continue

            protein_start, protein_end = prot_region

            if not intervals:
                continue

            for iv in intervals:

                chrom = iv.get("chrom")
                if chrom is None:
                    continue

                # Skip ALT contigs
                if str(chrom).startswith("HSCHR"):
                    continue

                chrom = str(chrom)
                if not chrom.startswith("chr"):
                    chrom = "chr" + chrom

                start = iv.get("start")
                end = iv.get("end")

                if start is None or end is None:
                    continue

                end = end + 1  # BED convention
                strand = iv.get("strand", "+")

                name = (
                    f"{protein}_{protein_start}_{protein_end}_"
                    f"{chrom}_{start}_{end}_{strand}"
                )

                row = [
                    chrom,
                    str(start),
                    str(end),
                    name,
                    "0",
                    strand,
                    str(start),
                    str(end),
                    "0",
                    "1",
                    str(end - start),
                    "0",
                ]

                f.write("\t".join(row) + "\n")


# ---------------- CLI ---------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert protein-genome mapping results to BED format"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input file (JSON or JSONL)"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output BED file"
    )

    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Input format (default: json)"
    )

    return parser.parse_args()


def load_results(input_path, fmt):
    if fmt == "json":
        with open(input_path) as f:
            return json.load(f)

    elif fmt == "jsonl":
        results = []
        with open(input_path) as f:
            for line in f:
                results.append(json.loads(line))
        return results


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    results = load_results(input_path, args.format)

    write_bed_from_results(results, output_path)

    print(f"Wrote BED file to {output_path}")


if __name__ == "__main__":
    main()