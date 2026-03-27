import argparse
import json
from pathlib import Path


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