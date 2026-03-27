# src/rgmotif/cli/process_blast.py

import argparse
import pandas as pd

from src.clean_up_blast_tools.expand import expand_hits_to_motifs
from src.clean_up_blast_tools.coverage import compute_rg_window_columns
from src.clean_up_blast_tools.assignment import assign_shared_accessions


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--blast_df", required=True)
    parser.add_argument("--windows_df", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--coverage_filter",
        choices=["full", "partial", "none", "all"],
        default="full"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df_blast = pd.read_pickle(args.blast_df)
    df_windows = pd.read_pickle(args.windows_df)

    df = expand_hits_to_motifs(df_blast, df_windows)
    df = compute_rg_window_columns(df)

    if args.coverage_filter != "all":
        df = df[df["rg_window_coverage"] == args.coverage_filter]

    df, report = assign_shared_accessions(df)

    df.to_parquet(args.output, index=False)

    print("Done.")
    print(report)


if __name__ == "__main__":
    main()