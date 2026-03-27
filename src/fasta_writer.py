# fasta_writer.py

from pathlib import Path


def build_fasta_records(df):

    records = []

    for _, row in df.iterrows():

        if not row["full_seq"]:
            continue

        header = (
            f">{row['UniqueID']}_{row['win_start']}-{row['win_end']}"
            f"|prot_range={row['win_start']}-{row['win_end']}"
            f"|motif_range={row['motif_start']}-{row['motif_end']}"
            f"|Lflank={row['left_flank']}"
            f"|Rflank={row['right_flank']}"
            f"|mode={row['mode']}"
        )

        records.append((header, row["full_seq"]))

    return records


def write_fasta(records, out_fasta):
    with open(out_fasta, "w") as f:
        for header, seq in records:
            f.write(header + "\n")
            f.write(seq + "\n")


def write_fasta_chunked(records, out_fasta, max_residues=100_000):

    out_path = Path(out_fasta)
    stem = out_path.stem
    suffix = out_path.suffix
    parent = out_path.parent

    chunk_idx = 1
    current_records = []
    current_residues = 0
    written_files = []

    for header, seq in records:
        seq_len = len(seq)

        if current_residues + seq_len > max_residues and current_records:
            out_chunk = parent / f"{stem}_part{chunk_idx}{suffix}"
            write_fasta(current_records, out_chunk)
            written_files.append(str(out_chunk))
            chunk_idx += 1
            current_records = []
            current_residues = 0

        current_records.append((header, seq))
        current_residues += seq_len

    if current_records:
        out_chunk = parent / f"{stem}_part{chunk_idx}{suffix}"
        write_fasta(current_records, out_chunk)
        written_files.append(str(out_chunk))

    return written_files