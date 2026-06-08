#!/bin/bash
#SBATCH -J gnomad_vep_filter_nhr_test
#SBATCH -o logs/gnomad_vep_filter_nhr_test.%A_%a.out
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -A ki-mireg
#SBATCH --array=0-23

set -euo pipefail

module purge
module load compiler/GCC/13.3.0
export PATH=/lustre/project/ki-mireg/temp_eric/software/bcftools/install/bin:$PATH
export BCFTOOLS_PLUGINS=/lustre/project/ki-mireg/temp_eric/software/bcftools/install/libexec/bcftools

chromosomes=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 \
             chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY)
chr=${chromosomes[$SLURM_ARRAY_TASK_ID]}

# ── Paths ────────────────────────────────────────────────────────────────────
# UPDATE bed_file to match the exact filename you uploaded!
exomes_dir="/lustre/project/ki-mireg/eric_m2/data_exomes"
input_file="${exomes_dir}/gnomad.exomes.v4.1.sites.${chr}.vcf.bgz"
bed_file="/lustre/project/ki-mireg/temp_eric/input/genomic_coords_combined_win5_4groups.bed"
output_dir="/lustre/project/ki-mireg/temp_eric/output_vep_nhr_test"
output_file="${output_dir}/vep_variants_${chr}.tsv"

mkdir -p "$output_dir" logs

# ── Sanity checks ────────────────────────────────────────────────────────────
[ -f "$input_file" ]       || { echo "ERROR: VCF not found: $input_file"; exit 1; }
[ -f "${input_file}.tbi" ] || { echo "ERROR: index not found: ${input_file}.tbi"; exit 1; }
[ -f "$bed_file" ]         || { echo "ERROR: BED not found: $bed_file"; exit 1; }

# ── Extract variants + VEP annotations ───────────────────────────────────────
# -a vep       gnomAD's INFO tag is 'vep' (not the default 'CSQ')
# -d           duplicate — one row per transcript consequence
# No -s        no transcript filter; filter to MANE in Python downstream
bcftools view -R "$bed_file" "$input_file" | \
bcftools +split-vep \
    -a vep -d \
    -f '%CHROM\t%POS\t%REF\t%ALT\t%FILTER\t%Consequence\t%IMPACT\t%SYMBOL\t%Gene\t%Feature\t%BIOTYPE\t%CANONICAL\t%MANE_SELECT\t%ENSP\t%UNIPROT_ISOFORM\t%Protein_position\t%Amino_acids\t%Codons\t%HGVSc\t%HGVSp\t%LoF\t%LoF_filter\t%LoF_flags\n' \
    > "$output_file"

n_rows=$(wc -l < "$output_file")
echo "Processed ${chr}: ${n_rows} transcript-consequence rows → $output_file"