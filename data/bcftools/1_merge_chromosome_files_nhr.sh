#!/bin/bash
#SBATCH -J gnomad_vep_combine_nhr
#SBATCH -o logs/gnomad_vep_combine_nhr.%j.out
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:30:00
#SBATCH -A ki-mireg

set -euo pipefail

output_dir="/lustre/project/ki-mireg/temp_eric/output_vep_nhr_test"
combined_file="${output_dir}/combined_vep_variants_4groups.tsv"

# Header — must match column order in gnomad_vep_array.sh
echo -e "CHROM\tPOS\tREF\tALT\tFILTER\tConsequence\tIMPACT\tSYMBOL\tGene\tFeature\tBIOTYPE\tCANONICAL\tMANE_SELECT\tENSP\tUNIPROT_ISOFORM\tProtein_position\tAmino_acids\tCodons\tHGVSc\tHGVSp\tLoF\tLoF_filter\tLoF_flags" > "$combined_file"

for chr in chr{1..22} chrX chrY; do
    f="${output_dir}/vep_variants_${chr}.tsv"
    if [ -f "$f" ]; then
        cat "$f" >> "$combined_file"
    else
        echo "WARNING: missing $f"
    fi
done

n_rows=$(($(wc -l < "$combined_file") - 1))
echo "Combined → $combined_file"
echo "Total transcript-consequence rows: $n_rows"