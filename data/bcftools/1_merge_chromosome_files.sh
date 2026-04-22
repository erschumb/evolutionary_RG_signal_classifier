#!/bin/bash
#SBATCH -J gnomad_vep_combine
#SBATCH -o logs/gnomad_vep_combine.%j.out
#SBATCH -p smp
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:30:00
#SBATCH -A m2_jgu-cbdm

set -euo pipefail

output_dir="/lustre/project/m2_jgu-cbdm/erschumb/gnomad/output_vep_combined"
combined_file="${output_dir}/combined_vep_variants.tsv"

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