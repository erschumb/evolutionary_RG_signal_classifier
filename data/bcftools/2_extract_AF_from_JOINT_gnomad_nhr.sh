#!/bin/bash
#SBATCH -J gnomad_joint_af
#SBATCH -o logs/gnomad_joint_af.%A_%a.out
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -t 08:00:00
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

# ── Paths (UPDATE joint_dir to your actual joint VCF location) ───────────────
joint_dir="/lustre/project/ki-mireg/cihan/population_proj/JOINT/data_joint"   # <-- set to your path
input_file="${joint_dir}/gnomad.joint.v4.1.sites.${chr}.vcf.bgz"
bed_file="/lustre/project/ki-mireg/temp_eric/input/genomic_coords_combined_win5_4groups.bed"
output_dir="/lustre/project/ki-mireg/temp_eric/output_joint_af"
output_file="${output_dir}/joint_variants_${chr}.tsv"
mkdir -p "$output_dir" logs

# ── Sanity checks ────────────────────────────────────────────────────────────
[ -f "$input_file" ] || { echo "ERROR: VCF not found: $input_file"; exit 1; }
[ -f "$bed_file" ]   || { echo "ERROR: BED not found: $bed_file"; exit 1; }
if [ ! -f "${input_file}.tbi" ] || [ "${input_file}" -nt "${input_file}.tbi" ]; then
  echo "Regenerating index for ${input_file}"
  bcftools index -t "$input_file"
fi

# ── Extract joint AF over the combined 4-group BED ───────────────────────────
bcftools view -R "$bed_file" "$input_file" | \
bcftools query -f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t%QUAL\t%FILTER\t%INFO/AC_joint\t%INFO/AN_joint\t%INFO/AF_joint\t%INFO/AC_genomes\t%INFO/AN_genomes\t%INFO/AF_genomes\t%INFO/AC_exomes\t%INFO/AN_exomes\t%INFO/AF_exomes\t%INFO/grpmax_joint\t%INFO/AF_joint_afr\t%INFO/AF_joint_amr\t%INFO/AF_joint_asj\t%INFO/AF_joint_eas\t%INFO/AF_joint_fin\t%INFO/AF_joint_nfe\t%INFO/AF_joint_mid\t%INFO/AF_joint_sas\t%INFO/AF_joint_remaining\n' \
    > "$output_file"

echo "Processed ${chr}: $(wc -l < "$output_file") rows -> $output_file"