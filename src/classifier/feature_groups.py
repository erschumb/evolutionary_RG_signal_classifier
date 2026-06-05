"""
Feature group definitions for the RG-motif classifier.

Two layers:
  FEATURE_GROUPS  — fine-grained groups, one per biological/data source. Used for
                    leave-one-group-out + group-in-isolation analyses and group-
                    level permutation importance.
  SUPERGROUPS     — coarse roll-ups for the headline narrative, esp. the
                    "baseline composition vs variant-derived selection" contrast.

Notes baked in:
  * 'substitution_score' is the ONLY fold-derived group (refit per CV fold by the
    SubstitutionScoreTransformer). Every other group is fold-static.
  * AM-derived AF features are pulled OUT of 'allele_frequency' into
    'am_af_interaction' so the AM-vs-AF circularity is measurable, not pre-mixed.
  * 'wt_physchem' is its own group (NOT merged with deltas) so the pure-sequence
    label-proxy can be ablated independently.
  * Pure-sequence groups are collected under the 'composition' supergroup for the
    composition-vs-selection ablation.
Column names are matched leniently at harness time (a group may list columns not
present in a given matrix; missing ones are silently skipped, and a column in no
group is reported so nothing is silently lost).
"""

FEATURE_GROUPS = {
    # ── pure-sequence, region-level (NO variants) — label-proxy candidates ──
    "sequence_basic": [
        "region_length", "n_rg_motifs", "rg_fraction",
    ],
    "wt_physchem": [
        "wt_ncpr", "wt_fcr", "wt_kappa", "wt_hydropathy",
        "wt_aromaticity", "wt_fraction_proline", "wt_n_pos", "wt_n_neg",
    ],
    "codon_usage": [
        # matches codon_source_aas=list("ADEGLPRS"); keep this in sync with the
        # codon_source_aas you pass to assemble_everything.
        "codon_A_GCT", "codon_A_GCC", "codon_A_GCA", "codon_A_GCG",
        "codon_D_GAT", "codon_D_GAC",
        "codon_E_GAA", "codon_E_GAG",
        "codon_G_GGT", "codon_G_GGC", "codon_G_GGA", "codon_G_GGG",
        "codon_L_TTA", "codon_L_TTG", "codon_L_CTT", "codon_L_CTC",
        "codon_L_CTA", "codon_L_CTG",
        "codon_P_CCT", "codon_P_CCC", "codon_P_CCA", "codon_P_CCG",
        "codon_R_CGT", "codon_R_CGC", "codon_R_CGA", "codon_R_CGG",
        "codon_R_AGA", "codon_R_AGG",
        "codon_S_TCT", "codon_S_TCC", "codon_S_TCA", "codon_S_TCG",
        "codon_S_AGT", "codon_S_AGC",
    ],
    "gc_codon_indices": [
        "gc", "gc3", "cpg_frac", "cpg_oe", "codon_mean_mutability",
    ],

    # ── variant burden / composition (variants, but no label contrast) ─────
    "variant_composition": [
        "density_synonymous", "fraction_synonymous",
        "density_missense", "fraction_missense",
        "density_inframe_indel", "fraction_inframe_indel",
        "density_LoF", "fraction_LoF",
        "density_other", "fraction_other",
        "n_synonymous", "n_missense", "n_inframe_indel", "n_LoF", "n_other",
        "n_variants_total", "variant_density",
    ],

    # ── deleteriousness predictors (external models) ───────────────────────
    "alphamissense": [
        "am_median", "am_mean", "am_max", "am_std", "am_fraction_pathogenic",
    ],
    "esm": [
        "esm_mean", "esm_median", "esm_min", "esm_std",
        "esm_fraction_disruptive", "esm_n_annotated",
    ],

    # ── population-genetic selection (gnomAD AF) — AM-derived ones removed ──
    "allele_frequency": [
        "af_n_syn", "af_n_mis", "af_n_indel", "af_n_lof",
        "af_median_log10_syn", "af_frac_singleton_syn", "af_frac_common_syn",
        "af_median_log10_mis", "af_frac_singleton_mis", "af_frac_common_mis",
        "af_median_log10_indel", "af_frac_singleton_indel", "af_frac_common_indel",
        "af_median_log10_lof", "af_frac_singleton_lof", "af_frac_common_lof",
        "af_log2ratio_mis_syn_count", "af_log2ratio_lof_syn_count",
        "af_median_log10_delta_mis_syn", "af_frac_rare_delta_mis_syn",
        "af_rg_n_variants", "af_rg_median_log10", "af_rg_frac_singleton",
        "af_rg_vs_nonrg_log10_delta_mis",
    ],
    # AM x AF interactions broken out so AF-in-isolation is AM-free
    "am_af_interaction": [
        "af_am_score_weighted_by_rarity", "af_n_likely_path_rare",
    ],

    # ── RG-motif-specific effects ──────────────────────────────────────────
    "rg_events": [
        "rg_event_fraction_no_change", "rg_event_fraction_gain",
        "rg_event_fraction_loss", "rg_event_fraction_movement",
        "delta_rg_ratio_rel_mean",
        "rg_fraction_rgs_hit_LoF", "rg_fraction_rgs_hit_inframe_indel",
        "rg_fraction_rgs_hit_missense", "rg_fraction_rgs_hit_synonymous",
        "rg_mean_burden_on_hit_LoF", "rg_mean_burden_on_hit_inframe_indel",
        "rg_mean_burden_on_hit_missense", "rg_mean_burden_on_hit_synonymous",
        "n_g_hits_disrupting", "n_r_hits_disrupting", "rg_r_fraction",
    ],

    # ── physchem deltas (variant-driven shifts; WT split off above) ─────────
    "physchem_deltas": [
        "delta_ncpr", "delta_fcr", "delta_kappa", "delta_hydropathy",
        "delta_aromaticity", "delta_fraction_proline", "delta_n_pos", "delta_n_neg",
    ],

    # # ── static substitution-biochemistry rates (NOT the folded score) ──────
    # "substitution_classes": [
    #     "sub_rate_charge_alter", "sub_rate_pos_to_anything",
    #     "sub_rate_aromatic_alter", "sub_rate_hydrophobic_alter",
    #     "sub_rate_polar_alter", "sub_rate_proline_intro",
    #     "sub_rate_glycine_intro", "sub_rate_conservative",
    # ],

    # ── FOLD-DERIVED: group-specific selection-corrected substitution score ─
    # (refit per CV fold; the only label-derived group)
    "substitution_score": [
        "sub_score_mean",
        "sub_score_mean_R", "sub_score_mean_G", "sub_score_mean_other",
    ],
}

# Coarse roll-ups for the headline composition-vs-selection contrast.
SUPERGROUPS = {
    # pure-sequence: separates the classes by baseline composition, not selection
    "composition": [
        "sequence_basic", "wt_physchem", "codon_usage", "gc_codon_indices",
    ],
    # variant-derived deleteriousness/constraint predictions
    "deleteriousness": ["alphamissense", "esm", "am_af_interaction"],
    # population-genetic + substitution selection signals (the on-narrative core)
    "selection": [
        "allele_frequency", "substitution_score",
        "physchem_deltas",
    ],
    # RG-motif-specific
    "rg_specific": ["rg_events"],
    # raw burden/composition of observed variants
    "burden": ["variant_composition"],
}

# the only fold-derived group (handled specially by the harness)
FOLDED_GROUPS = {"substitution_score"}