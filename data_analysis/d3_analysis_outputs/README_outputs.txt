D3 psychometric analysis outputs

00_data_quality_summary.csv
  Dataset size, participant counts, parse-success rates.

01_participant_llm_scores_long.csv
  Participant-level LLM scores, including individual models and MODEL_AVERAGE.

01_participant_model_average_scores_wide.csv
  Convenient merged participant-level file with model-average LLM scores and questionnaire scores.

02_within_model_repetition_icc.csv
  ICC(C,1)/ICC(C,k) and ICC(A,1)/ICC(A,k) across repeated calls within each model.

03_between_model_icc.csv
  ICC(C,1)/ICC(C,k) and ICC(A,1)/ICC(A,k) across models after averaging repetitions.

04_model_model_agreement.csv
  Pairwise model-model correlations for participant-level target-trait LLM scores.

05_model_sd3_matched_agreement.csv
  Each model's matched-trait correlation with SD3 scores.

06_multitrait_validity_correlations.csv
  Multitrait LLM-SD3 correlation matrix with raw and, if supplied, disattenuated correlations.

07_convergent_vs_discriminant_summary.csv
  Mean/median convergent vs discriminant correlations by model.

08_questionnaire_benchmark_correlations.csv
  Questionnaire-to-questionnaire correlation benchmarks from the same sample.

09_prompt_specificity_broad_vs_specific.csv
  Broad vs specific prompt validity comparisons using Williams dependent-correlation tests and bootstrap CIs.

10_mcsds_discrepancy_regressions.csv
  Regression of absolute standardized LLM-SD3 discrepancies on MCSDS.

11_mcsds_correlation_comparisons.csv
  Comparison of MCSDS-SD3 correlations with MCSDS-LLM correlations.

12_sensitivity_partial_available_validity_correlations.csv
  Same as output 06, but using all available trait-level ratings, including partially parsed calls.

13_sensitivity_partial_available_convergent_vs_discriminant.csv
  Same as output 07, using all available trait-level ratings.

14_sensitivity_partial_available_mcsds_discrepancy.csv
  Same as output 10, using all available trait-level ratings.

15_sensitivity_partial_available_mcsds_correlation_comparisons.csv
  Same as output 11, using all available trait-level ratings.

16_sd3_reliability_alphas.csv
  Optional output created when --item_long_csv is supplied; Cronbach's alpha for SD3 subscales.
