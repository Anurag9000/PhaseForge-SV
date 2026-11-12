# Strict Kaggle and Google research separation

## Kaggle-only train/evaluate

- Fit the Universal Background Model only on supplied Kaggle background.csv
  recordings and competition audio.zip. No external original Google recordings.
- Fit PCA, ICA and any learned normalization exclusively on background data.
- Create Z/T/S score-normalization cohorts only from background speakers.
- MAP-adapt development from dev_enrol and evaluation from eval_enrol using
  only the specified 1/2/3 enrol_index recordings, never test files.
- Treat each test trial as independent: average its phase-derived framewise
  speaker/UBM log likelihood ratio. Group-level four-claim adjustment is only
  exploratory and is not the required per-trial LLR.
- Verify all eval IDs, score finiteness and required CSV format.
- Label model/feature experiments as completed only when a run and its
  corresponding metrics/checkpoints exist.

## Original Google research

The original Speech Commands archive is source material for the Kaggle dataset,
including renamed and perturbed waveforms. A SHA mismatch is not a guarantee
of independence. Use the Google corpus only for separate speaker-disjoint
research; do not infer anonymized Kaggle hidden speaker identities from
original filenames or use extra Google audio for competition UBM training.

Generate a corpus-wide exact duplicate manifest, partition by original speaker
into locked development/test populations, and restrict UBM and PCA fitting to
research-background speaker data. Use all-word UBM only within research.

## Statistical and scientific reproducibility

For GMM behavior perform at least 20 seeds per initializer, plus covariance
type, AIC/BIC, heldout average LL, mixture sweep, degeneracy and remedy.
Report training times, EM histories, MAP relevance, enrolment size,
ROC/EER/AUC, FAR/FRR and grouped bootstrap uncertainty. Test models on
independent background proxy speakers and, separately, original Google corpus.

No Kaggle eval trial labels exist locally. The previous 4.61% development EER
is historical, not a verified result of a new repo training run.
