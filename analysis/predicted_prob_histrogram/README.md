# Paired modality-mask score analysis

This module collects and analyzes saved complete-input and zero-masked scores for seven Vanilla detectors. The directory spelling is retained. Full paired masked predictions are external inputs; the appendix fixture contains complete-input scores only.

## Collect paired inputs

The manifest expects `test/{Model}_train_on_Mega-MMDF/predictions/<dataset>.json` below `--log-root`. Masked predictions live at `{Model}_train_on_Mega-MMDF_mask_{audio|video}/predictions/<dataset>.json` below `--masked-log-root`; its default is `<log-root>/mask_modality_test_bkp`. Models are AVFF, AVH, Baseline, FGMDF, FRADE, FauForensics, and MCL.

```bash
python analysis/predicted_prob_histrogram/collect_mask_manifest.py \
  --log-root /path/to/logs --masked-log-root /path/to/masked_runs --output /path/to/mask_manifest.json
python analysis/predicted_prob_histrogram/collect_paired_predictions.py \
  --manifest /path/to/mask_manifest.json --output /path/to/rafv_pairs.json.gz --task RAFV
python analysis/predicted_prob_histrogram/collect_paired_predictions.py \
  --manifest /path/to/mask_manifest.json --output /path/to/farv_pairs.json.gz --task FARV
```

Collectors pair native windows by path/frame range and verify counts and sorted original-label fingerprints. Test-time mask outputs preserve `original_audio_label`, `original_video_label`, and `original_label`; older files need equivalent original-label provenance. Video aggregation uses all evaluated windows by `dirname(path)`, maximum score, and ORed original modality labels before group selection. Different native window lengths retain detector-specific eligible cohorts.

RAFV selects RARV+RAFV and compares full versus mask-audio scores. FARV selects RARV+FARV and compares full versus mask-video. Missing groups are skipped explicitly. Exports retain model/dataset, provenance/fingerprints, and paired `video_id`, `y`, scores, and window counts. These task subsets are separate from complete-domain ROC/PR inputs.

## Metrics and plots

```bash
python analysis/predicted_prob_histrogram/run_predicted_prob_histrogram.py \
  --pairs /path/to/rafv_pairs.json.gz --datasets Mega-MMDF
python analysis/predicted_prob_histrogram/run_mask_video_analysis.py \
  --pairs /path/to/farv_pairs.json.gz --reference /path/to/mask_manifest.json
```

The RAFV entry point exports metrics and histograms; options include `--reference`, `--output-dir`, `--bin-width` (0.01), and `--bandwidth` (0.02). Default plots are `outputs/<dataset>/<model>.{png,pdf}` and `all_models.{png,pdf}`. The FARV entry point requires a reference manifest and exports metrics under `outputs/FARV_mask_video/` by default.

`core.py` implements AUC, class means, Brier scores, score gaps, and paired changes. Standard Brier is `mean((p-y)**2)`; balanced Brier gives real and fake equal weight. Score gap is `mean(p|fake)-mean(p|real)` and deltas are masked minus full. JSON/CSV preserve unrounded values. Cross-domain summaries average dataset-level metrics with equal weight. Historical classwise error rates use fake iff `p >= 0.5`.

Histograms normalize each class separately. Gaussian binned KDE uses boundary reflection and fixed bandwidth; dashed lines show class means. The [paired-change tool](../predicted_prob_change_histogram/README.md) plots differences for the same video IDs.

## Four-group paper analysis

```bash
python analysis/predicted_prob_histrogram/collect_missing_groups.py \
  --manifest /path/to/mask_manifest.json --output /path/to/vanilla_pairs.json.gz --training Vanilla
python analysis/predicted_prob_histrogram/analyze_missing_groups.py --input /path/to/vanilla_pairs.json.gz --plots
```

`collect_missing_groups.py` preserves all RARV/RAFV/FARV/FAFV videos and their full/mask-audio/mask-video scores. `analyze_missing_groups.py` writes under project-relative `references/missing_modality_ID_assets/` and checks matching legacy task metrics/manifests at their expected locations. Supply those references with the paired cache.

`build_three_settings_report.py` and `build_modality_reliance_report.py` derive report assets from these metrics. `prepare_missing_input_paper.py`, `prepare_fafv_paper.py`, `prepare_classwise_mask_table.py`, and `plot_classwise_mask_histograms.py` build the paper's six-condition tables/histograms. They require `outputs/missing_groups_ID.json.gz`, retained numeric assertions in `references/three_settings_assets/` and `references/modality_reliance_assets/`, and derived metrics under project `outputs/missing_modality_paper/`. Only the small assertion fixtures are bundled. Classwise plots use detector columns and authentic/single-fake/both-fake rows for each masking direction.

Legacy `--training RMA RMV` export requires an explicit `--random-mask-root`; the corresponding augmented-training configs are external. Extend group selection after full-window aggregation and retain original-label fingerprints and explicit skipped-group reporting.
