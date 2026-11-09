# Mega-MMDF real-only misalignment tests

These configurations evaluate audiovisual correspondence while retaining source
authenticity labels. The seven trained-model YAML files use their ordinary
Mega-MMDF model settings. By default they shift audio by one complete window
within the same clip and evaluate source RARV records only.

Set `data_config.misalignment.shift: 0` for an aligned real-only control, or
`modality: video` to shift video and landmarks. Single-window clips remain
aligned. Use a different evaluation output directory whenever these settings
change, because existing prediction files are reused by existence.

## Run and compare

Run from the repository root after preparing the data and checkpoints:

```bash
bash commands/misaligned_test_commands.sh
```

The commands evaluate Mega-MMDF, save predictions without metrics or features,
and write:

`logs/misaligned_test/{Model}_train_on_Mega-MMDF_shift_audio_1/predictions/Mega-MMDF.json`

Filter `misalignment_applied == true` to select shifted samples, then join aligned
and shifted predictions by `(path, sample_frame_range)`. The records also contain
the actual audio/video ranges. Labels remain zero because they describe source
authenticity.

Add `--metrics 'sample-level accuracy'` to evaluate cached predictions without
repeating inference when all requested outputs exist. This real-only protocol
cannot support ordinary two-class AUC/EER.

The [misalignment command guide](../../commands/misaligned.md) describes the
single-model invocation, aligned control, and eligible paired cohort. The full
seven-model list is in [commands/misaligned_test_commands.sh](../../commands/misaligned_test_commands.sh).

## SpeechForensics

The evaluation-only adapter has a separate configuration and full Fairseq
checkpoint:

```bash
python test.py --detector_path configs/misaligned_test/speechforensics_misaligned.yaml --weight_path weights/SpeechForensics/large_vox_iter5.pt --log_dir logs/misaligned_test/SpeechForensics_shift_audio_1 --test_datasets Mega-MMDF
```

Use a copied configuration with `shift: 0` and a separate output directory for
its aligned control. The [checkpoint guide](../../weights/README.md) gives the
download source; the [paired-misalignment analysis](../../analysis/README.md)
describes comparison inputs and figures.
