# Missing-modality tests

The test-time mask represents unavailable audio by a zero waveform, or unavailable
video by black frames, before the model's representation and normalization steps.
No offline perturbation generation is needed.

```bash
python test.py \
  --detector_path configs/detectors/audio_video/baseline_train_on_megammdf.yaml \
  --weight_path /path/to/model_only_checkpoint.pth \
  --test_datasets Mega-MMDF \
  --robustness_type mask_audio \
  --log_dir logs/test/baseline_megammdf_mask_audio \
  --metrics 'video-level AUC'

python test.py \
  --detector_path configs/detectors/audio_video/baseline_train_on_megammdf.yaml \
  --weight_path /path/to/model_only_checkpoint.pth \
  --test_datasets Mega-MMDF \
  --robustness_type mask_video \
  --log_dir logs/test/baseline_megammdf_mask_video \
  --metrics 'video-level AUC'
```

Do not pass `--robustness_param` for modality masks. Each condition needs its own
test output directory, including the complete-input reference.

Saved test records contain effective labels after masking and the pre-mask
`original_audio_label`, `original_video_label`, and `original_label`. Removing a
forged modality can change the effective label. Select the intended original
authenticity groups when reproducing a RARV/RAFV/FARV/FAFV comparison; the command
does not automatically construct those separate cohorts.

Prediction-only export is available by omitting `--metrics`. The
[mask-analysis tools](../analysis/predicted_prob_histrogram/README.md) describe
the paired prediction inputs used for distributions and score changes.
