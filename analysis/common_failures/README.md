# Common-failure analysis

The scripts collect shared errors from seven Vanilla complete-input Mega-MMDF prediction runs and optionally inspect the corresponding media.

## Collect and report

```bash
python analysis/common_failures/collect.py --root logs/test --output analysis/common_failures/outputs
python analysis/common_failures/report.py
```

`collect.py` reads `{Model}_train_on_Mega-MMDF/predictions/Mega-MMDF.json` below `--root`. Models are AVFF, AVH, Baseline, FGMDF, FRADE, FauForensics, and MCL. Native records require path, frame range, overall/audio/video labels, and a finite `[0,1]` score. Complete native predictions are external inputs.

Video identity is `dirname(path)`; labels are ORed and scores maximized over evaluated windows. Shared errors require matching labels among compared models and retain each detector's coverage. This analysis uses fake iff `prob >= 0.5`. Identical native-window errors and overlapping temporal intervals are catalogued separately.

Outputs include `summary.json` with source hashes, `video_predictions.csv.gz`, `video_cases.csv.gz`, and native-window failure/exact-overlap/temporal-overlap CSVs. `report.py` reads this module's `outputs/` and writes all-seven/majority lists and HTML/Markdown reports.

## Inspect selected media

After reporting, select cases from `outputs/video_common_all7.csv`:

```bash
python analysis/common_failures/inspect_media_cases.py
python analysis/common_failures/inspect_media_cases.py \
  --manifest analysis/common_failures/inspection_outputs/selected_cases.json \
  --out analysis/common_failures/inspection_outputs \
  --preprocessed-root /path/to/preprocessed/audio-video \
  --original-root /path/to/original
```

Selection uses the largest and median minimum shared-error margin within each represented pipeline. With explicit media roots, inspection writes contact sheets and `media_audit.json` containing resolved paths, frame indices, and audio summaries. Supply media and FFmpeg/ffprobe plus the image/audio dependencies.

For 5-FPS chronological storyboards, cropped audio, playback files, and signal checks:

```bash
python analysis/common_failures/inspect_temporal_cases.py \
  analysis/common_failures/inspection_outputs/media_audit.json \
  analysis/common_failures/inspection_outputs/temporal
```

`report_media_inspection.py` renders manual observations tied to a specific 12-case audit. Review and replace those observations before using it with a different selection. Extending the detector set requires updating the collector's model list and the report coverage rules together.
