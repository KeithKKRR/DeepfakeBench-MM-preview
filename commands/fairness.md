# Demographic fairness evaluation

Fairness metrics require a `demographic_group` string in each evaluated prediction
record. Prepare these annotations in the dataset JSON **before inference**; the
dataset loader and `test.py` will carry them into the predictions.

## 1. Prepare the dataset annotations

The current preprocessors do not automatically add this field. For Mega-MMDF,
the original clip paths already contain dataset-supplied directories such as
`Female_Black`; FakeAVCeleb paths contain supplied race/gender directories such
as `African/men`. The tool below reads these existing labels. It does not estimate
demographic attributes from faces, voices, or other media content.

Run from the repository root after preprocessing and manifest assembly:

```bash
python tools/add_demographic_groups.py --dataset Mega-MMDF --split test --in_place
```

By default, this reads `json_dir/audio-video/Mega-MMDF.json` using
[`configs/path.yaml`](../configs/path.yaml), annotates only its test split, and
preserves all other fields, record order, and splits. `--in_place` explicitly
updates that manifest and retains its first backup as `Mega-MMDF.json.bak`.
An existing backup is not overwritten.

The built-in conventions match the existing research scripts:

| Dataset | Annotation read from `original_path` | Stored group |
|---|---|---|
| Mega-MMDF | `Female_Black` directory | `Black - Female` |
| Mega-MMDF | `Male_White` directory | `White - Male` |
| FakeAVCeleb | `African/men` directories | `African - men` |
| FakeAVCeleb | `Asian (East)/women` directories | `Asian (East) - women` |

For FakeAVCeleb, run the same command with `--dataset FakeAVCeleb`. Other input
locations can be selected with `--input_json`. Use `--split train`, `val`, or
`all` only when those splits also need annotations.

Existing nonempty group labels are preserved by default. `--overwrite` explicitly
recomputes them from the selected annotation source. Missing annotations for any
selected, unannotated record cause an error before writing, rather than silently
evaluating a demographic subset. If a manifest is reassembled by preprocessing,
repeat this annotation step.

## Other datasets and explicit group mappings

For another dataset, provide a JSON object mapping the exact `original_path`
values to known group labels:

```json
{
  "ExampleDataset/real/clip001.mp4": "Group A",
  "ExampleDataset/fake/clip002.mp4": "Group B"
}
```

Then run:

```bash
python tools/add_demographic_groups.py \
  --dataset ExampleDataset \
  --mapping_json /path/to/known_demographic_groups.json \
  --split test --in_place
```

The mapping must cover the selected records that do not already have a group.
Backslashes are normalized to slashes; basenames and path prefixes are not
guessed. To join a different record field, select it explicitly, for example
`--key_field path`. Use dataset-provided or otherwise established annotations.
Group names remain dataset-specific and are not harmonized by this tool.

`--output_json /path/to/output.json` is an alternative to `--in_place`. Before
testing, place that output at the configured dataset manifest location. The tool
does not rewrite `configs/path.yaml` or switch the loader to a new filename.

## 2. Run inference and fairness metrics

```bash
python test.py \
  --detector_path configs/detectors/audio_video/baseline_train_on_megammdf.yaml \
  --weight_path /path/to/model_only_checkpoint.pth \
  --test_datasets Mega-MMDF \
  --log_dir ./logs/test/baseline_megammdf_fairness \
  --metrics 'video-level DP' 'video-level EO' 'video-level OAE' 'video-level MEO'
```

Use a **new test output directory** if existing prediction files were generated
without `demographic_group`. Test output caching would otherwise reuse those
unannotated predictions. Once annotated predictions have been saved, additional
metrics can reuse the same output directory and unchanged checkpoint/settings.

Fairness evaluation needs at least two annotated groups. Metric definitions,
aggregation, thresholds, and handling of undefined class/group combinations are
documented in the [metric guide](../metrics/README.md).
