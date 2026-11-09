# Configurations

Configurations connect a detector to its input representation, datasets, losses,
and training settings. Filesystem roots are configured separately in
[path.yaml](path.yaml). Each detector YAML is a complete configuration; the
entrypoints do not inherit or merge it with another detector YAML.

| Path | Purpose |
|---|---|
| [path.yaml](path.yaml) | Raw media, prepared media, metadata, and logging roots |
| [detectors/audio_video](detectors/audio_video/README.md) | Detector configurations and detailed field reference |
| [exp](exp/README.md) | 66 single-pipeline training and transfer configurations |
| [small_data_scale](small_data_scale/README.md) | 42 reduced-data training configurations |
| [misaligned_test](misaligned_test/README.md) | Real-only correspondence controls |

## Configure filesystem paths

Edit `path.yaml` for your filesystem:

```yaml
original_dir: ./data/original
preprocessed_dir: ./data/preprocessed
json_dir: ./data/json
log_dir: ./logs
```

Prepared metadata lives at `json_dir/audio-video/{dataset_name}.json`. Its media
paths are relative to `preprocessed_dir/audio-video/`; raw `original_path` values
are relative to `original_dir`. See the
[preprocessing guide](../preprocess/audio_video/README.md) for producing these
files. The Python identifier `audio_video` and data directory `audio-video` use
different separators.

## Write a detector configuration

Start from an existing configuration for the detector you want to use:

```bash
cp configs/detectors/audio_video/baseline_train_on_megammdf.yaml configs/detectors/audio_video/my_detector.yaml
```

Then update the following components:

1. **Model and losses.** `model_name` selects the registered detector;
   `model_config` is passed to its constructor. Preserve that detector's expected
   loss names, initialization weights, and input representation. For a new model,
   follow the [detector interface and registration guide](../detectors/README.md).
2. **Data.** `data_type` selects the dataset class, and `data_config` defines
   sampling and audio/video representations. `train_datasets` and `val_datasets`
   name prepared JSON manifests. See the
   [dataset guide](../datasets/audio_video/README.md) and
   [augmentation guide](../datasets/utils/augmentations/README.md).
3. **Training and validation.** Set `train_type`, batch sizes, `num_epochs`, and
   the validation metric/frequency. The [trainer guide](../trainers/README.md)
   explains the basic and alternating loops, checkpointing, and resume behavior;
   the [metric guide](../metrics/README.md) lists validation metric names.
4. **Optimization.** Set the [optimizer](../optimizers/README.md) and optional
   [scheduler](../schedulers/README.md). Both use `type` and `params`; an enabled
   scheduler also requires `interval: step` or `interval: epoch`.

The example below trains and validates Baseline on Mega-MMDF with unweighted
cross-entropy. The supplied benchmark configurations retain their own loss
settings and validation datasets.

```yaml
model_name: C3DResNet18+SEResNet+MLP
model_config:
  losses:
    - name: CE
      func: cross_entropy
      params: {}

data_type: audio_video
data_config:
  frame_size: 224
  frame_mean: [0.485, 0.456, 0.406]
  frame_std: [0.229, 0.224, 0.225]
  num_frames: 16
  stride: 1
  spectrogram_type: MFCC
  spectrogram_backend: python_speech_features
num_workers: 8
aug_config: default

train_type: basic
train_datasets: [Mega-MMDF]
train_batch_size: 64
val_datasets: [Mega-MMDF]
val_batch_size: 64
metric_scoring: sample-level AUC
val_per_epoch: 2
num_epochs: 10
optimizer:
  type: Adam
  params:
    lr: 0.001
scheduler: null
```

For field definitions, method-specific representations, and initialization
settings, see the [audio-video configuration reference](detectors/audio_video/README.md).
Loss construction is documented in the [loss guide](../losses/README.md).

For alternating optimization, set `train_type: alternative` and define
`train_stages`, with each stage's `name`, `update_step`, `optimizer`, and optional
`scheduler`. The detector must implement the corresponding stage interface.
See the [trainer guide](../trainers/README.md) and the supplied
[AVFF adversarial pretraining configuration](detectors/audio_video/avff_pretrain_adv.yaml).

## Run the configuration

Run from the repository root after saving the example as `my_detector.yaml`:

```bash
python train.py --detector_path configs/detectors/audio_video/my_detector.yaml --log_dir logs/train/my_detector
```

`--log_dir` selects the exact run directory. Without `--resume`, that directory
is cleared before training. Add the boolean `--resume` flag with the same
configuration and directory to restore the latest full training state.
`--seed` defaults to 1024. Distributed settings are supplied by the entrypoint,
rather than written into the detector YAML. See the
[training commands](../commands/train.md) for additional examples.

Testing uses the corresponding detector configuration and a model-only checkpoint.
Dataset names, the checkpoint, output directory, and optional metrics are
selected with [test.py](../test.py) arguments; `--test_batch_size` overrides the
configuration's `val_batch_size`. See the [test commands](../commands/test.md).
Use a new output directory when the checkpoint or evaluation settings change,
because existing prediction/feature files are reused by existence.

## Specialized evaluations

The [commands guide](../commands/README.md) collects fairness, robustness,
missing-modality, misalignment, cross-pipeline, and data-scale procedures.
Those guides explain the additional manifests, annotations, or prepared files
required by each evaluation. Dataset-specific sampling and intervention fields
are described in the [audio-video configuration reference](detectors/audio_video/README.md).
