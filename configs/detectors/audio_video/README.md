# Audio-video detector configurations

These YAML files bind an integrated detector to its preprocessing representation, losses, datasets and optimization loop. Use the [method-to-key table](../../../detectors/audio_video/README.md) to select the right configuration. Each detector and dataset reads its own configuration fields.

## Model and loss fields

| Field | Meaning |
|---|---|
| `model_name` | Exact key in [DETECTOR](../../../tools/registry.py) |
| `model_config` | Dictionary passed to the detector constructor |
| `model_config.losses` | List of loss entries consumed by the detector |
| `losses[].name` | Detector-specific lookup/logging name such as `CE`, `CMC` or `L_nce` |
| `losses[].func` | Registered [loss function](../../../losses/README.md) name |
| `losses[].params` | Constructor keyword arguments |
| `losses[].weight` | Detector-managed loss multiplier, defaulted by most builders to 1; application depends on `get_losses` |

Model-specific initialization includes AVH's `pretrained`, MCL's `audio_ckpt`/`stage1_ckpt`, and AVFF's optional method-level `pretrained` plus `backbone.audio`/`backbone.video` settings. AVFF backbone settings use `name`, `version`, optional `params`, and **`weight`** for the initialization path.

Some methods also read fixed initialization paths from their source. These assets are loaded before the final `test.py --weight_path` checkpoint. [Backbone documentation](../../../backbones/README.md) explains registry and directly imported encoders.

## Dataset fields

`data_type` selects [choose_dataset](../../../datasets/builder.py): ordinary methods use `audio_video`; the real-only correspondence control uses `audio_video_misaligned_real`.

| `data_config` field | Implemented behavior / default |
|---|---|
| `frame_size` | Required square frame resize size |
| `num_frames` | Required number of returned video frames |
| `stride` | Required fixed frame stride; window span is `num_frames * stride` |
| `color_mode` | `RGB` by default; `Gray` requests grayscale |
| `frame_mean`, `frame_std` | Video normalization; defaults 0 and 1 |
| `with_landmark` | Load and transform aligned landmarks when true; singular spelling |
| `audio_sampling_rate` | Waveform loading rate, default 16000 |
| `remove_dc_offset` | Subtract waveform mean when true; default false |
| `spectrogram_type` | Leave unset/empty for raw waveform; otherwise request a supported conversion |
| `spectrogram_backend` | Conversion backend paired with the requested type |
| `spectrogram_params` | Backend keyword arguments; default empty mapping |
| `audio_mean`, `audio_std` | Spectrogram normalization; defaults 0 and 1 |
| `max_samples_per_data` | Train/validation cap per metadata record, default 2; not a test cap |
| `only_real_data` | Filter source-real records in train/validation only, default false |
| `misalignment` | Test-only whole-window correspondence settings for the specialized dataset |

`spectrogram_type` supports `STFT`, `Mel`, `Log-Mel` and `MFCC`, but not every type/backend combination is implemented. Refer to [convert_audio_spectrogram](../../../datasets/utils/data_helpers/audio_helpers.py) and retain each method's supplied representation. For example, AVH uses `python_speech_features`, AVFF uses Kaldi log-mel, FauForensics uses torchaudio log-mel, FRADE uses librosa log-mel, and FGMDF/MCL use raw waveforms at their configured rates.

Window selection, effective labels, augmentation and intervention details are described in the [audio-video dataset guide](../../../datasets/audio_video/README.md). Putting `only_real_data` at the top level does not activate it.

## Ordinary training fields

| Field | Use |
|---|---|
| `train_type: basic` | Select `BasicTrainer` |
| `train_datasets` | Names whose `train` lists are concatenated |
| `train_batch_size` | Training loader batch size |
| `val_datasets` | Validation dataset names; may be absent/empty for a pretraining run |
| `val_batch_size` | Validation batch size; also default test batch size |
| `metric_scoring` | Validation metric name such as `sample-level AUC` or `loss` |
| `val_every_epoch` | Validate every this many epochs, default 1 |
| `val_per_epoch` | Validation points per selected epoch, default 1 |
| `record_per_epoch` | Loss-log points per epoch, default 10 |
| `optimizer` | [Optimizer](../../../optimizers/README.md) `type` and `params` |
| `scheduler` | Optional [scheduler](../../../schedulers/README.md) mapping with `type`, `interval`, and `params`, or null |
| `num_epochs` | Number of training epochs |
| `num_workers` | Loader workers, default 4; current training loaders use persistent workers |

When `val_datasets` is nonempty, supply the validation batch/metric settings used by the trainer. An enabled scheduler requires `interval: step` for advancement after each optimizer update or `interval: epoch` for advancement after each complete epoch. Scheduler parameters use that interval's units.

## Alternating stages and inference

`train_type: alternative` selects `AlternativeTrainer`. Its `train_stages` entries define `name`, `update_step` (default 1), `optimizer`, and optional `scheduler`. AVFF adversarial pretraining uses `generator` and `discriminator`. These are alternating batch updates within one run, with separate configurations for pretraining and fine-tuning.

Place each stage's `scheduler` alongside `optimizer`, not inside it. Its `interval`
selects updates of that stage's optimizer or complete epochs. For step-based
custom schedules, the trainer counts the actual stage updates, including the
tail of the stage cycle, and supplies the corresponding total length. See the
[trainer guide](../../../trainers/README.md) for checkpoint and resume behavior.

Inference uses `model_name`, `model_config`, `data_type`, `data_config`, `num_workers` and, unless overridden, `val_batch_size`. A test-only YAML can omit training-loop fields such as `train_type`. Dataset names, final checkpoint, output directory, metrics, requested features and robustness settings are supplied by [test.py](../../../test.py). Cached outputs are reused by existence, so a protocol change requires a new output directory.

## SpeechForensics test configuration

`speechforensics.yaml` is evaluation-only and has no training section. It loads
25 grayscale frames with 68-point landmarks, uses the supplied mouth crop and
python_speech_features log-filterbanks, and restores the full Fairseq checkpoint
from `test.py --weight_path`. See the [detector guide](../../../detectors/audio_video/README.md) and
[checkpoint guide](../../../weights/README.md) for the full-checkpoint loader.
