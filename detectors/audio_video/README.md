# Audio-video detectors

Each constructor receives `model_config`; the dataset receives the separate `data_config`. Use the matching configuration for a method's video sampling and audio representation. See the [configuration guide](../../configs/detectors/audio_video/README.md) and [shared detector interface](../README.md).

| Method / stage | Registered `model_name` | Class | Example configuration |
|---|---|---|---|
| Baseline | `C3DResNet18+SEResNet+MLP` | [C3DResNet18_SEResNet_MLP](c3dresnet18_seresnet_detector.py) | [Mega-MMDF](../../configs/detectors/audio_video/baseline_train_on_megammdf.yaml) |
| AVFF classification | `AVFF` | [AVFFDetector](avff_detector.py) | [Mega-MMDF](../../configs/detectors/audio_video/avff_train_on_megammdf.yaml) |
| AVFF reconstruction pretraining | `AVFF-Pretrain` | [AVFFPretrain](avff_detector.py) | [Pretraining](../../configs/detectors/audio_video/avff_pretrain.yaml) |
| AVFF adversarial pretraining | `AVFF-Pretrain-Adv` | [AVFFPretrainAdv](avff_detector.py) | [Adversarial pretraining](../../configs/detectors/audio_video/avff_pretrain_adv.yaml) |
| AVH | `AVH` | [AVHDetector](avh_detector.py) | [Mega-MMDF](../../configs/detectors/audio_video/avh_train_on_megammdf.yaml) |
| FauForensics | `FauForensics` | [FauForensicsDetector](fauforensics_detector.py) | [Mega-MMDF](../../configs/detectors/audio_video/fauforensics_train_on_megammdf.yaml) |
| FGMDF | `FGMDF` | [FGMDFDetector](fgmdf_detector.py) | [Mega-MMDF](../../configs/detectors/audio_video/fgmdf_train_on_megammdf.yaml) |
| FRADE | `FRADE` | [FRADEDetector](frade_detector.py) | [Mega-MMDF](../../configs/detectors/audio_video/frade_train_on_megammdf.yaml) |
| MCL student training | `MCL` | [MCLStage2Detector](mcl_detector.py) | [Mega-MMDF](../../configs/detectors/audio_video/mcl_train_on_megammdf.yaml) |
| MCL image-teacher pretraining | `MCL-Pretrain` | [MCLStage1Detector](mcl_detector.py) | [Mega-MMDF pretraining](../../configs/detectors/audio_video/mcl_pretrain_on_megammdf.yaml) |
| SpeechForensics evaluation | `SpeechForensics` | [SpeechForensicsDetector](speechforensics_detector.py) | [Evaluation](../../configs/detectors/audio_video/speechforensics.yaml) |

## Shared use

For supervised methods, call `model(data_dict)` and then `model.get_losses(data_dict, predictions)`. Evaluation uses `model(data_dict, inference=True)`. Classification outputs include `cls` and the fake score `prob`; extra logits and feature keys are listed below. Detailed tensor layouts are documented on each adapter's interface methods.

Video enters the adapter in `(B, C, T, H, W)` order. Audio representation and required label fields vary by method. Several adapters replace prepared tensors in `data_dict` in place, so prepare a fresh batch when comparing independent method calls.

## Baseline

The baseline combines a 3D ResNet video encoder, SE-ResNet audio encoder and MLP. The supplied configuration uses 16 RGB frames and MFCC audio.

- `features(data_dict)` returns temporal `video` and `audio` sequences; `classifier(features)` mean-pools them and returns logits and pooled features.
- `forward` returns `cls`, `prob`, `audio_feature`, `video_feature` and `fusion_feature`. `get_losses` uses the overall `label` with `CE`.
- Both encoders are built in the constructor; `build_backbone` is an unused interface hook. There is no separate initialization checkpoint in this adapter.

## AVFF

AVFF combines AudioMAE and MARLIN tokens through audio-to-video and video-to-audio networks. The supplied configurations use 16 video frames at stride 5 and 128-bin Kaldi log-mel audio. The adapter pads or truncates audio to 1024 time steps.

- `AVFFDetector.features` returns `audio`, `video` and `fused`; `classifier` returns two-class logits. `forward` exposes `cls`, `prob` and the three modality/fusion feature keys. `get_losses` applies `CE` to the overall `label`.
- `AVFFPretrain.features` and `forward` return mean embeddings, masked targets and masked/unmasked reconstructions. `get_losses` combines `Lc` and `Lrec`. Complementary masks are sampled on every call, including `inference=True`; this stage has no classification score.
- `AVFFPretrainAdv` adds discriminator patch scores to the reconstruction outputs; it has no `cls` or `prob` classification score. `get_stage_parameters('generator')` selects both complete backbones, including their decoders, and the cross-modal networks; `'discriminator'` selects the two discriminators. Call `model(data_dict, stage_name=stage)` and `get_losses(data_dict, pred_dict, stage=stage)` with `'generator'` or `'discriminator'`. Omitting the stage uses the generator objective for validation.
- Generator updates differentiate through fixed discriminator weights. Discriminator updates use reconstructions computed without a generator gradient graph, treating masked reconstructions as fake and unmasked reconstructions as real. Loss terms use their configured weights. This stage prepares audio locally without replacing the batch tensor, restores AudioMAE tokens to temporal order before complementary masking, and applies fusion attention independently within each sample.

Launch adversarial pretraining from the repository root:

```bash
python train.py --detector_path configs/detectors/audio_video/avff_pretrain_adv.yaml
```

The backbone configuration selects each registry `name`, optional `version`, constructor `params` and initialization `weight`. Supervised AVFF also accepts a full-method `pretrained` state dictionary. These constructor inputs are separate from a final evaluation checkpoint. See the [backbone guide](../../backbones/README.md) for encoder APIs.

## AVH

AVH extracts separate frozen AV-HuBERT audio/video streams and trains an MLP. The supplied configuration uses 25 grayscale frames and the `python_speech_features` log-filterbank backend. Video uses a fixed mouth crop within aligned 256-pixel frames; audio frames are stacked in groups of four.

- `features` returns `video` and `audio` sequences. `classifier` returns a pooled scalar score and per-frame fused features.
- `forward` returns symmetric two-class `cls`, fake-class `prob` and audio/video/fusion features. `get_losses` uses overall-label `CE`.
- `model_config.pretrained` is a Fairseq checkpoint used to construct the AV-HuBERT encoder. A final AVH detector state dictionary also contains the trained MLP.

## FauForensics

FauForensics defines its Whisper encoder, CSN network and MEFARG facial-action network in the detector module. The supplied configuration uses 25 frames at 112 pixels and 80-bin torchaudio log-mel audio.

- `features` returns joint, audio and visual token-pair similarity features. `classifier` returns `total`, `audio` and `visual` logits.
- `forward` exposes `cls`, `prob`, `a_logits`, `v_logits`, `av_logits` and modality/fusion feature keys.
- `get_losses` combines weighted `CE_AV`, `CE_A` and `CE_V` using `label`, `audio_label` and `video_label` respectively.
- Construction loads `pretrained/whisper/whisper_base.pt`, `pretrained/CSN/CSN_32x2_R101.pyth` and `pretrained/MEFARG/au_112_12cls2.pth`. MEFARG parameters are frozen. The standalone Whisper source directory is not imported by this adapter.

## FGMDF

FGMDF combines raw audio and video features through graph attention. The supplied configuration provides 100 video frames. `forward` selects 40 frames using random segment offsets in training and fixed uniform indices in inference.

- `features` consumes the prepared channel-stacked video and returns `video_feature`/`audio_feature`; `classifier` returns video, audio and cross-modal logits.
- `forward` returns the logit dictionary as `cls`, the cross-modal fake score as `prob`, and modality/fusion features.
- `get_losses` sums `video_modality_CE`, `audio_modality_CE` and `cross_modality_CE`, using video, audio and overall authenticity labels respectively.
- The benchmark configuration uses two classes. The graph network is constructed directly; there is no separate initialization checkpoint in the adapter.

## FRADE

FRADE uses a timm ViT and the adapter/cross-modal networks defined in its module. The supplied configuration uses 16 RGB frames and 80-bin librosa log-mel audio. `features` takes the first 64 audio steps, expands them to three channels, and returns the head's logits and features.

- `forward` returns `cls`, `prob` and audio/video/fusion feature keys. Classification is included in `features`; `classifier` is an unused interface hook.
- `get_losses` combines configured `CE`, `CMC` and `CC` weights with the overall authenticity label.
- Construction reads `pretrained/jx_vit_base_p16_224-80ecf9dd.pth` through timm before a final detector checkpoint is loaded.

## MCL

MCL first trains an EfficientNet image teacher and then a ViViT video student with audio features. Both stages choose a random teacher frame during training and the middle frame during inference. Frame supervision uses that selected frame's `frame_label`.

- `MCLStage1Detector.features` returns image logits, features and `frame_idx`. `forward` exposes `cls`, `prob` and `frame_idx`; `get_losses` applies `CE` to the selected frame label.
- `MCLStage2Detector.features` returns student, teacher and composed features. `classifier` classifies the noise-filtered fusion feature. `forward` exposes `cls`, `prob`, `frame_idx` and auxiliary feature/logit pairs; see its docstring and return dictionary for the complete keys.
- Stage-two `get_losses` combines the configured unimodal, contrastive, JSD and fusion terms using the relevant overall, audio, video and frame labels.
- The video student is constructed for 30 frames. Audio uses raw waveforms processed by Cnn14. Keep the data configuration consistent with these inputs.
- Both stages initialize EfficientNet from `pretrained/EfficientNet/efficientnet-b7-dcc49843.pth`. Stage two also reads `stage1_ckpt` as an image-teacher state dictionary and the `model` entry in the `audio_ckpt` file as Cnn14 initialization. The image teacher stays frozen and in evaluation mode during stage two.

## SpeechForensics evaluation

SpeechForensics is an evaluation-only adapter using a frozen AV-HuBERT speech checkpoint. `test.py` calls `load_checkpoint(weight_path)` to restore the Fairseq model and saved task/model configuration, then freezes it. See the [checkpoint guide](../../weights/README.md).

```bash
python test.py --detector_path configs/detectors/audio_video/speechforensics.yaml --weight_path weights/SpeechForensics/large_vox_iter5.pt --log_dir logs/test/SpeechForensics --test_datasets FakeAVCeleb LAV-DF PolyGlotFake IDForge Mega-MMDF TalkingHeadBench DigiFakeAV MMDF --metrics 'video-level AUC' 'video-level AP'
```

The supplied configuration uses 25-frame grayscale windows, aligned 68-point landmarks and 26-bin audio log-filterbanks. The adapter extracts a landmark-centered 96x96 mouth region and center-crops it to 88x88. Audio frames are stacked in groups of four and layer-normalized.

- Call `forward(data_dict, inference=True)` after checkpoint loading. `features` returns separate audio/video sequences.
- `classifier` compares zero-padded offsets from `-offset_window` through `+offset_window` (default 15). It converts the maximum mean cosine similarity to a bounded fake score with `(1 - similarity) / 2`.
- Outputs are `cls` class scores, `prob`, the matching `score`, `audio_feature` and `video_feature`. This adapter supplies no fusion feature. Its `cls` values are scores rather than unnormalized logits.
- `train` keeps the backbone in evaluation mode. A call without `inference=True` raises; this adapter is not a training entry point.

## Checkpoints and extension

For methods other than SpeechForensics, `test.py --weight_path` consumes a model state dictionary. A trainer state containing optimizer/epoch fields is a separate resume artifact. Detector construction can read initialization files before the final model checkpoint is restored; relative paths resolve from the working directory.

To add a method, follow the [shared interface](../README.md), register and import the adapter, and supply matching preprocessing and label configuration. Qwen2.5-Omni and VideoLLaMA2.1 use external inference implementations; support for their saved predictions in analysis tools does not register a detector here.
