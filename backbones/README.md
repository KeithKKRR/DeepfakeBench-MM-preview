# Backbones

This package contains benchmark-facing encoder adapters and a vendored AV-HuBERT/Fairseq dependency. Some methods define their networks inside the detector module; see the [detector guide](../detectors/audio_video/README.md).

## Encoder adapters

| Adapter | Construction | Caller |
|---|---|---|
| [AudioMAEBackbone](audio_backbones/audiomae_backbone.py) | `BACKBONE['AudioMAE']` | AVFF classification and pretraining |
| [MARLINBackbone](video_backbones/marlin_backbone.py) | `BACKBONE['MARLIN']` | AVFF classification and pretraining |
| [Cnn14](audio_backbones/Cnn14_backbone.py) | Direct class import | MCL |
| [ViViT](video_backbones/ViViT_backbone.py) | Direct class import | MCL |

The [package initializer](__init__.py) registers AudioMAE and MARLIN through [BACKBONE](../tools/registry.py). Cnn14 and ViViT are imported directly by MCL.

### AudioMAE and MARLIN

Both registered adapters expose `predefined_versions`. AVFF selects a `model_config.backbone.<modality>.version`, applies explicit `params` overrides and constructs the chosen registry class. The detector reads the optional `weight` path. See the [AVFF configuration](../configs/detectors/audio_video/avff_train_on_megammdf.yaml) and [builder implementation](../detectors/audio_video/avff_detector.py).

AudioMAE presets are `mae_vit_small_patch16`, `mae_vit_base_patch16`, `mae_vit_large_patch16` and `mae_vit_huge_patch14`. `forward_encoder(x, mask_ratio, mask_2d=False)` returns encoded tokens, a removal mask, restoration indices and a placeholder. `forward_decoder` returns reconstructed patches and placeholders. The full `forward` additionally computes reconstruction loss. AVFF constructs a one-channel spectrogram encoder and calls the encoder/decoder APIs directly.

MARLIN presets are `marlin_vit_small`, `marlin_vit_base` and `marlin_vit_large`. `extract_features(x, keep_seq=True)` returns video token features; `keep_seq=False` averages them. `forward(x, mask)` performs masked reconstruction and requires `as_feature_extractor=False`. AVFF constructs this reconstruction-capable form and also uses its encoder directly.

### Cnn14 and ViViT

Cnn14 computes waveform spectrograms and log-mel features with `torchlibrosa`, then returns `clipwise_output` and `embedding`. MCL builds it for 32 kHz audio and a 527-class audio-tagging head. Its checkpoint loader expects the audio initialization under the checkpoint's `model` key.

ViViT returns `(logits, pooled_feature)`. Its learned temporal position embeddings depend on `num_frames`; MCL constructs a 30-frame student. Detailed input layouts and method outputs are documented in the adapter methods.

## Vendored dependency

[audio_video_backbones/av_hubert/](audio_video_backbones/av_hubert/) contains upstream AV-HuBERT and Fairseq packages, task/model registration modules and configuration resources. AVH loads a fine-tuned encoder from this dependency; SpeechForensics restores its full speech checkpoint through the same dependency. These packages are checkpoint infrastructure rather than `BACKBONE` registry adapters.

The detector adapters add the vendored directory to `sys.path` before importing its top-level `fairseq` and `avhubert` packages. Imports register models/tasks used by checkpoint restoration. Preserve the directory structure, required resources and upstream notices when packaging it.

## Adding an encoder

1. Choose registry construction or a direct import according to the caller's needs.
2. For registry construction, use `@BACKBONE.register_module(module_name='YourBackbone')` and import the class in the package initializer.
3. Match the caller's feature and reconstruction APIs. AVFF expects AudioMAE/MARLIN-specific token operations in addition to a registry name.
4. Document inputs, constructor settings and checkpoint loading with the detector configuration, and retain source provenance with adapted code.
