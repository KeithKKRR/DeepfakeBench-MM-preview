# Initialization and analysis weights

Download these assets separately and place them at the paths below, relative to
`pretrained/`. Final detector checkpoints belong in [weights](../weights/README.md).
Some detector constructors read initialization files before loading a final
checkpoint. FAD/FVD assets are needed only for uncached feature extraction.

## External assets

| Target relative to this directory | Publisher and acquisition | Preparation |
|---|---|---|
| `whisper/whisper_base.pt` | OpenAI; [Download](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt) | Rename multilingual base.pt; keep model_state_dict container. |
| `CSN/CSN_32x2_R101.pyth` | PyTorchVideo; [Download](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/CSN_32x2_R101.pyth) | Keep model_state container. |
| `MEFARG/au_112_12cls2.pth` | [FauForensics official repository](https://github.com/JWang-cs21/FauForensics); ME-GraphAU-based AU branch | AU pretraining checkpoint; exact download URL pending. |
| `jx_vit_base_p16_224-80ecf9dd.pth` | timm release; [Download](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) | Download exact file; do not substitute a different model variant. |
| `avhubert/self_large_vox_433h.pt` | AV-HuBERT authors; [Download](https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt) | Download exact file; do not substitute a different model variant. |
| `audiomae/pretrained.pth` | AudioMAE authors; [Download](https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link) | Use the Google Drive download; see stage-specific loading support below. |
| `marlin/marlin_vit_base_ytf.full.pt` | MARLIN authors; [Download](https://github.com/ControlNet/MARLIN/releases/download/model_v1/marlin_vit_base_ytf.full.pt) | Download exact file; do not substitute a different model variant. |
| `Cnn14_mAP=0.431.pth` | PANNs authors; [Download](https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1) | Download the checkpoint under the exact target filename. |
| `EfficientNet/efficientnet-b7-dcc49843.pth` | EfficientNet-PyTorch maintainer; [Download](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth) | Download exact file; do not substitute a different model variant. |
| `vggish-10086976.pth` | torchvggish maintainer; [Download](https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth) | PyTorch port, not a Google-native weight file; PCA file not needed for local FAD branch. |
| `i3d_torchscript.pt` | StyleGAN-V authors (FVD conversion); [Download](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1) | TorchScript feature extractor used for FVD. |

Baseline and FGMDF do not read an external initialization file in their current
constructors. Create the parent directories and save each binary under its
specified filename. Google Drive and Zenodo provide interactive download controls;
save the checkpoint rather than the HTML preview.

For direct downloads, run from the repository root:

```bash
mkdir -p pretrained/whisper pretrained/CSN
curl -L --fail 'https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt' -o pretrained/whisper/whisper_base.pt
curl -L --fail 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/CSN_32x2_R101.pyth' -o pretrained/CSN/CSN_32x2_R101.pyth
```

On Windows, use `curl.exe` and create the same parent directories first.

## Method-pretraining checkpoints

The AVFF and MCL method-stage checkpoints will be distributed separately; the
download repository link is pending. Expected inputs are:

- `avff/epoch100_step2020.pth`: optional AVFF method-pretraining checkpoint
  selected by the corresponding configuration.
- `MCL/MCL_pretrained_on_<dataset>.pth`: selected MCL stage-one image teacher
  for FakeAVCeleb, LAV-DF, PolyGlotFake, IDForge, or Mega-MMDF. Save the selected
  model-only checkpoint under this filename or update `stage1_ckpt` accordingly.

These stage files are separate from the external AudioMAE, MARLIN, EfficientNet,
and Cnn14 initialization assets.

## Directory structure

```text
pretrained/
  avhubert/self_large_vox_433h.pt
  audiomae/pretrained.pth
  marlin/marlin_vit_base_ytf.full.pt
  whisper/whisper_base.pt
  CSN/CSN_32x2_R101.pyth
  MEFARG/au_112_12cls2.pth
  jx_vit_base_p16_224-80ecf9dd.pth
  EfficientNet/efficientnet-b7-dcc49843.pth
  Cnn14_mAP=0.431.pth
  avff/epoch100_step2020.pth
  MCL/MCL_pretrained_on_FakeAVCeleb.pth
  MCL/MCL_pretrained_on_LAV-DF.pth
  MCL/MCL_pretrained_on_PolyGlotFake.pth
  MCL/MCL_pretrained_on_IDForge.pth
  MCL/MCL_pretrained_on_Mega-MMDF.pth
  vggish-10086976.pth
  i3d_torchscript.pt
```

## Known loading incompatibilities

**MEFARG:** the [official FauForensics loader](https://github.com/JWang-cs21/FauForensics/blob/main/utils/pretrain.py)
uses `au_112_12cls2.pth`; its exact public download link remains pending. The
related [ME-GraphAU weights](https://github.com/CVI-SZU/ME-GraphAU#pretrained-models)
contain three folds and are not verified substitutes. The available research
checkpoint has `module.`-prefixed keys, while the current bare MEFARG loader does
not strip the prefix, ignores incompatible keys under `strict=False`, and freezes
the module. Those prefixed keys therefore do not initialize the corresponding
unprefixed parameters. Loading a final full detector state is a separate step.

**AudioMAE:** [upstream loading code](https://github.com/facebookresearch/AudioMAE/blob/main/main_finetune_as.py#L355-L367)
extracts `checkpoint['model']`. `AVFF-Pretrain-Adv` accepts raw backbone state
dictionaries and checkpoint containers with a `model` or `state_dict` entry for
both AudioMAE and MARLIN initialization. It loads matching keys non-strictly
without rewriting parameter prefixes, rejects files with no matching keys, and
reports tensor shape mismatches through PyTorch's loader.

The non-adversarial `AVFF-Pretrain` and supervised `AVFF` constructors still pass
the outer checkpoint object directly to `load_state_dict(..., strict=False)`.
The official AudioMAE container therefore needs to be extracted for those
constructors. Supervised AVFF also rebuilds the audio module before optionally
loading the full AVFF stage checkpoint, so its initial audio load alone does not
persist.

Check missing/unexpected keys and tensor shapes when preparing these assets.
The historical timm tag identifies the FRADE download; retain the installed timm
version required by the code's APIs.

The [asset manifest](asset_manifest.json) lists all 17 target paths. Source details
are recorded in [FauForensics/FRADE sources](fau_frade_sources.json) and
[other asset sources](other_sources.json). Provider license and access conditions
apply to each checkpoint.
