# Fréchet Video Distance

`run_fvd.py` extracts StyleGAN-V I3D features with `AudioVideoDataset` test-mode dense tiling, caches them, and compares real/fake video distributions.

```bash
python analysis/fvd/run_fvd.py \
  --log_dir logs/fvd_review \
  --datasets PolyGlotFake FakeAVCeleb IDForge Mega-MMDF
```

Dataset paths come from `configs/path.yaml`. The device is CUDA when available, otherwise CPU. Default non-overlapping 16-frame clips are resized to 256 × 256 RGB and passed to I3D with `rescale=True`, `resize=True`, and `return_features=True`, producing 400-dimensional features.

The default detector is `pretrained/i3d_torchscript.pt`, the TorchScript model from [StyleGAN-V](https://github.com/universome/stylegan-v). Override it with `--detector /path/to/i3d_torchscript.pt`. A missing local model enables the existing download fallback.

Temporary per-clip NPY files are aggregated into `<log_dir>/features/<dataset>.pkl` and removed. Features are grouped by `video_label`; datasets lacking a real or fake class are skipped with a logged reason. FVD uses the squared mean distance plus the covariance trace term `Tr(Sigma_real + Sigma_fake - 2 sqrtm(Sigma_real @ Sigma_fake))`.

Progress and per-dataset results go to `<log_dir>/log.txt`, followed by a final table. `--recompute` ignores existing feature pickles.
