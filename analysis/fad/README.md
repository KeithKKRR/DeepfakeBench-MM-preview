# Fréchet Audio Distance

`run_fad.py` extracts VGGish audio features with `AudioVideoDataset` test-mode dense tiling, caches them, and compares real/fake audio distributions.

```bash
python analysis/fad/run_fad.py \
  --log_dir logs/fad_review \
  --datasets PolyGlotFake FakeAVCeleb IDForge Mega-MMDF
```

Dataset paths come from `configs/path.yaml`. The device is CUDA when available, otherwise CPU. Default non-overlapping 25-frame windows yield 1.0 s at 25 fps and one 128-dimensional VGGish embedding. `--num-frames` controls the window length; VGGish requires 15,600 audio samples for a full 96-frame log-mel example, so 24 frames/15,360 samples is insufficient.

The default local weight is `pretrained/vggish-10086976.pth`. Override it with `--vggish-weight /path/to/vggish-10086976.pth`. When available, the local checkpoint uses the vendored `torchvggish/` implementation from [harritaylor/torchvggish](https://github.com/harritaylor/torchvggish). A missing local checkpoint enables the existing `frechet_audio_distance` or `torch.hub` download fallback.

Temporary per-window NPY files are aggregated into `<log_dir>/features/<dataset>.pkl` and removed. Features are grouped by `audio_label`; datasets lacking a real or fake class are skipped with a logged reason. FAD uses the squared mean distance plus the covariance trace term `Tr(Sigma_real + Sigma_fake - 2 sqrtm(Sigma_real @ Sigma_fake))`.

Progress and per-dataset results go to `<log_dir>/log.txt`, followed by a final table. `--recompute` ignores existing feature pickles.
