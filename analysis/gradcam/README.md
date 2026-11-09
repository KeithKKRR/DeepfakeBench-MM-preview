# Grad-CAM

This module computes attribution maps and renders audio/video examples.

- `engine.py` computes raw maps at a selected layer's native resolution.
- `render.py` renders flat images, overlays, grayscale maps, and video through the [shared presenter](../utils/README.md).
- `audio_video_detectors/baseline_gradcam.py` adapts `C3DResNet18+SEResNet+MLP`, including its visual/audio resize and coordinate mapping. Other detector adapters are not supplied.

```bash
python analysis/gradcam/audio_video_detectors/baseline_gradcam.py \
  --detector_path configs/detectors/audio_video/baseline_train_on_megammdf.yaml \
  --weight_path weights/Baseline_train_on_Mega-MMDF.pth \
  --dataset_name Mega-MMDF \
  --sample_index 0 \
  --display_mode image
```

Supply the detector checkpoint and media paths configured for the dataset. `--display_mode` accepts `image` (presenter layout), `overlay` (colored heatmaps over frames/audio), `gray` (attribution only), or `video` (MP4 with a moving audio pointer). Select target layers with `--visual_layer` and `--audio_layer`.
