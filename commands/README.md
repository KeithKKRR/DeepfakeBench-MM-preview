# Running experiments

Run these examples from the repository root after configuring
[paths](../configs/path.yaml), preprocessing the required datasets, and preparing
the model's [initialization weights](../pretrained/README.md) and
[detector checkpoints](../weights/README.md).

| Task | Instructions |
|---|---|
| Basic training and resume | [train.md](train.md) |
| Basic testing and metric-only reruns | [test.md](test.md) |
| Demographic fairness | [fairness.md](fairness.md) |
| Offline post-processing robustness | [robustness.md](robustness.md) |
| Missing-modality tests | [mask_modality.md](mask_modality.md) |
| Real-only audiovisual window misalignment | [misaligned.md](misaligned.md) |
| Single-pipeline training and cross-pipeline evaluation | [cross_pipeline.md](cross_pipeline.md) |
| Reduced-data training and evaluation | [data_scale.md](data_scale.md) |

`--log_dir` selects an exact run/output directory. Training without `--resume`
clears the selected run before starting; the boolean `--resume` flag preserves
it and restores a full state when available. Testing reuses existing outputs,
so select a new output directory when the checkpoint or data setting changes.

The full command lists retain their named checkpoint and dataset requirements.
They do not download weights, generate dataset subsets, or launch preprocessing
automatically.
