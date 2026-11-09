# Reduced-data training configurations

These 42 YAML files cover seven detectors at 1%, 2%, 5%, 10%, 20%, and 50% of the
Mega-MMDF development data. The ordinary full-data configurations provide the
100% baseline. Baseline uses the registry key `C3DResNet18+SEResNet+MLP`.

Each `{model}_train_on_megammdf_{percentage}.yaml` selects a manifest such as
`Mega-MMDF-1%` for training and the Mega-MMDF validation component. The other four
development validation datasets remain configured. Prepare the subset manifests
and their membership explicitly; percentage names do not construct the subsets.

## Run

The [training list](../../commands/small_scale_train_commands.txt) selects each run with
`--log_dir` and uses the boolean `--resume` flag to preserve files and restore
its latest full state when available. Start a fresh run with the configuration
directly, omitting `--resume`:

```bash
python train.py --detector_path configs/small_data_scale/avh_train_on_megammdf_1.yaml --log_dir "logs/small_scale_train/AVH_train_on_1%_Mega-MMDF"
```

The [test list](../../commands/small_scale_test_commands.txt) evaluates selected checkpoints
on the full named test sets, independently of the training fraction. Configure
paths and provide the initialization assets and final checkpoints first.
Training `--log_dir` is the exact run directory, as it is for testing. A fresh
training run clears that directory first; `--resume` preserves existing files.

The [scaling figure](../../analysis/data_scale/README.md) reads a 49-point snapshot,
including the 100% endpoints, or an explicit compatible workbook. To add a
fraction or detector, provide its subset manifests and configuration, then
update the command lists and plotting schema. Preserve the checkpoint-selection
protocol and account for changed optimization steps when comparing data scales.
The [data-scale command guide](../../commands/data_scale.md) collects the running
instructions.
