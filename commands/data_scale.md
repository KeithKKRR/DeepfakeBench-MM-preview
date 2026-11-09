# Reduced-data training and evaluation

Prepare the reduced-data manifests selected by
[configs/small_data_scale](../configs/small_data_scale/README.md). The 42
configurations cover seven detectors and 1%, 2%, 5%, 10%, 20%, and 50% of the
Mega-MMDF development data. Percentage names do not create the subsets or
determine their membership automatically.

The [training list](small_scale_train_commands.txt) uses exact run directories
with the boolean `--resume` flag. For example:

```bash
python train.py \
  --detector_path configs/small_data_scale/baseline_train_on_megammdf_1.yaml \
  --log_dir 'logs/small_scale_train/Baseline_train_on_1%_Mega-MMDF' \
  --resume
```

Remove `--resume` to start fresh and clear that run directory. Prepare every
training and validation manifest named in the configuration, not only the
Mega-MMDF subset.

The [test list](small_scale_test_commands.txt) evaluates selected model-only
checkpoints on the full test datasets. Supply its named weights and source
datasets first; testing is independent of the training-data fraction. See the
[scaling figure guide](../analysis/data_scale/README.md) for plotting inputs.
