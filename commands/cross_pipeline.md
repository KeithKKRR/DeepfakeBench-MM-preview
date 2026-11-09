# Single-pipeline training and cross-pipeline testing

Prepare the named `Mega-MMDF-pipeline<ID>` manifests before running these
experiments. Each manifest must contain the intended fake pipeline and real
references in the appropriate splits. A dataset name does not generate those
manifests automatically.

The [configurations](../configs/exp/README.md) cover FGMDF, FRADE, and FauForensics
trained separately on pipelines 1–22. For example:

```bash
python train.py \
  --detector_path configs/exp/fgmdf_train_on_pipeline1.yaml \
  --log_dir logs/exp_train/FGMDF_train_on_pipeline1
```

This command starts fresh and clears the selected run directory. Add `--resume`
to preserve and resume it. The complete [training list](single_pipeline_train_commands.sh)
contains all 66 configurations:

```bash
bash commands/single_pipeline_train_commands.sh
```

Select a checkpoint for each configuration using the intended validation
protocol. The [test list](single_pipeline_test_commands.sh) evaluates each
selected model on all 22 pipeline test manifests:

```bash
WEIGHTS_DIR=/path/to/selected/checkpoints bash commands/single_pipeline_test_commands.sh
```

`WEIGHTS_DIR` contains model-only files named as documented in the list, such as
`FGMDF_train_on_pipeline1.pth`. Test results are written under
`logs/exp/<Model>_train_on_pipeline<ID>/`. Saved-result visualization is documented
in the [pipeline figure guide](../analysis/pipeline_figures/README.md).
