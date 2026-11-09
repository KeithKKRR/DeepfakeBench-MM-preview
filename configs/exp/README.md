# Single-pipeline training configurations

These 66 YAML files cover FGMDF, FRADE, and FauForensics trained separately on
Mega-MMDF pipelines 1–22. Each `{model}_train_on_pipeline{ID}.yaml` selects the
corresponding `Mega-MMDF-pipeline{ID}` training and validation manifest.

Prepare those manifests with real references and the intended fake pipeline,
configure [path.yaml](../path.yaml), and obtain the method's initialization assets.
The pipeline name selects prepared data; it does not generate a split.

## Run

The [training templates](../../commands/single_pipeline_train_commands.sh) cover all 66
configurations. They start fresh and clear their named run directories; add
the boolean `--resume` flag to an individual command to preserve and resume it:

```bash
bash commands/single_pipeline_train_commands.sh
```

Run an individual configuration with:

```bash
python train.py --detector_path configs/exp/fgmdf_train_on_pipeline1.yaml --log_dir logs/exp_train/FGMDF_train_on_pipeline1
```

The [test templates](../../commands/single_pipeline_test_commands.sh) evaluate each
selected checkpoint on all 22 pipeline test manifests using video-level AUC:

```bash
WEIGHTS_DIR=/path/to/selected/checkpoints bash commands/single_pipeline_test_commands.sh
```

The expected checkpoint names include `FGMDF_train_on_pipeline1.pth`. Select them
using the training/validation protocol. Test predictions are written beneath
`logs/exp/{Model}_train_on_pipeline{ID}/predictions/`. Training and testing
`--log_dir` values both select their exact output directories. Training preserves
an existing directory only when `--resume` is supplied.

To add a pipeline or model, supply matching manifests and configurations, then
update both command lists. The [pipeline figure guide](../../analysis/pipeline_figures/README.md)
describes the separate five-training-source comparison from saved values.
The [cross-pipeline command guide](../../commands/cross_pipeline.md) summarizes
the required manifests, training, and checkpoint selection.
