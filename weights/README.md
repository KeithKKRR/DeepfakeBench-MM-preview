# Final detector checkpoints

Final benchmark detector checkpoints are distributed separately. The download
repository link and selected checkpoint list are pending. AVFF/MCL stage inputs
are listed in the [initialization guide](../pretrained/README.md).

`test.py --weight_path` expects a model-only state dictionary, such as a file
under a training run's `checkpoints/` directory. Full `training_states/` files
are resume containers. Detector constructors may also read initialization assets
before the final checkpoint is loaded.

## SpeechForensics

The evaluation adapter uses the AV-HuBERT clean-pretraining checkpoint
[large_vox_iter5.pt](https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt),
linked by the SpeechForensics reference implementation. Place it at
`weights/SpeechForensics/large_vox_iter5.pt` and pass that path to
`test.py --weight_path`.

This full Fairseq checkpoint includes model/task configuration and is restored
through the adapter's loading hook, which freezes the backbone. AVH's VSR
fine-tuned checkpoint serves a different model configuration. Checkpoint files
are not bundled; retain each provider's license and access conditions.
