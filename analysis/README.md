# Analysis tools

The modules below analyze saved predictions, model features, or dataset metadata. FAD, FVD, and Grad-CAM also support feature or attribution extraction from supplied media and model assets.

| Directory | Purpose |
|---|---|
| appendix_scores | Paper ROC/PR, score distributions, and demographic figures from bundled video-score fixtures. |
| [pipeline_figures](pipeline_figures/README.md) | Four pipeline-comparison pages from the bundled 770 AUC values and provenance. |
| [roc](roc/README.md), [pr](pr/README.md) | Curves from native prediction JSON at sample or video level. |
| [confusion_matrix](confusion_matrix/README.md) | Thresholded binary metrics and normalized confusion matrices. |
| [probability_distribution](probability_distribution/README.md) | Real/fake or four-modality-group score distributions. |
| [fairness](fairness/README.md) | Metrics for explicitly annotated demographic groups. |
| [tsne](tsne/README.md) | Feature projections and paper layouts based on saved projections. |
| [gradcam](gradcam/README.md), [utils](utils/README.md) | Attribution maps and shared audio/video presentation. |
| [fad](fad/README.md), [fvd](fvd/README.md) | Audio/video distribution distances with VGGish/I3D features. |
| [predicted_prob_histrogram](predicted_prob_histrogram/README.md) | Paired full-input and modality-masked scores; directory spelling is retained. |
| [predicted_prob_change_histogram](predicted_prob_change_histogram/README.md) | Paired mask-audio score changes. |
| [misalignment](misalignment/README.md) | Aligned/shifted real-window comparisons and the ten-method paper figure. |
| [common_failures](common_failures/README.md) | Shared-error catalogues and optional media inspection. |
| [data_scale](data_scale/README.md) | Training-data scaling figures from the saved table or a supplied workbook. |
| [dataset_statistics](dataset_statistics/README.md) | Raw-clip metadata collection and plots. |

Commands in the module READMEs are run from the project root unless stated otherwise. Native prediction tools read `test.py` JSON files; bundled appendix fixtures are already aggregated to video level. Required external predictions, feature caches, media, and checkpoints are described next to the tool that uses them.

Training-free inputs describes the SpeechForensics, Qwen2.5-Omni, and VideoLLaMA2.1 saved-score fixtures. Experiment setup is documented with the [single-pipeline](../configs/exp/README.md), [scaling](../configs/small_data_scale/README.md), and [misalignment](../configs/misaligned_test/README.md) configs; weight acquisition is documented in [pretrained](../pretrained/README.md).

Retain `data_scale/outputs/data.json` and the saved t-SNE HTML projection inputs when excluding generated outputs: those existing paths are reproduction inputs.
