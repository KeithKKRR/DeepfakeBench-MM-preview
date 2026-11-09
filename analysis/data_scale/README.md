# Training-data scaling figure

`plot.py` reads the paper's seven-model scaling table and produces its figures.

```bash
python analysis/data_scale/plot.py --plot-only
```

The bundled input is this module's `outputs/data.json`; retain it when excluding generated figures. It contains 49 records: seven detectors at 1%, 2%, 5%, 10%, 20%, 50%, and 100%. Provenance includes workbook/hash, worksheet rows, recorded within-domain AUC, cross-domain mean, seven cross-domain components, and source discrepancies.

To extract a compatible workbook:

```bash
python analysis/data_scale/plot.py --workbook /path/to/results.xlsx --extract-only
```

Omit both mode flags to extract and plot. Extraction requires `--workbook`, opens it read-only, and reads `Small-scale scaling law!A8:P56`. Columns C/D supply the plotted values and F:L the seven cross-domain components. Recomputed means remain audit information; recorded values and source conflicts are preserved.

The module-local `outputs/` directory receives `data_scale`, `data_scale_delta`, `data_scale_within`, and `data_scale_cross` PNG/PDF pairs plus `report.md`; extraction also writes `data.json`. PNGs use 600 dpi. The compact main figure is 3.5 inches wide with within/cross-domain panels and actual proportions on a log axis. Delta plots subtract each detector's 100% result; lines connect measured points.

The [scaling configs](../../configs/small_data_scale/README.md) cover six reduced train/validation subsets; 100% uses the full training config. Some target domains participate in validation, so the recorded cross-domain mean is not an unseen-only mean. Changes to models, fractions, or workbook layout require updating the explicit schema checks.
