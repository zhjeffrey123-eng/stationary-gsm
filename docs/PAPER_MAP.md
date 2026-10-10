# Manuscript files and Python entry points

This package supports *A Stationary GSM Extraction Method for 2D Periodic Metal Pixel Metasurfaces*. `P` denotes `output/awpl/revision20_metal_paper/`, and `T` denotes `output/awpl/revision19_metal_xy/`.

| Manuscript evidence | Data | Python implementation |
|---|---|---|
| Section II, paired incidence, extraction identity | `T/data/controlled_scaling.csv`, `controlled_slopes.csv`, `quadratic_identity.csv` | `physics.py`, `extraction.py`, `validation.demo` |
| Fig. 2, current-grid and Green-spectrum refinement | `P/fine_reference/refined_response.csv`, `refined_deltas.csv`, reference NPZ files | `workflows.fine_validation` |
| Fig. 3, PEC masks, current and electric-field maps | `P/fine_reference/selected_mask_701.csv`, `fine_fields_t701_f7_a0_N96_Q768.npz` | `fields.py`, `workflows.field_maps` |
| Fig. 3, TE/TM spectra | `P/stack_validation/single_sheet_spectra.csv` | `physics.py`, `stack.reflectance` |
| Network architecture and frozen current predictor | `model.npz`, `P/nn/local_training_history.csv`, `output/pec8x8_trial/dataset.h5` | `network.py` |
| Table I, 1000 unseen patterns and same-current errors | `P/nn/cocg_novel_test_quality.csv`, `cocg_failed_cases.csv`, `novel_test_labels.h5` | `validation.statistics`, `workflows.evaluate`, `correction.py` |
| Fig. 4 and Table II, two- and three-layer coupling | `P/stack_validation/stack_validation_full.csv`, `stack_spectra.csv`, `stack_gap_amplitudes_f7_th40.csv`, reference NPZ files | `stack.py`, `workflows.stack_validation` |
| Main-figure rendering | `P/figures/Figure_1.json` through `Figure_4.json` | `figures.py` |
| Format integrity | `docs/data_conversion.json` | Exact value checks for every converted source data file |
| Porting validation | `docs/python_validation.json` | `python -m stationary_gsm audit` |

The original CSV values and numerical metadata remain unchanged. Source-hash fields in original experiment JSON files identify the original implementation, rather than the Python source. The conversion report links original data files to their HDF5/NPZ counterparts. `MANIFEST_SHA256.csv` identifies the current distributed files.

Two dataset ID spaces are separate. The 900-pattern dataset contains the fixed illustrations 701–703. The new 1000-pattern dataset contains the failed case 954. Converted reference-cache `info.ids` retain one-based indices. Newly computed Python caches explicitly record zero-based indices.
