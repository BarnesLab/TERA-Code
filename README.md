# TERA Jupyter notebook package

Run the notebooks in numerical order. All source data remain in the original TERA Analysis-2 folder; no participant-level source data are duplicated here.

## Notebooks

1. `00_data_inventory_and_quality_control.ipynb`
2. `01_edm_preprocessing_and_figure2.ipynb`
3. `02_daily_multimodal_lstm.ipynb`
4. `03_weekly_multimodal_lstm.ipynb`
5. `04_survey_only_models.ipynb`
6. `05_daily_and_weekly_ablation.ipynb`
7. `06_results_tables_and_export.ipynb`

## Setup

Create an environment and install `requirements.txt`. If the data folder moves, update `DATA_ROOT` in the first code cell of each notebook and in the two scripts under `src/`.

## Reproducibility

- Fixed random seeds are used.
- Cross-validation is grouped by participant.
- Imputation, scaling, and encoding are fitted only on training participants.
- The output directory is `results/`; publication graphics are written to `figures/`.
- Deep-learning results can vary slightly across TensorFlow versions and hardware.
