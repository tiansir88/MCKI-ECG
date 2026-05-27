# MCKI-ECG: Multi-Level Clinical Knowledge Injection for ECG Representation Learning

This repository contains the source code and analysis scripts for **MCKI-ECG**, a Multi-Level Clinical Knowledge Injection framework for multi-label 12-lead ECG representation learning.

MCKI-ECG integrates four complementary forms of clinical knowledge:

1. **Diagnostic-relation Knowledge**: models clinically confusable diagnostic relations through graph-informed hard negative modeling.
2. **Lead-topology Knowledge**: uses the structured multi-view nature of 12-lead ECG.
3. **Acquisition-robustness Knowledge**: improves robustness under imperfect or missing lead acquisition.
4. **Local-morphology Knowledge**: enhances representation learning through local waveform morphology.

> **Naming note.** Some internal scripts or historical checkpoints may use `GHNM`. In this repository, `GHNM` refers to the central diagnostic-relation hard negative mechanism within the full MCKI-ECG framework. The paper-level method name is **MCKI-ECG**.

---

## Repository structure

```text
MCKI-paper-release/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── resources/
│   └── confusable_pairs_v1.csv
│
├── src/
│   ├── MCKI_backbone_factory.py
│   ├── MCKI_loss_pro.py
│   ├── MCKI_relation_builder_stage4.py
│   ├── dataset_v3.py
│   ├── losses_multilabel.py
│   ├── resnet1d.py
│   ├── train_v3_2stage.py
│   ├── st_mem.py
│   ├── run_stage8_leadaware_multiscale_MCKI_5protocols.py
│   └── ablation_stage8.py
│
├── external_eval/
│   ├── evaluate_external_georgia.py
│   ├── evaluate_external_sph.py
│   ├── external_dataset.py
│   └── metrics_external.py
│
├── scripts/
│   ├── data_preparation/
│   │   ├── georgia_step1_audit_and_extract_dx.py
│   │   ├── georgia_step2_make_ptbxl5_manifest.py
│   │   ├── georgia_step3_build_external_processed_auto.py
│   │   ├── sph_step1_build_statement_table.py
│   │   ├── sph_step2_make_ptbxl5_manifest.py
│   │   └── sph_step3_build_external_processed_auto.py
│   │
│   ├── train_eval/
│   │   └── run_stage8_leadaware_multiscale_MCKI_export_arrays.py
│   │
│   └── analysis/
│       ├── plot_mi_vs_sttc_score_distribution.py
│       ├── build_confusable_pair_table.py
│       └── build_hndr_supp_table_from_master_md.py
```

---

## Main components

### `src/`

Core model, training, and ablation code.The core implementation code for this part is currently not publicly available, pending completion of institutional intellectual property review.

- `MCKI_backbone_factory.py`: builds the MCKI-ECG backbone.
- `MCKI_loss_pro.py`: implements the MCKI-ECG training objective, including diagnostic-relation hard negative modeling.
- `MCKI_relation_builder_stage4.py`: builds diagnostic relation structures used by the relation-guided mechanism.
- `dataset_v3.py`: dataset loading utilities.
- `losses_multilabel.py`: multi-label classification losses and related utilities.
- `resnet1d.py`: 1D ResNet ECG encoder implementation.
- `train_v3_2stage.py`: two-stage training entry point.
- `st_mem.py`: supporting memory or state-tracking utilities.
- `run_stage8_leadaware_multiscale_MCKI_5protocols.py`: main protocol-level MCKI-ECG evaluation script.
- `ablation_stage8.py`: ablation experiments for MCKI-ECG knowledge components.

### `external_eval/`

External validation code for Georgia and SPH datasets.

- `evaluate_external_georgia.py`: Georgia external evaluation.
- `evaluate_external_sph.py`: SPH external evaluation.
- `external_dataset.py`: external dataset loading utilities.
- `metrics_external.py`: external evaluation metrics.

### `scripts/data_preparation/`

Dataset preprocessing and manifest construction scripts.

- `georgia_step1_audit_and_extract_dx.py`: audits Georgia diagnostic statements and extracts target diagnostic labels.
- `georgia_step2_make_ptbxl5_manifest.py`: builds Georgia-to-PTB-XL-style manifest files.
- `georgia_step3_build_external_processed_auto.py`: generates processed Georgia arrays for external evaluation.
- `sph_step1_build_statement_table.py`: builds SPH diagnostic statement tables.
- `sph_step2_make_ptbxl5_manifest.py`: builds SPH-to-PTB-XL-style manifest files.
- `sph_step3_build_external_processed_auto.py`: generates processed SPH arrays for external evaluation.

### `scripts/train_eval/`

Training/evaluation helper scripts.

- `run_stage8_leadaware_multiscale_MCKI_export_arrays.py`: reruns the Stage-8 lead-aware and multiscale MCKI-ECG setting and exports intermediate arrays such as prediction probabilities, targets, thresholds, and validation outputs for downstream analysis.

### `scripts/analysis/`

Figure and supplementary analysis scripts.

- `plot_mi_vs_sttc_score_distribution.py`: plots MI-vs-STTC score or margin distributions.
- `build_confusable_pair_table.py`: builds confusable-pair analysis tables.
- `build_hndr_supp_table_from_master_md.py`: builds supplementary HNDR tables from master result files.

### `resources/`

- `confusable_pairs_v1.csv`: predefined clinically confusable diagnostic pairs used by diagnostic-relation knowledge modeling and pair-level analysis.

---

## Installation

Create a clean Python environment:

```bash
conda create -n mcki-ecg python=3.10 -y
conda activate mcki-ecg
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the repository root as `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH}"
```

---

## Data preparation

This repository does not redistribute raw ECG datasets. Please download the datasets from their official sources and preprocess them into the expected local format.

Typical preprocessing workflow:

```bash
# Georgia preprocessing
python scripts/data_preparation/georgia_step1_audit_and_extract_dx.py
python scripts/data_preparation/georgia_step2_make_ptbxl5_manifest.py
python scripts/data_preparation/georgia_step3_build_external_processed_auto.py

# SPH preprocessing
python scripts/data_preparation/sph_step1_build_statement_table.py
python scripts/data_preparation/sph_step2_make_ptbxl5_manifest.py
python scripts/data_preparation/sph_step3_build_external_processed_auto.py
```

Before running the scripts, check and update dataset paths, output directories, and label mappings according to your local environment.

---

## Training and protocol-level evaluation

Run the main MCKI-ECG training or evaluation script:

```bash
python src/run_stage8_leadaware_multiscale_MCKI_5protocols.py
```

For two-stage training:

```bash
python src/train_v3_2stage.py
```

For ablation experiments:

```bash
python src/ablation_stage8.py
```

If your scripts expose command-line arguments, set the following paths explicitly:

```bash
--data-dir /path/to/processed/data
--output-dir /path/to/outputs
--confusable-pairs resources/confusable_pairs_v1.csv
--checkpoint /path/to/checkpoint.pth
```

---

## External evaluation

After preprocessing Georgia or SPH data, run:

```bash
python external_eval/evaluate_external_georgia.py
python external_eval/evaluate_external_sph.py
```

Make sure the external evaluation scripts point to:

```text
resources/confusable_pairs_v1.csv
```

rather than any legacy internal path such as `src3/confusable_pairs_v1.csv`.

---

## Exporting arrays for supplementary analyses

Some supplementary analyses require saved model outputs such as probabilities, targets, validation probabilities, validation targets, and thresholds.

Run:

```bash
python scripts/train_eval/run_stage8_leadaware_multiscale_MCKI_export_arrays.py
```

Expected exported files may include:

```text
test_probs.npy
test_targets.npy
thresholds.npy
val_probs.npy
val_targets.npy
```

These files are used by the analysis scripts below.

---

## Analysis and figure generation

Run MI-vs-STTC score distribution analysis:

```bash
python scripts/analysis/plot_mi_vs_sttc_score_distribution.py
```

Build confusable-pair tables:

```bash
python scripts/analysis/build_confusable_pair_table.py
```

Build supplementary HNDR tables:

```bash
python scripts/analysis/build_hndr_supp_table_from_master_md.py
```

Check each script for its expected input paths before execution.

---

## Reproducibility notes

For reproducible results, report or fix the following settings:

- random seed;
- dataset split;
- input sampling frequency and signal length;
- label set and label order;
- threshold selection strategy;
- checkpoint path;
- evaluation protocol;
- confusable-pair definition file.

The file `resources/confusable_pairs_v1.csv` should be version-controlled because it affects diagnostic-relation knowledge modeling and HNDR-related analysis.

---

## Expected outputs

Depending on the entry point, outputs may include:

```text
checkpoints/
results/
logs/
test_probs.npy
test_targets.npy
thresholds.npy
val_probs.npy
val_targets.npy
figures/
supplementary_tables/
```

Large files such as checkpoints, processed datasets, cached arrays, and generated figures should not be committed unless explicitly required. Use `.gitignore` to exclude them.

---

## Recommended `.gitignore` entries

```text
__pycache__/
*.pyc
*.pyo
*.pyd
.ipynb_checkpoints/

checkpoints/
outputs/
results/
logs/
figures/
processed/
cache/

*.npy
*.npz
*.pth
*.pt
*.ckpt
*.zip
*.tar
*.gz
```

---

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@article{mcki_ecg,
  title   = {MCKI-ECG: Multi-Level Clinical Knowledge Injection for ECG Representation Learning},
  author  = {Author names omitted for review or to be updated},
  journal = {To be updated},
  year    = {To be updated}
}
```

---

## License

This repository is released under the license specified in `LICENSE`.

---

## Contact

For questions about the code or experiments, please contact the corresponding author listed in the paper.
