# Pancreatic Tumor Classification Pipeline with 3D Foundation Model (CT-FM)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-blueviolet.svg)](https://monai.io/)
[![CT-FM](https://img.shields.io/badge/Foundation_Model-CT--FM_(77M)-success.svg)](https://github.com/project-lighter/CT-FM)

An end-to-end deep learning pipeline for **Pancreatic Tumor Classification** based on the 3D Computed Tomography foundation model **CT-FM** (*Harvard / MGH*). 

This pipeline takes **raw 3D abdominal CT scans**, **clinical metadata**, and **pancreatic segmentation masks**, performs anatomical ROI extraction and multi-phase harmonization, and trains an imbalanced-aware classifier with evaluation metrics including ROC-AUC, Balanced Accuracy, Sensitivity, and Specificity.

---

## 📌 Architecture & Pipeline Overview

```
Raw 3D Abdominal CT (.nii.gz)
         +
Pancreas / Lesion Segmentation Mask
         │
         ▼
[1] 3D Anatomical ROI Bounding Box Extraction (union of pancreas + lesion masks, 20 mm margin on each axis)
         │
         ▼
[2] Standardized Preprocessing (MONAI MetaTensor w/ affine: SPL orientation + 2.5mm isotropic resampling + CT-FM HU window [-1024, 2048])
         │
         ▼
[3] Phase-stratified negative sampling + 5-fold stratified CV assignment
         │
         ▼
[4] Pre-trained CT-FM Feature Extractor (77M SegResEncoder Backbone, Frozen)
         │
         ▼
[5] Class-Weighted MLP Classification Head (weights from training labels)
         │
         ▼
[6] Comprehensive Quantitative Evaluation (AUC, Sensitivity, Specificity, per-case report CSV)
```

---

## 🌟 Key Features

1. **Pancreas-centred ROI extraction**: bounding box of the pancreas mask ∪ lesion mask with a margin in
   millimetres (orientation/spacing independent). The affine is preserved (MONAI `MetaTensor`), so
   `Orientation(SPL)` and `Spacing(2.5 mm)` act on the true geometry. If a ROI exceeds the model input
   size a warning is emitted (the ROI would be centre-cropped).
2. **Phase-stratified negative sampling**: negatives are sampled proportionally to the CT-phase mix; the
   phase × label cross-table is printed so confounding is visible. It does **not** fully remove phase
   confounding.
3. **Imbalance-aware training**: class weights derived from the training labels.
4. **Parameter-hashed tensor cache**: deterministic preprocessing is cached, random augmentation is not.

---

## 📂 Repository Structure

```
pancreas-ctfm-xai/
├── README.md                     # Comprehensive documentation & reproducibility guide
├── requirements.txt              # Python dependency specifications
├── .gitignore                    # Excludes raw CTs, tensor caches, and large weights
├── data/
│   ├── cohort_50.csv             # 50-case pilot cohort incl. `fold` column (5-fold CV)
│   ├── train_split.csv           # fold != 0 (40 cases)
│   ├── val_split.csv             # fold == 0 (10 cases)
│   └── val_predictions_<tag>.csv # per-case predictions written by train.py
├── tests/test_preprocessing.py   # synthetic-data checks for ROI / orientation / spacing
├── src/
│   ├── config.py                 # paths (override raw data root with $PANTS_DATA_ROOT)
│   ├── dataset.py                # PyTorch / MONAI Dataset with 3D ROI crop and tensor caching
│   ├── model.py                  # CTFMClassifier architecture (CT-FM Backbone + MLP Head)
│   ├── generate_cohort.py        # Stratified sampling + K-fold assignment
│   ├── visualize_roi.py          # 2D/3D visual inspection comparing raw vs. ROI vs. ground-truth
│   ├── train.py                  # End-to-end training, checkpointing, and evaluation loop
│   └── quick_prototype.py        # Single-batch sanity-check prototype
└── checkpoints/                  # Saved model weights (excluded from git tracking)
```

---

## 🚀 Getting Started

### 1. Environment Installation

Ensure Python 3.10+ (Python 3.11 recommended) is installed.

```bash
# Clone the repository
git clone https://github.com/BingerLiang77/pancreas-ctfm-xai.git
cd pancreas-ctfm-xai

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Getting the Data (PanTS)

This repo contains **no CT images or annotations**, only tables. Download PanTS from the official sources
(the pilot used the `PanTSMini` release):

| Resource | Link |
| :--- | :--- |
| Official GitHub (scripts, docs) | https://github.com/MrGiovanni/PanTS |
| Hugging Face (`PanTSMini` images + `metadata.xlsx`) | https://huggingface.co/datasets/BodyMaps/PanTSMini |
| Paper | https://arxiv.org/abs/2507.01291 |

```bash
# Option A: official scripts (images ~300 GB in total, plus labels)
git clone https://github.com/MrGiovanni/PanTS.git
cd PanTS
bash download_PanTS_data.sh
bash download_PanTS_label.sh

# Option B: only the chunk used by this pilot (cases 1-1000) from Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download BodyMaps/PanTSMini PanTSMini_ImageTr_00000001_00001000.tar.gz metadata.xlsx \
    --repo-type dataset --local-dir ./PanTS_data
tar -xzf PanTS_data/PanTSMini_ImageTr_00000001_00001000.tar.gz -C PanTS_data
```

The segmentation labels (`PanTSMini_Label`) come from the official `download_PanTS_label.sh` script.
Check the PanTS repository for the current label location if the script changes.

> **License:** PanTS is released under **CC BY-NC-SA 4.0** (non-commercial, share-alike, attribution
> required). Respect these terms when using or redistributing the data and derived tables.

Expected layout (point `PANTS_DATA_ROOT` at the parent folder; folder names are set in `src/config.py`):

```
$PANTS_DATA_ROOT/
├── PanTSMini_ImageTr_00000001_00001000/
│   └── PanTS_00000001/ct.nii.gz
└── PanTSMini_Label 2/                  # labels folder, name as in config.py (LABEL_DIR)
    └── PanTS_00000001/segmentations/
        ├── pancreas.nii.gz
        └── pancreatic_lesion.nii.gz
```

> **Metadata columns required** (`data/metadata.xlsx`, same file as on Hugging Face):
> - `PanTS ID`: scan identifier (e.g. `PanTS_00000029`).
> - `tumor?`: binary label (`0` normal/benign, `1` tumour).
> - `ct phase`: `Non-contrast`, `Arterial` or `Venous`.

---

## 🛠️ Step-by-Step Pipeline Execution

### Step 1: Quality Inspection & Visual Verification

Run the visualizer to inspect how raw CT scans compare against the extracted pancreas ROI and ground-truth masks:

```bash
python src/visualize_roi.py \
    --data_dir "/path/to/ct_images" \
    --label_dir "/path/to/segmentations" \
    --out_path "pancreas_roi_comparison.png"
```

The script outputs a multi-panel comparison showing:
- Column 1: Full Abdominal CT slice.
- Column 2: Pancreas ROI bounding box crop with soft-tissue HU windowing $[-135, 215]$.
- Column 3: Ground-truth overlay (Green: normal pancreas parenchyma, Red: tumor lesion).

---

### Step 2: Stratified Cohort Generation & 8:2 Split

Sample a balanced cohort maintaining the natural empirical phase distribution:

```bash
python src/generate_cohort.py
```

This outputs `data/cohort_50.csv` (with a `fold` column), `data/train_split.csv` and `data/val_split.csv`
(fold 0 as validation). Free-text report columns are dropped.

---

### Step 3: Model Training & Evaluation

Train the CT-FM classifier with class-weighted loss on Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU:

```bash
export PANTS_DATA_ROOT="/path/to/data"     # contains PanTSMini_ImageTr_* and PanTSMini_Label 2
for k in 0 1 2 3 4; do python src/train.py --fold $k --epochs 10; done
python tests/test_preprocessing.py         # quick sanity checks of the preprocessing
```

Per fold: `checkpoints/ctfm_head_fold<k>.pth` (classifier head only), `checkpoints/history_fold<k>.csv`,
`data/val_predictions_fold<k>.csv`.

---

## 📊 Pilot Status

> ⚠️ **This is a pipeline feasibility pilot, not a performance claim.**
> The cohort has only **5 tumour-positive cases** (45 negatives), so every fold/validation set holds a
> single positive. AUC/sensitivity from one fold are statistically meaningless; the earlier
> "AUC = 1.000" table came from a 1-positive validation set, a buggy preprocessing pipeline and a
> post-hoc threshold, and has been removed.

What the pilot does show: ROI extraction, orientation/spacing normalisation, caching, CT-FM feature
extraction and the training/evaluation loop run end to end on Apple Silicon (MPS).

Recommended evaluation for now: 5-fold stratified CV (`--fold 0..4`, one positive per fold), pool the
out-of-fold probabilities, and report AUC with a bootstrap confidence interval. Pick the threshold
inside the training folds, never on the validation fold.

### Known limitations
- Tiny sample (5 positives); class weights are derived from the training labels.
- CT phase is not balanced across labels (positives: 2 non-contrast / 1 arterial / 2 venous).
  Scanner/site/study type also differ between groups: a confounding risk.
- ROI uses **ground-truth** pancreas (∪ lesion) masks. At deployment an automatic segmentation will
  be needed; expect a distribution shift.
- Cases with empty pancreas masks or truncated CT files are excluded by `generate_cohort.py`.
- No explainability (XAI) module yet.

---

## 📈 Scaling to Larger Cohorts

1. **Batch offline preprocessing**: `.pt` ROI tensors are cached in `data/cache_roi_pt/<param-hash>/`;
   the cache key includes margin / spacing / size / HU window, so changing them never reuses stale tensors.
   Random augmentation is applied *after* the cache.
2. **Features**: with a frozen backbone and no augmentation, features can be cached once for fast CV.
3. **Unsegmented CTs**: run TotalSegmentator to obtain `pancreas.nii.gz` first.
4. Multi-GPU / mixed precision are **not implemented** yet.

---

## 📜 Citation & Acknowledgements

- **CT-FM Model**: Harvard / MGH [project-lighter/CT-FM](https://github.com/project-lighter/CT-FM)
- **PanTS Dataset**: Pancreatic Tumor Segmentation Dataset ([arXiv:2507.01291](https://arxiv.org/abs/2507.01291))
- **MONAI Framework**: [Project MONAI](https://monai.io/)

---

## 📄 License
License: not yet specified (add a LICENSE file before redistributing).
