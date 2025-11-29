# Kaggle Setup & Training Guide

Quick setup guide for ASL Fingerspelling Recognition training in Kaggle Notebooks.

## Prerequisites

1. Create Kaggle Notebook → Enable **TPU v5e8** (Settings → Accelerator → TPU)
2. Add datasets via Data sidebar:
   - `asl-fingerspelling-preprocessing-train-dataset`
   - `asl-fingerspelling-preprocessed-supp-dataset`

## Step 1: Clone Repository

```python
!git clone https://github.com/tenten634/kaggle-asl-fingerspelling-1st-place-solution.git
%cd kaggle-asl-fingerspelling-1st-place-solution
```

## Step 2: Install Packages

```python
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0
!pip install cloud-tpu-client==0.10 torch-xla[tpu]==2.9.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

**Note**: Dependency warnings (e.g., protobuf conflicts) are expected and can be ignored. Installation will complete successfully.

## Step 3: Prepare Data

**Note**: Only CSV/JSON files are copied. Landmark data uses virtual merge from `/kaggle/input/` (no copying needed).

```python
import os
import shutil

# Make sure we're in the repository directory
repo_dir = '/kaggle/working/kaggle-asl-fingerspelling-1st-place-solution'
os.chdir(repo_dir)

# Create datamount directory in repository
os.makedirs('datamount', exist_ok=True)

# Copy CSV/JSON files from input datasets
input_base = '/kaggle/input'
for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if os.path.isdir(dataset_path):
        for file in os.listdir(dataset_path):
            if file.endswith(('.csv', '.json')):
                src = os.path.join(dataset_path, file)
                dst = os.path.join('datamount', file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"✅ Copied {file}")
```

## Step 4: Verify Setup

```python
!python check_kaggle_environment.py
```

Should show: ✅ TPU available, ✅ Input datasets found, ✅ Data files ready

## Step 5: Start Training

### Quick Path (Recommended)

Skip Round 1 - go directly to Round 2:

```python
%cd kaggle-asl-fingerspelling-1st-place-solution

# Train cfg_2 with 2 seeds
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu
```

**Time**: ~3-5 hours per seed on TPU v5e8

### Full Pipeline

```python
# Round 1: Train 4 folds
!python train_kaggle.py -C cfg_1 --fold 0 --use_tpu
!python train_kaggle.py -C cfg_1 --fold 1 --use_tpu
!python train_kaggle.py -C cfg_1 --fold 2 --use_tpu
!python train_kaggle.py -C cfg_1 --fold 3 --use_tpu

# Generate OOF predictions
!python scripts/get_train_folded_oof_supp.py

# Round 2: Train fullfit (2 seeds)
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu
```

### Convert to TF-Lite

```python
!python scripts/convert_cfg_2_to_tf_lite.py
```

## Troubleshooting

- **TPU not available**: Settings → Accelerator → TPU v5e8 → Restart
- **Dependency conflicts**: Warnings about protobuf/API versions are expected - ignore them
- **Out of memory**: Reduce `batch_size` in config or enable `mixed_precision = True`
- **Data not found**: Verify datasets are added in Data sidebar
- **Module not found**: `!pip install package_name`

## Key Points

- **Virtual Merge**: Landmark data accessed directly from `/kaggle/input/` (no copying)
- **Checkpoints**: Saved to `datamount/weights/` (auto-saved to Output tab)
- **Time Limit**: 9 hours per session (free tier)
