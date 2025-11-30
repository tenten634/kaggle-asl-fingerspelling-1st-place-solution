# Kaggle Setup & Training Guide

Setup guide for training ASL Fingerspelling Recognition models in Kaggle Notebooks.

## Prerequisites

1. Create Kaggle Notebook
2. Enable accelerator: **GPU P100** (recommended) or **TPU v5e8**
3. Add datasets via Data sidebar:
   - `asl-fingerspelling-preprocessing-train-dataset`
   - `asl-fingerspelling-preprocessed-supp-dataset`

## Step 1: Clone Repository

```python
!git clone https://github.com/tenten634/kaggle-asl-fingerspelling-1st-place-solution.git
%cd kaggle-asl-fingerspelling-1st-place-solution
!git checkout dev
```

## Step 2: Install Packages

### GPU Setup

```python
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0
```

### TPU Setup

```python
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0
!pip install cloud-tpu-client==0.10 torch-xla[tpu]==2.8.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# Fix protobuf version (TPU installs old version, transformers needs newer)
!pip install --upgrade --no-deps protobuf>=5.28.0
!pip install --upgrade google-api-core>=2.27.0 google-api-python-client>=2.0.0

**Note**: 
- Using `torch-xla==2.8.0` (2.9.0 has symbol errors with current PyTorch)
- Dependency warnings about google-api packages are expected and harmless
- Protobuf MUST be >=5.28.0 or training will fail with import errors

## Step 3: Prepare Data

Copy CSV/JSON metadata files. Landmark data uses virtual merge from `/kaggle/input/` (no copying needed).

```python
import os
import shutil

repo_dir = '/kaggle/working/kaggle-asl-fingerspelling-1st-place-solution'
os.chdir(repo_dir)
os.makedirs('datamount', exist_ok=True)

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
```

## Step 4: Verify Setup

```python
!python check_kaggle_environment.py
```

Expected: Data files ready, input datasets found, project files exist.

## Step 5: Start Training

### Quick Path (Recommended)

Train cfg_2 directly (skips Round 1):

**GPU:**
```python
%cd kaggle-asl-fingerspelling-1st-place-solution
!python train_kaggle.py -C cfg_2 --fold -1
!python train_kaggle.py -C cfg_2 --fold -1
```

**TPU:**
```python
%cd kaggle-asl-fingerspelling-1st-place-solution
# Note: Batch size is automatically reduced to 32 for TPU (TPU v5e8 has 15.75GB memory)
# If you still get OOM errors, manually reduce: --batch_size 16 or --batch_size 8
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu
```

**Training Time:**
- GPU P100: ~6-10 hours per seed
- TPU v5e8: ~3-5 hours per seed (first iteration: 5-15 min JIT compilation)

### Full Pipeline

**Round 1 - Train 4 folds:**
```python
# GPU: Remove --use_tpu flag
!python train_kaggle.py -C cfg_1 --fold 0 --use_tpu  # TPU
!python train_kaggle.py -C cfg_1 --fold 1 --use_tpu  # TPU
!python train_kaggle.py -C cfg_1 --fold 2 --use_tpu  # TPU
!python train_kaggle.py -C cfg_1 --fold 3 --use_tpu  # TPU
```

**Generate OOF predictions:**
```python
!python scripts/get_train_folded_oof_supp.py
```

**Round 2 - Train fullfit (2 seeds):**
```python
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu  # TPU
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu  # TPU
```

**Convert to TF-Lite:**
```python
!python scripts/convert_cfg_2_to_tf_lite.py
```

## Troubleshooting

### GPU
- **GPU not available**: Settings → Accelerator → GPU P100 → Restart
- **CUDA out of memory**: Reduce `batch_size` in config or enable `mixed_precision = True`

### TPU
- **First iteration slow (5-15 min)**: Normal - TPU JIT compilation
- **Out of memory (OOM)**: TPU v5e8 has 15.75GB HBM - batch_size is automatically reduced to 32 for TPU
- **Device or resource busy**: Restart notebook session
- **Protobuf import error**: Re-run TPU package installation from Step 2
- **TPU symbol error**: Try `torch-xla==2.8.0` instead of 2.9.0
- **Memory error persists**: Manually reduce batch_size: `--batch_size 16` or `--batch_size 8`

### General
- **Data not found**: Verify datasets added in Data sidebar
- **Module not found**: Install missing package: `!pip install package_name`

## Notes

- **Virtual Merge**: Landmark data accessed directly from `/kaggle/input/` (no copying)
- **Checkpoints**: Saved to `datamount/weights/` (auto-saved to Output tab)
- **Time Limit**: 9 hours per session (free tier)
- **GPU vs TPU**: GPU is more stable; TPU is faster but requires dependency fixes
