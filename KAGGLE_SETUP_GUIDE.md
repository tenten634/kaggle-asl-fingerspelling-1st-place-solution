# Kaggle Setup & Training Guide

## Prerequisites

1. Create Kaggle Notebook
2. Enable accelerator: **GPU P100** or **TPU v5e8** (or CPU for local)
3. Add datasets via Data sidebar:
   - `asl-fingerspelling-preprocessing-train-dataset`
   - `asl-fingerspelling-preprocessed-supp-dataset`

## Step-by-Step Setup

### Step 1: Clone Repository

```python
!git clone https://github.com/tenten634/kaggle-asl-fingerspelling-1st-place-solution.git
%cd kaggle-asl-fingerspelling-1st-place-solution
!git checkout dev
```

### Step 2: Install Packages

**GPU:**
```python
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0
```

**TPU:**
```python
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0
!pip install cloud-tpu-client==0.10 torch-xla[tpu]==2.8.0 -f https://storage.googleapis.com/libtpu-releases/index.html
!pip install --upgrade --no-deps protobuf>=5.28.0
!pip install --upgrade google-api-core>=2.27.0 google-api-python-client>=2.0.0
```

### Step 3: Prepare Data

```python
import os, shutil
os.chdir('/kaggle/working/kaggle-asl-fingerspelling-1st-place-solution')
os.makedirs('datamount', exist_ok=True)
for d in os.listdir('/kaggle/input'):
    p = os.path.join('/kaggle/input', d)
    if os.path.isdir(p):
        for f in os.listdir(p):
            if f.endswith(('.csv', '.json')):
                shutil.copy2(os.path.join(p, f), os.path.join('datamount', f))
```

## Full Training Pipeline

The complete pipeline consists of 3 rounds. **Note: If `datamount/train_folded_oof_supp.csv` already exists, you can skip Round 1 and start from Round 2.**

### Round 1: Train cfg_1 (4 folds) - Generate OOF Predictions

**Purpose:** Train a smaller model to generate out-of-fold (OOF) predictions used as auxiliary targets for Round 2.

**GPU:**
```python
%cd kaggle-asl-fingerspelling-1st-place-solution
!python train_kaggle.py -C cfg_1 --epochs 300 --batch_size 64 --grad_accumulation 8
!python train_kaggle.py -C cfg_1 --fold 1 --epochs 300 --batch_size 64 --grad_accumulation 8
!python train_kaggle.py -C cfg_1 --fold 2 --epochs 300 --batch_size 64 --grad_accumulation 8
!python train_kaggle.py -C cfg_1 --fold 3 --epochs 300 --batch_size 64 --grad_accumulation 8
```

**TPU:**
```python
%cd kaggle-asl-fingerspelling-1st-place-solution
!python train_kaggle.py -C cfg_1 --use_tpu --epochs 300 --batch_size 16 --grad_accumulation 32
!python train_kaggle.py -C cfg_1 --fold 1 --use_tpu --epochs 300 --batch_size 16 --grad_accumulation 32
!python train_kaggle.py -C cfg_1 --fold 2 --use_tpu --epochs 300 --batch_size 16 --grad_accumulation 32
!python train_kaggle.py -C cfg_1 --fold 3 --use_tpu --epochs 300 --batch_size 16 --grad_accumulation 32
```

**After Round 1, generate OOF predictions:**
```python
!python scripts/get_train_folded_oof_supp.py
```

This creates `datamount/train_folded_oof_supp.csv` with OOF predictions from cfg_1.

### Round 2: Train cfg_2 (2 seeds) - Main Model Training

**Purpose:** Train the main model using OOF predictions from Round 1. Two seeds are trained for ensemble.

**Important Parameters:**
- `--batch_size`: Adjust based on memory (GPU: 64, TPU: ≤16)
- `--grad_accumulation`: Maintain effective batch size = 512 (GPU: 8, TPU: 32)
- `--epochs`: Default is 400. Use `--epochs 35` only for Kaggle 9-hour limit.

**GPU Training (Full 400 epochs per seed):**
```python
!python train_kaggle.py -C cfg_2 --fold -1 --epochs 400 --batch_size 64 --grad_accumulation 8
!python train_kaggle.py -C cfg_2 --fold -1 --epochs 400 --batch_size 64 --grad_accumulation 8
```

**GPU Training (35 epochs per session - for Kaggle 9h limit):**

Since Kaggle has a 9-hour limit, train 400 epochs by splitting into multiple sessions. Each seed needs ~11 sessions (400 ÷ 35 ≈ 11):

1. **First session (epochs 0-35):**
   ```python
   !python train_kaggle.py -C cfg_2 --fold -1 --epochs 35 --batch_size 64 --grad_accumulation 8 --total_epochs 400
   ```
   - After completion, download from Output tab:
     - **Checkpoint:** `datamount/weights/cfg_2/fold-1/checkpoint_last_seed{seed}.pth` (required for next session)
     - **Validation data:** `datamount/weights/cfg_2/fold-1/val_data_seed{seed}.pth` (not needed for continuation, only final epoch 400 version is needed)

2. **Upload checkpoint to datamount:**
   - Upload only the **checkpoint** file to `datamount/weights/cfg_2/fold-1/` in your new session
   - Validation data files are not needed for continuation (only the final epoch 400 version is needed for TF-Lite conversion)

3. **Second session (epochs 35-70):**
   ```python
   !python train_kaggle.py -C cfg_2 --fold -1 --epochs 35 --batch_size 64 --grad_accumulation 8 --total_epochs 400 --resume datamount/weights/cfg_2/fold-1/checkpoint_last_seed{seed}.pth
   ```
   - Replace `{seed}` with the actual seed number from first session
   - Download the new checkpoint after completion (validation data not needed)

4. **Repeat until reaching 400 epochs:**
   - Continue uploading checkpoints and resuming until epoch 400
   - Each session trains 35 more epochs, continuing from the previous checkpoint
   - Use `--total_epochs 400` to ensure learning rate schedule is correct
   - **Final session (epoch 400):** Download both checkpoint and validation data for TF-Lite conversion

**Important:** Without `--resume`, each run starts with a **different random seed** (creating separate models for ensemble). Use `--resume` to continue the same seed.

**TPU Training (Full 400 epochs per seed):**
```python
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu --epochs 400 --batch_size 16 --grad_accumulation 32
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu --epochs 400 --batch_size 16 --grad_accumulation 32
```

**TPU Training (35 epochs per session - for Kaggle 9h limit):**

Follow the same multi-session approach as GPU, but with TPU parameters:
- Use `--use_tpu` flag
- Use `--batch_size 16 --grad_accumulation 32`
- Use `--total_epochs 400` with `--resume` for continuation

**OOM Errors?**
- GPU: Reduce `--batch_size`, increase `--grad_accumulation` (maintain effective batch size = 512)
- TPU: Use `--batch_size 8 --grad_accumulation 64`

### Round 3: TF-Lite Conversion

**Purpose:** Convert trained models to TF-Lite format for Kaggle submission.

```python
!python scripts/convert_cfg_2_to_tf_lite.py
```

**Output files:**
- `datamount/weights/cfg_2/fold-1/model.tflite`
- `datamount/weights/cfg_2/fold-1/inference_args.json`

These files can be added to a Kaggle kernel and submitted.

## Command-Line Parameters

- `--epochs N`: Number of training epochs (35 for Kaggle 9h limit, 400 for full training)
- `--batch_size N`: **Critical** - Adjust based on memory (GPU: 64, TPU: ≤16)
- `--grad_accumulation N`: **Critical** - Effective batch size = batch_size × grad_accumulation (GPU: 8, TPU: 32)
- `--fold N`: Fold number (-1 for full training, 0-3 for cross-validation)
- `--use_tpu`: Enable TPU training
- `--resume PATH`: Resume training from checkpoint (use with `--total_epochs` for multi-session training)
- `--total_epochs N`: Total epochs for learning rate schedule (use with `--resume` to continue long training)
