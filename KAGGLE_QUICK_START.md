# Kaggle Quick Start Guide

## 🚀 Fast Setup (5 Minutes)

### Step 1: Create Notebook & Add Datasets
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Set **Accelerator** to **GPU P100** (or T4 for Pro)
4. Click **"Create"**

### Step 2: Add Datasets
In the **Data** sidebar (right side):
1. Click **"+ Add data"**
2. Search: `asl-fingerspelling-preprocessing-train-dataset` → **Add**
3. Click **"+ Add data"** again
4. Search: `asl-fingerspelling-preprocessed-supp-dataset` → **Add**

### Step 3: Run Setup Code
Copy-paste this into a code cell and run:

```python
# Clone repository
!git clone https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution.git
%cd kaggle-asl-fingerspelling-1st-place-solution

# Install packages
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0

# Setup data
import os, shutil
os.makedirs('/kaggle/working/datamount', exist_ok=True)

# Copy CSV/JSON files
for dataset_dir in os.listdir('/kaggle/input'):
    dataset_path = f'/kaggle/input/{dataset_dir}'
    if os.path.isdir(dataset_path):
        for file in os.listdir(dataset_path):
            if file.endswith(('.csv', '.json')):
                shutil.copy2(f'{dataset_path}/{file}', f'/kaggle/working/datamount/{file}')

# Copy landmarks
landmarks_dst = '/kaggle/working/datamount/train_landmarks_npy'
os.makedirs(landmarks_dst, exist_ok=True)

for dataset_dir in os.listdir('/kaggle/input'):
    dataset_path = f'/kaggle/input/{dataset_dir}'
    if 'train' in dataset_dir.lower() and 'preprocessing' in dataset_dir.lower():
        landmarks_path = f'{dataset_path}/train_landmarks_npy'
        if os.path.exists(landmarks_path):
            !cp -r {landmarks_path}/* {landmarks_dst}/
    elif 'supp' in dataset_dir.lower():
        landmarks_path = f'{dataset_path}/supplemental_landmarks'
        if os.path.exists(landmarks_path):
            !cp -r {landmarks_path}/* {landmarks_dst}/

# Verify
!python check_kaggle_environment.py
```

### Step 4: Start Training
```python
!python train_kaggle.py -C cfg_1 --fold 0
```

---

## 📋 Complete Training Workflow

### Round 1: Train 4 Folds
```python
!python train_kaggle.py -C cfg_1 --fold 0
!python train_kaggle.py -C cfg_1 --fold 1
!python train_kaggle.py -C cfg_1 --fold 2
!python train_kaggle.py -C cfg_1 --fold 3
```

### Generate OOF Predictions
```python
!python scripts/get_train_folded_oof_supp.py
```

### Round 2: Train Fullfit
```python
!python train_kaggle.py -C cfg_2 --fold -1
!python train_kaggle.py -C cfg_2 --fold -1  # Second seed
```

### Convert to TF-Lite
```python
!python scripts/convert_cfg_2_to_tf_lite.py
```

---

## 💡 Key Points

- **Data Location**: `/kaggle/input/` (read-only) → Copy to `/kaggle/working/`
- **Output Location**: `/kaggle/working/` (auto-saved to Output tab)
- **GPU**: Enable in Settings → Accelerator → GPU
- **Time Limit**: 9 hours per session (free tier)

---

## 🔧 Common Commands

```python
# Check environment
!python check_kaggle_environment.py

# List input datasets
!ls /kaggle/input/

# Check data
!ls /kaggle/working/datamount/

# Monitor GPU
!nvidia-smi

# Check disk space
!df -h
```

---

For detailed instructions, see **KAGGLE_SETUP.md**

