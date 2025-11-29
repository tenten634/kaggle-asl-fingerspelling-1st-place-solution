# Kaggle Notebook - Copy-Paste Code Cells

Copy these cells into your Kaggle notebook in order. Each cell verifies a step before moving to the next.

---

## Cell 1: Clone Repository

```python
# Clone the repository
!git clone https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution.git
%cd kaggle-asl-fingerspelling-1st-place-solution
print("✅ Repository cloned")
```

---

## Cell 2: Install Packages

```python
# Install required packages
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0
print("✅ Packages installed")
```

---

## Cell 3: Prepare Data

```python
import os
import shutil

# Create data directory
os.makedirs('/kaggle/working/datamount', exist_ok=True)
print("✅ Created datamount directory")

# Copy CSV and JSON files from input datasets
input_base = '/kaggle/input'
copied_files = []

for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if os.path.isdir(dataset_path):
        for file in os.listdir(dataset_path):
            if file.endswith(('.csv', '.json')):
                src = os.path.join(dataset_path, file)
                dst = os.path.join('/kaggle/working/datamount', file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied_files.append(file)
                    print(f"✅ Copied {file}")

print(f"\n📊 Copied {len(copied_files)} files: {copied_files}")

# Create landmarks directory
landmarks_dst = '/kaggle/working/datamount/train_landmarks_npy'
os.makedirs(landmarks_dst, exist_ok=True)
print(f"✅ Created {landmarks_dst}")

# Copy training landmarks
print("\n📦 Copying training landmarks...")
for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if 'train' in dataset_dir.lower() and 'preprocessing' in dataset_dir.lower():
        landmarks_path = os.path.join(dataset_path, 'train_landmarks_npy')
        if os.path.exists(landmarks_path):
            print(f"   Found training landmarks in: {dataset_dir}")
            !cp -r {landmarks_path}/* {landmarks_dst}/
            print("   ✅ Training landmarks copied")

# Copy supplemental landmarks
print("\n📦 Copying supplemental landmarks...")
for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if 'supp' in dataset_dir.lower():
        landmarks_path = os.path.join(dataset_path, 'supplemental_landmarks')
        if os.path.exists(landmarks_path):
            print(f"   Found supplemental landmarks in: {dataset_dir}")
            !cp -r {landmarks_path}/* {landmarks_dst}/
            print("   ✅ Supplemental landmarks copied and merged")

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE")
print("="*60)
```

---

## Cell 4: Environment Check (CRITICAL!)

```python
!python check_kaggle_environment.py
```

**⚠️ IMPORTANT**: Review the output carefully. All items should show ✅ before proceeding.

---

## Cell 5: Data Verification

```python
import os

# Check required files
required_files = [
    'datamount/character_to_prediction_index.json',
    'datamount/train_folded.csv',
    'datamount/symmetry.csv',
    'datamount/train_landmarks_npy'
]

print("="*60)
print("DATA VERIFICATION")
print("="*60)

for file_path in required_files:
    if os.path.exists(file_path):
        if os.path.isdir(file_path):
            count = len([f for f in os.listdir(file_path) if f.endswith('.npy')])
            print(f"✅ {file_path}/ ({count} .npy files)")
        else:
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"✅ {file_path} ({size:.2f} MB)")
    else:
        print(f"❌ {file_path} - NOT FOUND")

# Count landmarks
landmarks_dir = 'datamount/train_landmarks_npy'
if os.path.exists(landmarks_dir):
    npy_files = [f for f in os.listdir(landmarks_dir) if f.endswith('.npy')]
    print(f"\n📊 Total landmark files: {len(npy_files)}")
    if len(npy_files) > 0:
        print("✅ Landmarks directory is ready")
    else:
        print("⚠️  Landmarks directory is empty - check data copy step")
```

---

## Cell 6: GPU Verification

```python
import torch

print("="*60)
print("GPU VERIFICATION")
print("="*60)

if torch.cuda.is_available():
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU Count: {torch.cuda.device_count()}")
    print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"✅ GPU computation test: SUCCESS")
    print(f"   Result shape: {z.shape}")
else:
    print("❌ CUDA NOT AVAILABLE")
    print("   Enable GPU in Settings → Accelerator → GPU")
```

---

## Cell 7: Final Checklist

```python
import os
import torch

print("="*60)
print("PRE-TRAINING CHECKLIST")
print("="*60)

checks = {
    "Repository cloned": os.path.exists('train_kaggle.py'),
    "Configs exist": os.path.exists('configs/cfg_1.py'),
    "Data directory exists": os.path.exists('datamount'),
    "Train CSV exists": os.path.exists('datamount/train_folded.csv'),
    "Landmarks exist": os.path.exists('datamount/train_landmarks_npy'),
    "GPU available": torch.cuda.is_available(),
}

all_passed = True
for check_name, status in checks.items():
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {check_name}")
    if not status:
        all_passed = False

print("\n" + "="*60)
if all_passed:
    print("🎉 ALL CHECKS PASSED - READY TO TRAIN!")
    print("="*60)
    print("\nNext step: Run training with:")
    print("  !python train_kaggle.py -C cfg_1 --fold 0")
else:
    print("⚠️  SOME CHECKS FAILED - FIX ISSUES ABOVE")
    print("="*60)
```

---

## Cell 8: Start Training (Only after all checks pass!)

```python
# Round 1 - Fold 0
!python train_kaggle.py -C cfg_1 --fold 0
```

---

## Notes

- Run cells **in order** (1 → 7)
- **Don't skip Cell 4** (Environment Check) - it's critical!
- Only proceed to Cell 8 if **ALL checks in Cell 7 pass**
- Each cell should complete without errors before moving to the next

---

For detailed explanations, see **KAGGLE_STEP_BY_STEP.md**

