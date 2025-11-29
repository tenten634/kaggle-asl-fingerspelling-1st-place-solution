# Git Pull Workflow in Kaggle - Safe Update Guide

## ⚠️ Important: What Git Pull Affects

### ✅ Safe - Git Pull WON'T Affect:

1. **Data files** (not tracked by git):
   - `datamount/train_landmarks_npy/` - Your data directory
   - `datamount/*.csv` - CSV files
   - `datamount/*.json` - JSON files (except those in repo)
   - Any files you created or copied

2. **Output files** (not tracked by git):
   - `datamount/weights/` - Model checkpoints
   - `datamount/weights/*/checkpoint_*.pth` - Saved models
   - Any training outputs

3. **Symlinks** - These are filesystem links, not git-tracked

### ❌ Will Be Overwritten - Git Pull WILL Affect:

1. **All tracked Python files**:
   - `train_kaggle.py`
   - `check_kaggle_environment.py`
   - `configs/*.py`
   - `data/*.py`
   - `models/*.py`
   - Any other `.py` files in the repo

2. **Documentation files**:
   - `*.md` files
   - `README.md`

3. **Any modifications you made to tracked files**

---

## 🔍 Check What Will Be Affected Before Pulling

### Step 1: Check Git Status

```python
%cd kaggle-asl-fingerspelling-1st-place-solution

# Check what files have been modified
!git status

# See what changes you have locally
!git diff
```

### Step 2: Check What Will Be Updated

```python
# See what commits you'll pull
!git fetch
!git log HEAD..origin/main --oneline

# See what files will change
!git diff HEAD origin/main --name-only
```

---

## ✅ Safe Git Pull Workflow

### Option A: Pull Without Losing Local Changes (Recommended)

```python
%cd kaggle-asl-fingerspelling-1st-place-solution

# 1. Check if you have local changes
!git status

# 2. If you have changes you want to keep, stash them
!git stash save "Local changes before pull"

# 3. Pull updates
!git pull

# 4. If you stashed, restore your changes (if needed)
# !git stash pop  # Only if you want to restore stashed changes
```

### Option B: Force Update (Discard Local Changes)

```python
%cd kaggle-asl-fingerspelling-1st-place-solution

# 1. Discard any local changes to tracked files
!git reset --hard HEAD

# 2. Pull updates
!git pull
```

### Option C: Check First, Then Decide

```python
%cd kaggle-asl-fingerspelling-1st-place-solution

# 1. See what's different
!git fetch
!git diff HEAD origin/main --stat

# 2. If you're okay with the changes, pull
!git pull

# 3. If you have local changes you want to keep, use Option A
```

---

## 🛡️ Protecting Your Setup

### Files That Are Safe (Not in Git):

Your setup creates these files which are **NOT tracked by git**:

```
datamount/
├── train_landmarks_npy/     # ✅ Safe - not in git
├── train_folded.csv         # ✅ Safe - not in git  
├── train_folded_oof_supp.csv # ✅ Safe - not in git
├── character_to_prediction_index.json # ⚠️  Check - might be in git
├── symmetry.csv             # ⚠️  Check - might be in git
└── weights/                 # ✅ Safe - not in git
    └── cfg_2/
        └── fold-1/
            └── checkpoint_*.pth
```

### Verify What's Tracked

```python
# Check if a file is tracked by git
!git ls-files | grep datamount

# Check what files in datamount are tracked
!git ls-files datamount/
```

---

## 🔄 Recommended Workflow

### During Setup (First Time):

1. **Clone repository** ✅
2. **Setup data** ✅ (creates untracked files - safe)
3. **Verify environment** ✅
4. **Start training** ✅

### When Updating Code:

```python
# Safe update workflow
%cd kaggle-asl-fingerspelling-1st-place-solution

# 1. Check what will change
print("Checking what will be updated...")
!git fetch
changed_files = !git diff HEAD origin/main --name-only
print(f"Files that will change: {len(changed_files)}")

# 2. If you're okay, pull
!git pull

# 3. Verify your data is still there
import os
if os.path.exists('datamount/train_landmarks_npy'):
    print("✅ Data directory still exists")
else:
    print("⚠️  Data directory missing - may need to recreate symlink")
```

---

## ⚠️ Common Scenarios

### Scenario 1: You Modified a Config File

```python
# If you modified configs/cfg_2.py and want to keep changes:
!git stash
!git pull
!git stash pop  # Restore your changes (may have conflicts)

# If you want to discard your changes:
!git checkout -- configs/cfg_2.py
!git pull
```

### Scenario 2: You Modified train_kaggle.py

```python
# Check what you changed
!git diff train_kaggle.py

# If you want to keep changes, stash first
!git stash
!git pull
!git stash pop

# If you want the updated version, discard changes
!git checkout -- train_kaggle.py
!git pull
```

### Scenario 3: Data Directory Disappears After Pull

This shouldn't happen, but if it does:

```python
# Recreate the symlink (if you were using symlinks)
import os

landmarks_dir = 'datamount/train_landmarks_npy'
if not os.path.exists(landmarks_dir):
    # Find the source
    for dataset_dir in os.listdir('/kaggle/input'):
        dataset_path = f'/kaggle/input/{dataset_dir}'
        if 'train' in dataset_dir.lower() and 'preprocessing' in dataset_dir.lower():
            landmarks_path = f'{dataset_path}/train_landmarks_npy'
            if os.path.exists(landmarks_path):
                os.makedirs('datamount', exist_ok=True)
                os.symlink(landmarks_path, landmarks_dir)
                print(f"✅ Recreated symlink: {landmarks_dir}")
                break
```

---

## 📋 Quick Reference

| Action | Safe? | What Happens |
|--------|-------|--------------|
| `git pull` | ✅ Usually | Updates tracked files, keeps untracked files |
| `git pull` with local changes | ⚠️  May conflict | Git will warn about conflicts |
| `git reset --hard` then `git pull` | ✅ Safe | Discards local changes, gets latest |
| `git stash` then `git pull` | ✅ Safe | Saves changes, pulls, can restore |
| Data files | ✅ Always safe | Not tracked, never affected |
| Checkpoints | ✅ Always safe | Not tracked, never affected |
| Symlinks | ✅ Usually safe | Filesystem links, not git-tracked |

---

## 💡 Best Practice

**Before pulling, always:**

1. ✅ Check what will change: `!git fetch && !git diff HEAD origin/main --stat`
2. ✅ Verify your data exists: `!ls datamount/train_landmarks_npy/`
3. ✅ Stash local changes if you have any: `!git stash`
4. ✅ Pull: `!git pull`
5. ✅ Verify data still exists: `!ls datamount/train_landmarks_npy/`

**Your data and checkpoints are always safe!** 🛡️

