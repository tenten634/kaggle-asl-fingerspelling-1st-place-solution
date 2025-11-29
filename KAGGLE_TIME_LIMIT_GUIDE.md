# Kaggle GPU Time Limit - What Happens & How to Handle It

## ⏰ Kaggle Time Limits

- **Session Time Limit**: 9 hours per session (free tier)
- **GPU Time Limit**: 30 hours per week (free tier)
- **What Happens**: When time limit is reached, the notebook session stops/kills the process

---

## 🛡️ What Gets Saved Automatically

### ✅ Saved to Output Tab

All files in `/kaggle/working/` are **automatically saved** to the **Output** tab when:
- Training completes normally
- Session times out
- You manually stop the notebook

**This includes:**
- Model checkpoints (`datamount/weights/cfg_2/fold-1/checkpoint_last_seed{seed}.pth`)
- Validation data
- Any other files created during training

### ⚠️ Important: Checkpoint Saving Behavior

Looking at `cfg_2.py`:
- `cfg.save_only_last_ckpt = True` - **Only saves the final checkpoint**
- Checkpoints are saved:
  - At the end of each epoch (if `save_only_last_ckpt = False`)
  - **At the very end** when training completes (always)

**This means:**
- ✅ If training completes → Checkpoint is saved
- ⚠️ If session times out mid-epoch → **Last checkpoint might not be saved yet**
- ✅ Files are still saved to Output tab (even if incomplete)

---

## 🔍 Check What Was Saved

Run this after a session ends (or times out):

```python
%cd kaggle-asl-fingerspelling-1st-place-solution

import os

checkpoint_dir = 'datamount/weights/cfg_2/fold-1'
print("="*60)
print("CHECKING SAVED CHECKPOINTS")
print("="*60)

if os.path.exists(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    checkpoints = [f for f in files if 'checkpoint' in f]
    
    if checkpoints:
        print(f"✅ Found {len(checkpoints)} checkpoint(s):")
        for ckpt in sorted(checkpoints):
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            size = os.path.getsize(ckpt_path) / (1024*1024)  # MB
            mtime = os.path.getmtime(ckpt_path)
            from datetime import datetime
            time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   - {ckpt}")
            print(f"     Size: {size:.2f} MB")
            print(f"     Last modified: {time_str}")
    else:
        print("❌ No checkpoints found")
        print("   Training may have been interrupted before saving")
else:
    print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
    print("   Training may not have started or was interrupted early")

print("\n" + "="*60)
```

---

## 🔄 Resume Training (If Interrupted)

The current code **doesn't have built-in resume functionality**, but you can manually resume by:

### Option 1: Check Epoch Progress

If training was interrupted, check how many epochs completed:

```python
# Check Neptune logs (if you're using Neptune)
# Or check the last validation output in the notebook

# Then manually adjust the epoch range:
# Instead of: for epoch in range(cfg.epochs):
# Use: for epoch in range(start_epoch, cfg.epochs):
```

### Option 2: Modify Code to Resume (Advanced)

You would need to:
1. Load the checkpoint
2. Load optimizer state
3. Load scheduler state
4. Resume from the correct epoch

**This requires code modifications.**

---

## 💡 Best Practices to Avoid Issues

### 1. Monitor Training Progress

```python
# Add this to track progress
import time
start_time = time.time()

# In training loop, periodically check:
if time.time() - start_time > 8 * 3600:  # 8 hours
    print("⚠️  Approaching 9-hour limit! Saving checkpoint...")
    # Force save checkpoint
    torch.save({"model": model.state_dict()}, 
              f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_emergency_seed{cfg.seed}.pth")
```

### 2. Save More Frequently

You could modify the config to save more often:

```python
# In cfg_2.py, change:
cfg.save_only_last_ckpt = False  # Save every epoch instead of just at end
```

### 3. Use Multiple Sessions

Since each seed takes ~6-10 hours:
- **Seed 1**: Run in one session
- **Seed 2**: Run in a new session (after Seed 1 completes)

This avoids hitting the 9-hour limit.

---

## 📊 What You'll See in Output Tab

After a session ends (completed or timed out), check the **Output** tab:

```
datamount/
└── weights/
    └── cfg_2/
        └── fold-1/
            ├── checkpoint_last_seed{seed}.pth  (if saved)
            └── val_data_seed{seed}.pth  (if validation ran)
```

**Download these files** - they're your saved progress!

---

## ⚠️ If Training Was Interrupted

### Scenario 1: Training Completed
- ✅ Checkpoint is saved
- ✅ You can proceed to Seed 2 or TF-Lite conversion

### Scenario 2: Training Interrupted Mid-Epoch
- ⚠️ Last checkpoint might not be saved (if `save_only_last_ckpt = True`)
- ✅ Previous epoch's checkpoint might exist (if `save_only_last_ckpt = False`)
- ✅ You can download what was saved from Output tab

### Scenario 3: Training Interrupted Early
- ❌ No checkpoint saved yet
- ⚠️ You'll need to restart training

---

## 🎯 Recommendation

For **cfg_2** (400 epochs, ~6-10 hours):
- **Most likely**: Training will complete within 9 hours
- **If it doesn't**: The checkpoint at the last completed epoch will be in Output tab
- **Best practice**: Check Output tab after each session to verify what was saved

**For peace of mind**, you could modify the code to save checkpoints more frequently, but the default (save at end) is usually fine since training typically completes within the time limit.

---

## 📝 Quick Check Script

Run this to see what was saved:

```python
import os
from datetime import datetime

checkpoint_dir = 'datamount/weights/cfg_2/fold-1'
if os.path.exists(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    checkpoints = [f for f in files if 'checkpoint' in f and f.endswith('.pth')]
    
    if checkpoints:
        print(f"✅ Found {len(checkpoints)} checkpoint(s)")
        for ckpt in sorted(checkpoints):
            path = os.path.join(checkpoint_dir, ckpt)
            size = os.path.getsize(path) / (1024*1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            print(f"   {ckpt}: {size:.2f} MB, saved at {mtime}")
    else:
        print("❌ No checkpoints found")
else:
    print("❌ Checkpoint directory doesn't exist")
```

