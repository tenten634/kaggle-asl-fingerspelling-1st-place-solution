"""
Verify data structure matches original repository requirements
"""
import os
import json

print("="*60)
print("VERIFYING DATA STRUCTURE (vs Original Requirements)")
print("="*60)

# Expected structure based on README.md
expected_files = {
    'datamount/train_folded.csv': 'Required CSV file',
    'datamount/train_folded_oof_supp.csv': 'Required CSV file (output of step 1)',
    'datamount/character_to_prediction_index.json': 'Required JSON file',
    'datamount/symmetry.csv': 'Required CSV file',
    'datamount/supplemental_metadata_folded.csv': 'Optional CSV file',
}

expected_dirs = {
    'datamount/train_landmarks_npy': 'Required - merged train + supplemental landmarks',
    'datamount/weights': 'Optional - for model checkpoints',
}

print("\n📋 Checking Required Files:")
all_good = True
for file_path, description in expected_files.items():
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"   ✅ {file_path} ({size:.2f} MB) - {description}")
    else:
        print(f"   ❌ {file_path} - MISSING - {description}")
        all_good = False

print("\n📁 Checking Required Directories:")
for dir_path, description in expected_dirs.items():
    if os.path.exists(dir_path):
        if os.path.islink(dir_path):
            print(f"   ⚠️  {dir_path} is a SYMLINK (should be real directory)")
            print(f"      Points to: {os.readlink(dir_path)}")
            all_good = False
        elif os.path.isdir(dir_path):
            items = os.listdir(dir_path)
            dirs = [i for i in items if os.path.isdir(os.path.join(dir_path, i))]
            files = [i for i in items if os.path.isfile(os.path.join(dir_path, i))]
            print(f"   ✅ {dir_path}/ ({len(dirs)} dirs, {len(files)} files) - {description}")
        else:
            print(f"   ❌ {dir_path} exists but is not a directory")
            all_good = False
    else:
        print(f"   ❌ {dir_path} - MISSING - {description}")
        all_good = False

# Detailed check of train_landmarks_npy
print("\n🔍 Detailed Check: datamount/train_landmarks_npy/")
landmarks_dir = 'datamount/train_landmarks_npy'
if os.path.exists(landmarks_dir) and os.path.isdir(landmarks_dir):
    items = os.listdir(landmarks_dir)
    dirs = [i for i in items if os.path.isdir(os.path.join(landmarks_dir, i))]
    files = [i for i in items if os.path.isfile(os.path.join(landmarks_dir, i))]
    
    print(f"   Total directories: {len(dirs)}")
    print(f"   Total files: {len(files)}")
    
    # Check for inference_args.json
    inference_file = os.path.join(landmarks_dir, 'inference_args.json')
    if os.path.exists(inference_file):
        print(f"   ✅ inference_args.json found")
        try:
            with open(inference_file, 'r') as f:
                data = json.load(f)
            if 'selected_columns' in data:
                print(f"      Has {len(data['selected_columns'])} selected columns")
        except:
            print(f"      ⚠️  Could not read JSON")
    else:
        print(f"   ⚠️  inference_args.json not found (may be in input dataset)")
    
    # Sample a few directories to verify structure
    if len(dirs) > 0:
        print(f"\n   Sample directories:")
        for i, dir_name in enumerate(dirs[:5]):
            dir_path = os.path.join(landmarks_dir, dir_name)
            try:
                files_in_dir = os.listdir(dir_path)
                npy_files = [f for f in files_in_dir if f.endswith('.npy')]
                print(f"      {dir_name}/: {len(npy_files)} .npy files")
            except:
                print(f"      {dir_name}/: (cannot access)")
    
    # Expected: 68 (train) + 53 (supp) = 121 directories
    expected_total = 121
    if len(dirs) == expected_total:
        print(f"\n   ✅ Perfect! Has {len(dirs)} directories (68 train + 53 supp = 121)")
    elif len(dirs) > expected_total:
        print(f"\n   ⚠️  Has {len(dirs)} directories (expected {expected_total}) - may have duplicates")
    else:
        print(f"\n   ⚠️  Has {len(dirs)} directories (expected {expected_total}) - may be incomplete")

# Check config expectations
print("\n⚙️  Checking Config Expectations:")
try:
    import sys
    sys.path.append('configs')
    import cfg_2
    cfg = cfg_2.cfg
    
    print(f"   cfg.data_folder: {cfg.data_folder}")
    if os.path.exists(cfg.data_folder):
        print(f"      ✅ Exists")
    else:
        print(f"      ❌ Does not exist")
        all_good = False
    
    print(f"   cfg.train_df: {cfg.train_df}")
    if os.path.exists(cfg.train_df):
        print(f"      ✅ Exists")
    else:
        print(f"      ❌ Does not exist")
        all_good = False
    
    print(f"   cfg.symmetry_fp: {cfg.symmetry_fp}")
    if os.path.exists(cfg.symmetry_fp):
        print(f"      ✅ Exists")
    else:
        print(f"      ❌ Does not exist")
        all_good = False
except Exception as e:
    print(f"   ⚠️  Could not check config: {e}")

# Compare to original README structure
print("\n📖 Original README Structure (from README.md):")
print("   Expected after setup:")
print("   datamount/")
print("   ├── train_landmarks_npy/     # Merged: train + supplemental")
print("   │   ├── {file_id}/")
print("   │   │   └── {sequence_id}.npy")
print("   │   └── inference_args.json")
print("   ├── train_folded.csv")
print("   ├── train_folded_oof_supp.csv")
print("   ├── character_to_prediction_index.json")
print("   └── symmetry.csv")

# Final verdict
print("\n" + "="*60)
if all_good and os.path.exists(landmarks_dir) and not os.path.islink(landmarks_dir):
    items = os.listdir(landmarks_dir)
    dirs = [i for i in items if os.path.isdir(os.path.join(landmarks_dir, i))]
    if len(dirs) >= 121:
        print("✅ STRUCTURE LOOKS GOOD!")
        print("   Matches original repository requirements")
        print("   Ready for training!")
    else:
        print("⚠️  STRUCTURE INCOMPLETE")
        print(f"   Has {len(dirs)} directories, expected at least 121")
else:
    print("❌ STRUCTURE ISSUES DETECTED")
    print("   Please fix the issues above")

print("="*60)

