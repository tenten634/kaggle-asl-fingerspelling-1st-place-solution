"""
Debug script to find where division by zero occurs
"""
import os
import sys
import traceback

# Set up paths
BASEDIR = '/kaggle/working/kaggle-asl-fingerspelling-1st-place-solution'
os.chdir(BASEDIR)

for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

print("="*60)
print("DEBUGGING CONFIG LOAD")
print("="*60)

try:
    print("\n1. Loading cfg_2...")
    import importlib
    cfg_module = importlib.import_module('cfg_2')
    cfg = cfg_module.cfg
    print("✅ Config loaded successfully")
    
    print(f"\n2. Checking data files...")
    print(f"   inference_args.json: {os.path.exists(cfg.data_folder + 'inference_args.json')}")
    print(f"   train_df: {os.path.exists(cfg.train_df)}")
    print(f"   symmetry_fp: {os.path.exists(cfg.symmetry_fp)}")
    
    print(f"\n3. Loading dataset module...")
    dataset_module = importlib.import_module(cfg.dataset)
    print(f"✅ Dataset module imported: {cfg.dataset}")
    
    print(f"\n4. Checking inference_args.json content...")
    import json
    with open(cfg.data_folder + 'inference_args.json', 'r') as f:
        data = json.load(f)
    columns = data['selected_columns']
    print(f"   Number of columns: {len(columns)}")
    print(f"   First few columns: {columns[:5]}")
    
    # Check the division that might cause issues
    xyz_landmarks = columns
    num_landmarks = len(xyz_landmarks) // 3
    print(f"\n5. Checking landmark calculation...")
    print(f"   Total columns: {len(xyz_landmarks)}")
    print(f"   Expected landmarks: {num_landmarks}")
    print(f"   Division result: {len(xyz_landmarks) // 3}")
    
    if len(xyz_landmarks) == 0:
        print("   ❌ ERROR: xyz_landmarks is empty!")
    elif len(xyz_landmarks) % 3 != 0:
        print(f"   ⚠️  WARNING: {len(xyz_landmarks)} is not divisible by 3")
    else:
        print(f"   ✅ Landmark calculation looks OK")
    
    print(f"\n6. Testing dataset initialization (this might fail)...")
    import pandas as pd
    df = pd.read_csv(cfg.train_df)
    print(f"   DataFrame shape: {df.shape}")
    
    if len(df) == 0:
        print("   ❌ ERROR: DataFrame is empty!")
    else:
        print(f"   ✅ DataFrame has {len(df)} rows")
    
    # Try to create dataset
    try:
        CustomDataset = dataset_module.CustomDataset
        # Don't actually create it, just check if we can
        print(f"   ✅ CustomDataset class found")
    except Exception as e:
        print(f"   ❌ Error accessing CustomDataset: {e}")
        traceback.print_exc()
    
except ZeroDivisionError as e:
    print(f"\n❌ DIVISION BY ZERO ERROR!")
    print(f"   Error: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*60)

