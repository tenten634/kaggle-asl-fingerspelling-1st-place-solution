"""
Diagnose current state in Kaggle to understand the issue
"""
import os
import sys

print("="*60)
print("KAGGLE STATE DIAGNOSIS")
print("="*60)

# Check current directory
print(f"\n📁 Current directory: {os.getcwd()}")

# Check if we're in the repo
repo_dir = 'kaggle-asl-fingerspelling-1st-place-solution'
if os.path.exists(repo_dir):
    print(f"✅ Repository directory exists: {repo_dir}")
    os.chdir(repo_dir)
    print(f"   Changed to: {os.getcwd()}")
else:
    print(f"⚠️  Repository directory not found")

# Check input datasets
print(f"\n📦 Input Datasets (/kaggle/input/):")
input_base = '/kaggle/input'
if os.path.exists(input_base):
    datasets = []
    for dataset_name in sorted(os.listdir(input_base)):
        dataset_path = os.path.join(input_base, dataset_name)
        if os.path.isdir(dataset_path):
            datasets.append(dataset_name)
            print(f"\n   📁 {dataset_name}/")
            
            # Check contents
            items = os.listdir(dataset_path)
            for item in sorted(items):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    try:
                        sub_items = os.listdir(item_path)
                        if item in ['train_landmarks_npy', 'supplemental_landmarks']:
                            dirs = [i for i in sub_items if os.path.isdir(os.path.join(item_path, i))]
                            files = [i for i in sub_items if os.path.isfile(os.path.join(item_path, i))]
                            print(f"      📂 {item}/ ({len(dirs)} dirs, {len(files)} files)")
                            if 'inference_args.json' in files:
                                print(f"         ✅ Has inference_args.json")
                        else:
                            print(f"      📂 {item}/ ({len(sub_items)} items)")
                    except:
                        print(f"      📂 {item}/ (cannot access)")
                else:
                    size = os.path.getsize(item_path) / (1024*1024)
                    print(f"      📄 {item} ({size:.2f} MB)")
    
    print(f"\n   Total datasets found: {len(datasets)}")
else:
    print(f"   ❌ /kaggle/input doesn't exist")

# Simulate the detection logic from train_kaggle.py
print(f"\n🔍 Simulating Detection Logic:")
if os.path.exists(input_base):
    train_landmarks_found = False
    for dataset_name in os.listdir(input_base):
        dataset_path = os.path.join(input_base, dataset_name)
        if os.path.isdir(dataset_path):
            train_landmarks_path = os.path.join(dataset_path, 'train_landmarks_npy')
            print(f"\n   Checking: {dataset_name}/train_landmarks_npy")
            if os.path.exists(train_landmarks_path):
                print(f"      ✅ Exists")
                try:
                    items = os.listdir(train_landmarks_path)
                    dirs = [i for i in items if os.path.isdir(os.path.join(train_landmarks_path, i))]
                    files = [i for i in items if os.path.isfile(os.path.join(train_landmarks_path, i))]
                    print(f"      Directories: {len(dirs)}")
                    print(f"      Files: {len(files)}")
                    if len(dirs) > 10:
                        print(f"      ✅ Would be selected (has {len(dirs)} > 10 directories)")
                        abs_path = os.path.realpath(train_landmarks_path) + '/'
                        print(f"      Would set cfg.data_folder = '{abs_path}'")
                        train_landmarks_found = True
                    else:
                        print(f"      ⚠️  Would NOT be selected (only {len(dirs)} directories)")
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            else:
                print(f"      ❌ Does not exist")
    
    if not train_landmarks_found:
        print(f"\n   ⚠️  No suitable training landmarks found!")

# Check what cfg.data_folder would be set to
print(f"\n⚙️  Config Check:")
try:
    # Set up paths like train_kaggle.py does
    BASEDIR = os.getcwd()
    for DIRNAME in 'configs data models postprocess metrics'.split():
        sys.path.append(f'{BASEDIR}/{DIRNAME}/')
    
    import importlib
    cfg_module = importlib.import_module('cfg_2')
    cfg = cfg_module.cfg
    
    print(f"   Original cfg.data_folder: '{cfg.data_folder}'")
    print(f"   Exists: {os.path.exists(cfg.data_folder.rstrip('/'))}")
    
    # Check inference_args.json
    inference_path = cfg.data_folder + 'inference_args.json'
    print(f"\n   inference_args.json path: '{inference_path}'")
    print(f"   Exists: {os.path.exists(inference_path)}")
    
    if not os.path.exists(inference_path):
        print(f"   ⚠️  NOT FOUND - will need to search in input")
        # Search for it
        if os.path.exists(input_base):
            for dataset_name in os.listdir(input_base):
                dataset_path = os.path.join(input_base, dataset_name)
                train_landmarks = os.path.join(dataset_path, 'train_landmarks_npy')
                if os.path.exists(train_landmarks):
                    alt_path = os.path.join(train_landmarks, 'inference_args.json')
                    if os.path.exists(alt_path):
                        print(f"   ✅ Found in: {alt_path}")
                        break
    
except Exception as e:
    print(f"   ❌ Error loading config: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)

