"""
Quick script to check and copy inference_args.json if needed
"""
import os
import json
import shutil

print("="*60)
print("CHECKING inference_args.json")
print("="*60)

# Check in working directory
working_path = 'datamount/train_landmarks_npy/inference_args.json'
if os.path.exists(working_path):
    try:
        with open(working_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Found at: {working_path}")
        print(f"   Keys: {list(data.keys())}")
        if 'selected_columns' in data:
            print(f"   Selected columns: {len(data['selected_columns'])} columns")
    except Exception as e:
        print(f"❌ File exists but error reading: {e}")
else:
    print(f"❌ Not found at: {working_path}")

# Check in input datasets
print("\n📦 Checking input datasets...")
input_base = '/kaggle/input'
found = False

for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if os.path.isdir(dataset_path):
        landmarks_path = os.path.join(dataset_path, 'train_landmarks_npy')
        if os.path.exists(landmarks_path):
            inference_file = os.path.join(landmarks_path, 'inference_args.json')
            if os.path.exists(inference_file):
                print(f"✅ Found in: {dataset_dir}")
                print(f"   Path: {inference_file}")
                
                # Try to read it
                try:
                    with open(inference_file, 'r') as f:
                        data = json.load(f)
                    print(f"   ✅ File is valid JSON")
                    print(f"   Keys: {list(data.keys())}")
                    if 'selected_columns' in data:
                        print(f"   Selected columns: {len(data['selected_columns'])} columns")
                    
                    # Copy to working directory if needed
                    dst = 'datamount/train_landmarks_npy/inference_args.json'
                    if not os.path.exists(dst) or os.path.islink(dst):
                        # If it's a symlink or doesn't exist, copy the file
                        if os.path.islink('datamount/train_landmarks_npy'):
                            # Copy to the actual directory, not the symlink
                            os.makedirs('datamount/train_landmarks_npy', exist_ok=True)
                            if os.path.islink('datamount/train_landmarks_npy'):
                                os.unlink('datamount/train_landmarks_npy')
                                os.makedirs('datamount/train_landmarks_npy', exist_ok=True)
                        
                        shutil.copy2(inference_file, dst)
                        print(f"   ✅ Copied to: {dst}")
                    found = True
                    break
                except Exception as e:
                    print(f"   ❌ Error reading file: {e}")

if not found:
    print("\n⚠️  inference_args.json not found in any dataset")
    print("   You may need to copy it manually or use default columns")

print("\n" + "="*60)

