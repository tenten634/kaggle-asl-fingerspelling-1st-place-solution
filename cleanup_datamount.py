"""
Clean up unnecessary files in datamount directory
Keeps essential files: CSV, JSON, weights, and inference_args.json
Removes: empty directories, symlinks that point to input, duplicate data
"""
import os
import shutil

print("="*60)
print("CLEANING UP DATAMOUNT DIRECTORY")
print("="*60)

datamount_dir = 'datamount'

if not os.path.exists(datamount_dir):
    print(f"❌ {datamount_dir} directory doesn't exist")
    exit()

# Files/directories to keep
essential_files = [
    'train_folded.csv',
    'train_folded_oof_supp.csv',
    'supplemental_metadata_folded.csv',
    'character_to_prediction_index.json',
    'symmetry.csv',
    'inference_args.json',
]

essential_dirs = [
    'weights',  # Model checkpoints
    'train_landmarks_npy',  # Only if it's not a symlink to input
]

print(f"\n📁 Analyzing {datamount_dir}...")

# Check what's in datamount
items = os.listdir(datamount_dir)
print(f"   Found {len(items)} items")

# Check train_landmarks_npy
landmarks_path = os.path.join(datamount_dir, 'train_landmarks_npy')
if os.path.exists(landmarks_path):
    if os.path.islink(landmarks_path):
        link_target = os.readlink(landmarks_path)
        print(f"\n🔗 Found symlink: train_landmarks_npy -> {link_target}")
        if '/kaggle/input' in link_target:
            print("   ⚠️  Points to input directory - can be removed (using input directly now)")
            response = input("   Remove this symlink? (y/n): ").strip().lower()
            if response == 'y':
                os.unlink(landmarks_path)
                print("   ✅ Removed symlink")
        else:
            print("   ℹ️  Points to working directory - keeping it")
    elif os.path.isdir(landmarks_path):
        items_in_landmarks = os.listdir(landmarks_path)
        if len(items_in_landmarks) == 0:
            print(f"\n📂 train_landmarks_npy is empty")
            response = input("   Remove empty directory? (y/n): ").strip().lower()
            if response == 'y':
                os.rmdir(landmarks_path)
                print("   ✅ Removed empty directory")
        else:
            # Check if it's actually being used (has subdirectories with .npy files)
            has_data = False
            for item in items_in_landmarks[:5]:  # Check first 5
                item_path = os.path.join(landmarks_path, item)
                if os.path.isdir(item_path):
                    files = os.listdir(item_path)
                    if any(f.endswith('.npy') for f in files):
                        has_data = True
                        break
            
            if has_data:
                print(f"   ✅ Contains {len(items_in_landmarks)} directories with data - keeping it")
            else:
                print(f"   ⚠️  Contains {len(items_in_landmarks)} items but may be empty/unused")
                response = input("   Remove it? (y/n): ").strip().lower()
                if response == 'y':
                    shutil.rmtree(landmarks_path)
                    print("   ✅ Removed directory")

# Check for other unnecessary files/directories
print(f"\n🧹 Checking for unnecessary files...")
removed = []
kept = []

for item in items:
    item_path = os.path.join(datamount_dir, item)
    
    # Skip essential files
    if item in essential_files:
        if os.path.exists(item_path):
            kept.append(item)
            continue
    
    # Skip essential directories
    if item in essential_dirs:
        if os.path.exists(item_path):
            kept.append(item)
            continue
    
    # Check if it's a file or directory
    if os.path.isfile(item_path):
        # Check if it's a temporary or unnecessary file
        if item.startswith('.') or item.endswith('.tmp') or item.endswith('.bak'):
            print(f"   🗑️  Removing temporary file: {item}")
            os.remove(item_path)
            removed.append(item)
        else:
            print(f"   ⚠️  Unknown file: {item}")
            response = input(f"      Remove {item}? (y/n): ").strip().lower()
            if response == 'y':
                os.remove(item_path)
                removed.append(item)
            else:
                kept.append(item)
    elif os.path.isdir(item_path):
        # Check if directory is empty
        try:
            dir_items = os.listdir(item_path)
            if len(dir_items) == 0:
                print(f"   🗑️  Removing empty directory: {item}")
                os.rmdir(item_path)
                removed.append(item)
            else:
                print(f"   ⚠️  Unknown directory: {item} ({len(dir_items)} items)")
                response = input(f"      Remove {item}? (y/n): ").strip().lower()
                if response == 'y':
                    shutil.rmtree(item_path)
                    removed.append(item)
                else:
                    kept.append(item)
        except PermissionError:
            print(f"   ⚠️  Cannot access {item} (permission denied)")

# Summary
print(f"\n" + "="*60)
print("CLEANUP SUMMARY")
print("="*60)
print(f"✅ Kept: {len(kept)} items")
for item in kept:
    print(f"   - {item}")

if removed:
    print(f"\n🗑️  Removed: {len(removed)} items")
    for item in removed:
        print(f"   - {item}")
else:
    print(f"\n🗑️  Removed: 0 items (nothing to remove)")

# Check disk space
print(f"\n💾 Disk space check:")
total, used, free = shutil.disk_usage(datamount_dir)
print(f"   Free space: {free / (1024**3):.2f} GB")

print("\n" + "="*60)
print("CLEANUP COMPLETE")
print("="*60)

