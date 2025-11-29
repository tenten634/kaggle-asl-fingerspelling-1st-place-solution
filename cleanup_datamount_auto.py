"""
Automatically clean up unnecessary files in datamount directory (non-interactive)
Removes: symlinks to input, empty directories, temporary files
Keeps: CSV, JSON, weights, and data directories with actual content
"""
import os
import shutil

print("="*60)
print("AUTOMATIC CLEANUP OF DATAMOUNT DIRECTORY")
print("="*60)

datamount_dir = 'datamount'

if not os.path.exists(datamount_dir):
    print(f"❌ {datamount_dir} directory doesn't exist")
    exit()

# Essential files to always keep
essential_files = {
    'train_folded.csv',
    'train_folded_oof_supp.csv',
    'supplemental_metadata_folded.csv',
    'character_to_prediction_index.json',
    'symmetry.csv',
}

# Essential directories to keep
essential_dirs = {'weights'}

print(f"\n📁 Analyzing {datamount_dir}...")
items = os.listdir(datamount_dir)
print(f"   Found {len(items)} items")

removed = []
kept = []

# Check train_landmarks_npy specifically
landmarks_path = os.path.join(datamount_dir, 'train_landmarks_npy')
if os.path.exists(landmarks_path):
    if os.path.islink(landmarks_path):
        link_target = os.readlink(landmarks_path)
        if '/kaggle/input' in link_target:
            print(f"   🗑️  Removing symlink to input: train_landmarks_npy")
            os.unlink(landmarks_path)
            removed.append('train_landmarks_npy (symlink)')
        else:
            print(f"   ✅ Keeping symlink: train_landmarks_npy -> {link_target}")
            kept.append('train_landmarks_npy (symlink)')
    elif os.path.isdir(landmarks_path):
        items_in_landmarks = os.listdir(landmarks_path)
        if len(items_in_landmarks) == 0:
            print(f"   🗑️  Removing empty directory: train_landmarks_npy")
            os.rmdir(landmarks_path)
            removed.append('train_landmarks_npy (empty)')
        else:
            # Check if it has actual data
            has_npy_files = False
            for item in items_in_landmarks[:10]:  # Check first 10
                item_path = os.path.join(landmarks_path, item)
                if os.path.isdir(item_path):
                    files = os.listdir(item_path)
                    if any(f.endswith('.npy') for f in files):
                        has_npy_files = True
                        break
            
            if has_npy_files:
                print(f"   ✅ Keeping train_landmarks_npy ({len(items_in_landmarks)} directories with data)")
                kept.append('train_landmarks_npy (has data)')
            else:
                print(f"   🗑️  Removing train_landmarks_npy (no .npy files found)")
                shutil.rmtree(landmarks_path)
                removed.append('train_landmarks_npy (no data)')

# Check inference_args.json - keep if it exists
inference_args_path = os.path.join(datamount_dir, 'inference_args.json')
if os.path.exists(inference_args_path):
    # Check if it's in train_landmarks_npy subdirectory
    if os.path.exists(os.path.join(datamount_dir, 'train_landmarks_npy', 'inference_args.json')):
        # Keep the one in subdirectory, can remove root one if duplicate
        pass
    kept.append('inference_args.json')

# Clean up other items
print(f"\n🧹 Cleaning up other items...")
for item in items:
    if item == 'train_landmarks_npy':  # Already handled
        continue
        
    item_path = os.path.join(datamount_dir, item)
    
    # Keep essential files
    if item in essential_files:
        if os.path.exists(item_path):
            kept.append(item)
            continue
    
    # Keep essential directories
    if item in essential_dirs:
        if os.path.exists(item_path):
            kept.append(item)
            continue
    
    # Remove temporary/backup files
    if os.path.isfile(item_path):
        if item.startswith('.') or item.endswith(('.tmp', '.bak', '.old', '~')):
            print(f"   🗑️  Removing temporary file: {item}")
            os.remove(item_path)
            removed.append(item)
        elif item == 'inference_args.json':
            # Keep inference_args.json (already handled above)
            kept.append(item)
        else:
            # Unknown file - ask or keep it
            print(f"   ⚠️  Unknown file (keeping): {item}")
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
                # Keep non-empty directories (might be important)
                print(f"   ✅ Keeping directory: {item} ({len(dir_items)} items)")
                kept.append(item)
        except (PermissionError, OSError) as e:
            print(f"   ⚠️  Cannot access {item}: {e}")
            kept.append(item)

# Summary
print(f"\n" + "="*60)
print("CLEANUP SUMMARY")
print("="*60)
print(f"✅ Kept: {len(kept)} items")
for item in sorted(kept):
    print(f"   - {item}")

if removed:
    print(f"\n🗑️  Removed: {len(removed)} items")
    for item in sorted(removed):
        print(f"   - {item}")
else:
    print(f"\n🗑️  Removed: 0 items")

# Check disk space
try:
    total, used, free = shutil.disk_usage(datamount_dir)
    print(f"\n💾 Free space: {free / (1024**3):.2f} GB")
except:
    pass

print("\n" + "="*60)
print("CLEANUP COMPLETE")
print("="*60)

