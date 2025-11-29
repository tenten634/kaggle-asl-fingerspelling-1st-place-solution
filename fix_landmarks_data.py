"""
Script to find and copy landmark data files to the correct location
"""
import os
import shutil

print("="*60)
print("FIXING LANDMARKS DATA")
print("="*60)

# Target directory
target_dir = 'datamount/train_landmarks_npy'
os.makedirs(target_dir, exist_ok=True)

# Check what's currently in target
print(f"\n📁 Current target directory: {target_dir}")
if os.path.exists(target_dir):
    items = os.listdir(target_dir)
    print(f"   Contains {len(items)} items")
    if len(items) > 0:
        print(f"   Sample items: {items[:5]}")
        # Check if any are directories
        dirs = [item for item in items if os.path.isdir(os.path.join(target_dir, item))]
        if dirs:
            print(f"   Found {len(dirs)} directories")
            sample_dir = os.path.join(target_dir, dirs[0])
            if os.path.exists(sample_dir):
                files = os.listdir(sample_dir)
                print(f"   Sample directory '{dirs[0]}' has {len(files)} files")
else:
    print("   ⚠️  Target directory doesn't exist")

# Search in input datasets
print(f"\n📦 Searching in input datasets...")
input_base = '/kaggle/input'
found_landmarks = []

for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if os.path.isdir(dataset_path):
        landmarks_path = os.path.join(dataset_path, 'train_landmarks_npy')
        if os.path.exists(landmarks_path):
            print(f"\n✅ Found landmarks in: {dataset_dir}")
            print(f"   Path: {landmarks_path}")
            
            # Check what's in it
            items = os.listdir(landmarks_path)
            dirs = [item for item in items if os.path.isdir(os.path.join(landmarks_path, item))]
            print(f"   Contains {len(dirs)} directories")
            
            if len(dirs) > 0:
                sample_dir = os.path.join(landmarks_path, dirs[0])
                if os.path.exists(sample_dir):
                    files = os.listdir(sample_dir)
                    print(f"   Sample directory '{dirs[0]}' has {len(files)} .npy files")
                    found_landmarks.append((dataset_dir, landmarks_path, len(dirs)))

# Copy from the first found dataset
if found_landmarks:
    source_dataset, source_path, num_dirs = found_landmarks[0]
    print(f"\n📋 Copying from: {source_dataset}")
    print(f"   Source: {source_path}")
    print(f"   Target: {target_dir}")
    print(f"   Will copy {num_dirs} directories...")
    print("   (This may take a while)")
    
    # Check disk space first
    import shutil
    total, used, free = shutil.disk_usage(target_dir)
    print(f"\n💾 Disk space:")
    print(f"   Total: {total / (1024**3):.2f} GB")
    print(f"   Used: {used / (1024**3):.2f} GB")
    print(f"   Free: {free / (1024**3):.2f} GB")
    
    if free < 5 * 1024**3:  # Less than 5GB free
        print("   ⚠️  WARNING: Low disk space!")
        print("   You may run out of space during copying")
    
    # Copy directories
    copied = 0
    failed = 0
    
    for item in os.listdir(source_path):
        src = os.path.join(source_path, item)
        dst = os.path.join(target_dir, item)
        
        if os.path.isdir(src):
            try:
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
                    copied += 1
                    if copied % 100 == 0:
                        print(f"   Copied {copied} directories...")
                else:
                    # Already exists, skip
                    pass
            except Exception as e:
                failed += 1
                if failed <= 5:  # Only show first 5 errors
                    print(f"   ⚠️  Failed to copy {item}: {e}")
    
    print(f"\n✅ Copy complete!")
    print(f"   Copied: {copied} directories")
    if failed > 0:
        print(f"   Failed: {failed} directories")
    
    # Verify
    print(f"\n🔍 Verifying...")
    target_items = os.listdir(target_dir)
    target_dirs = [item for item in target_items if os.path.isdir(os.path.join(target_dir, item))]
    print(f"   Target now has {len(target_dirs)} directories")
    
    if len(target_dirs) > 0:
        sample_dir = os.path.join(target_dir, target_dirs[0])
        if os.path.exists(sample_dir):
            files = os.listdir(sample_dir)
            print(f"   Sample directory '{target_dirs[0]}' has {len(files)} files")
            print(f"   ✅ Data appears to be copied correctly!")
else:
    print("\n❌ No landmarks found in input datasets")
    print("   Make sure you've added the datasets in Kaggle's Data sidebar")

print("\n" + "="*60)

