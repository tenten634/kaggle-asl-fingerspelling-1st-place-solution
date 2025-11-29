"""
Check data structure in working directory to identify redundancy
"""
import os
import shutil

print("="*60)
print("INVESTIGATING DATA STRUCTURE")
print("="*60)

# Check datamount directory
datamount_dir = 'datamount'
if os.path.exists(datamount_dir):
    print(f"\n📁 {datamount_dir}/")
    items = os.listdir(datamount_dir)
    for item in sorted(items):
        item_path = os.path.join(datamount_dir, item)
        if os.path.isdir(item_path):
            try:
                sub_items = os.listdir(item_path)
                if item == 'train_landmarks_npy':
                    # Count directories and files
                    dirs = [i for i in sub_items if os.path.isdir(os.path.join(item_path, i))]
                    files = [i for i in sub_items if os.path.isfile(os.path.join(item_path, i))]
                    print(f"   📂 {item}/ ({len(dirs)} dirs, {len(files)} files)")
                    if len(dirs) > 0:
                        # Check a sample directory
                        sample_dir = os.path.join(item_path, dirs[0])
                        if os.path.exists(sample_dir):
                            sample_files = os.listdir(sample_dir)
                            npy_files = [f for f in sample_files if f.endswith('.npy')]
                            print(f"      Sample: {dirs[0]}/ has {len(npy_files)} .npy files")
                elif item == 'weights':
                    print(f"   📂 {item}/ ({len(sub_items)} items)")
                else:
                    print(f"   📂 {item}/ ({len(sub_items)} items)")
            except:
                print(f"   📂 {item}/ (cannot access)")
        else:
            size = os.path.getsize(item_path) / (1024*1024)  # MB
            print(f"   📄 {item} ({size:.2f} MB)")
else:
    print(f"\n❌ {datamount_dir} doesn't exist")

# Check input datasets
print(f"\n📦 Input datasets (/kaggle/input/):")
input_base = '/kaggle/input'
if os.path.exists(input_base):
    for dataset_dir in sorted(os.listdir(input_base)):
        dataset_path = os.path.join(input_base, dataset_dir)
        if os.path.isdir(dataset_path):
            items = os.listdir(dataset_path)
            print(f"\n   📁 {dataset_dir}/")
            for item in sorted(items):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    try:
                        sub_items = os.listdir(item_path)
                        if item == 'train_landmarks_npy' or item == 'supplemental_landmarks':
                            dirs = [i for i in sub_items if os.path.isdir(os.path.join(item_path, i))]
                            print(f"      📂 {item}/ ({len(dirs)} directories)")
                            if len(dirs) > 0:
                                sample_dir = os.path.join(item_path, dirs[0])
                                if os.path.exists(sample_dir):
                                    sample_files = os.listdir(sample_dir)
                                    npy_files = [f for f in sample_files if f.endswith('.npy')]
                                    print(f"         Sample: {dirs[0]}/ has {len(npy_files)} .npy files")
                        else:
                            print(f"      📂 {item}/ ({len(sub_items)} items)")
                    except:
                        print(f"      📂 {item}/ (cannot access)")
                else:
                    size = os.path.getsize(item_path) / (1024*1024)  # MB
                    print(f"      📄 {item} ({size:.2f} MB)")
else:
    print("   ❌ /kaggle/input doesn't exist")

# Check for redundancy
print(f"\n🔍 Checking for redundancy...")
datamount_landmarks = os.path.join(datamount_dir, 'train_landmarks_npy')
if os.path.exists(datamount_landmarks):
    datamount_dirs = set()
    if os.path.isdir(datamount_landmarks):
        try:
            datamount_dirs = set([d for d in os.listdir(datamount_landmarks) 
                                 if os.path.isdir(os.path.join(datamount_landmarks, d))])
            print(f"   datamount/train_landmarks_npy/ has {len(datamount_dirs)} directories")
        except:
            pass
    
    # Check if any are symlinks
    if os.path.islink(datamount_landmarks):
        print(f"   ⚠️  datamount/train_landmarks_npy is a SYMLINK")
        print(f"      Points to: {os.readlink(datamount_landmarks)}")
    
    # Check input datasets
    if os.path.exists(input_base):
        for dataset_dir in os.listdir(input_base):
            dataset_path = os.path.join(input_base, dataset_dir)
            if os.path.isdir(dataset_path):
                train_landmarks = os.path.join(dataset_path, 'train_landmarks_npy')
                supp_landmarks = os.path.join(dataset_path, 'supplemental_landmarks')
                
                if os.path.exists(train_landmarks):
                    try:
                        train_dirs = set([d for d in os.listdir(train_landmarks) 
                                         if os.path.isdir(os.path.join(train_landmarks, d))])
                        overlap = datamount_dirs & train_dirs
                        if overlap:
                            print(f"   ⚠️  Overlap with {dataset_dir}/train_landmarks_npy: {len(overlap)} directories")
                    except:
                        pass
                
                if os.path.exists(supp_landmarks):
                    try:
                        supp_dirs = set([d for d in os.listdir(supp_landmarks) 
                                        if os.path.isdir(os.path.join(supp_landmarks, d))])
                        overlap = datamount_dirs & supp_dirs
                        if overlap:
                            print(f"   ⚠️  Overlap with {dataset_dir}/supplemental_landmarks: {len(overlap)} directories")
                        else:
                            print(f"   ✅ {dataset_dir}/supplemental_landmarks has {len(supp_dirs)} directories (not yet merged)")
                    except:
                        pass

# Check disk space
print(f"\n💾 Disk space:")
try:
    total, used, free = shutil.disk_usage('.')
    print(f"   Total: {total / (1024**3):.2f} GB")
    print(f"   Used: {used / (1024**3):.2f} GB")
    print(f"   Free: {free / (1024**3):.2f} GB")
except:
    pass

print("\n" + "="*60)

