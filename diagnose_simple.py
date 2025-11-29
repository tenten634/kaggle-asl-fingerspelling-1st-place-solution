"""
Simple diagnosis without loading config
"""
import os

print("="*60)
print("SIMPLE DIAGNOSIS (No Config Loading)")
print("="*60)

# Check input datasets
print(f"\n📦 Input Datasets:")
input_base = '/kaggle/input'
if os.path.exists(input_base):
    for dataset_name in sorted(os.listdir(input_base)):
        dataset_path = os.path.join(input_base, dataset_name)
        if os.path.isdir(dataset_path):
            print(f"\n   📁 {dataset_name}/")
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
                                abs_path = os.path.realpath(os.path.join(item_path, 'inference_args.json'))
                                print(f"         Path: {abs_path}")
                    except:
                        print(f"      📂 {item}/ (cannot access)")

# Check what would be detected
print(f"\n🔍 What Would Be Detected:")
if os.path.exists(input_base):
    for dataset_name in os.listdir(input_base):
        dataset_path = os.path.join(input_base, dataset_name)
        if os.path.isdir(dataset_path):
            train_landmarks_path = os.path.join(dataset_path, 'train_landmarks_npy')
            if os.path.exists(train_landmarks_path):
                try:
                    items = os.listdir(train_landmarks_path)
                    dirs = [i for i in items if os.path.isdir(os.path.join(train_landmarks_path, i))]
                    if len(dirs) > 10:
                        abs_path = os.path.realpath(train_landmarks_path) + '/'
                        print(f"   ✅ {dataset_name}/train_landmarks_npy")
                        print(f"      Would set: cfg.data_folder = '{abs_path}'")
                        print(f"      Has {len(dirs)} directories")
                except:
                    pass

print("\n" + "="*60)

