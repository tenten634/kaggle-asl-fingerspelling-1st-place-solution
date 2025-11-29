# Google Colab Setup Guide

This guide will help you set up and run the ASL Fingerspelling Recognition training in Google Colab.

## Quick Start

### Option 1: Using the Interactive Script (Recommended)

1. **Upload the repository to Colab**
   ```python
   # In a Colab cell, run:
   !git clone https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution
   %cd kaggle-asl-fingerspelling-1st-place-solution
   ```

2. **Check environment**
   ```python
   !python check_colab_environment.py
   ```

3. **Setup dependencies**
   ```python
   !python setup_colab.py
   ```

4. **Run the interactive pipeline**
   ```python
   !python run_colab.py
   ```

### Option 2: Using Individual Scripts

1. **Check environment**
   ```python
   !python check_colab_environment.py
   ```

2. **Install dependencies**
   ```python
   !python setup_colab.py
   # Or manually:
   !pip install -r requirements_colab.txt
   ```

3. **Download data (if needed)**
   ```python
   # Set up Kaggle API
   from google.colab import files
   files.upload()  # Upload your kaggle.json
   
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   
   # Download data
   %cd datamount
   !kaggle datasets download -d darraghdog/asl-fingerspelling-preprocessing-train-dataset
   !unzip -n asl-fingerspelling-preprocessing-train-dataset.zip
   !rm asl-fingerspelling-preprocessing-train-dataset.zip
   !kaggle datasets download -d darraghdog/asl-fingerspelling-preprocessed-supp-dataset
   !unzip -n asl-fingerspelling-preprocessed-supp-dataset.zip
   !mv supplemental_landmarks/* train_landmarks_npy/
   !rm asl-fingerspelling-preprocessed-supp-dataset.zip
   !rm -rf supplemental_landmarks/
   %cd ..
   ```

4. **Run training**
   ```python
   # Round 1 - Train 4 folds
   !python train_colab.py -C cfg_1 --fold 0
   !python train_colab.py -C cfg_1 --fold 1
   !python train_colab.py -C cfg_1 --fold 2
   !python train_colab.py -C cfg_1 --fold 3
   
   # Generate OOF predictions
   !python scripts/get_train_folded_oof_supp.py
   
   # Round 2 - Train fullfit
   !python train_colab.py -C cfg_2 --fold -1
   !python train_colab.py -C cfg_2 --fold -1  # Second seed
   
   # Convert to TF-Lite
   !python scripts/convert_cfg_2_to_tf_lite.py
   ```

## GPU/TPU Setup

### Enable GPU
1. Go to **Runtime → Change runtime type**
2. Select **GPU** (T4, V100, or A100)
3. Click **Save**

### Enable TPU (Optional)
1. Go to **Runtime → Change runtime type**
2. Select **TPU**
3. Click **Save**
4. Install TPU libraries:
   ```python
   !pip install cloud-tpu-client==0.10 torch-xla
   ```
5. Use TPU flag when training:
   ```python
   !python train_colab.py -C cfg_1 --fold 0 --use_tpu
   ```

## Configuration

### Adjusting for Colab

The `train_colab.py` script automatically adjusts:
- `num_workers`: Reduced to 2 (Colab doesn't handle high num_workers well)
- `pin_memory`: Set to False (can cause issues in Colab)
- Neptune logging: Made optional (can be disabled with `--disable_neptune`)

### Memory Management

If you run into OOM (Out of Memory) errors:
- Reduce `batch_size` in config files
- Reduce `num_workers` (already done automatically)
- Use gradient accumulation (already configured)
- Enable mixed precision (set `mixed_precision = True` in config)

## Troubleshooting

### Issue: "CUDA not available"
**Solution**: Make sure GPU is enabled in Runtime settings

### Issue: "Data files not found"
**Solution**: Download data using Kaggle API (see step 3 above)

### Issue: "Neptune connection error"
**Solution**: Use `--disable_neptune` flag or set up Neptune API token:
```python
import os
os.environ['NEPTUNE_API_TOKEN'] = 'your_token_here'
```

### Issue: "Out of Memory"
**Solution**: 
- Reduce batch size in config
- Use smaller model (cfg_1 instead of cfg_2)
- Enable mixed precision

### Issue: "Module not found"
**Solution**: Run `setup_colab.py` or manually install:
```python
!pip install -r requirements_colab.txt
```

## File Structure

```
kaggle-asl-fingerspelling-1st-place-solution/
├── check_colab_environment.py  # Environment check
├── setup_colab.py              # Dependency installation
├── train_colab.py              # Colab-compatible training script
├── run_colab.py                # Interactive pipeline runner
├── requirements_colab.txt      # Colab-specific requirements
├── COLAB_SETUP.md             # This file
├── configs/                    # Configuration files
├── data/                       # Dataset classes
├── models/                     # Model architectures
├── datamount/                  # Data directory
└── ...
```

## Differences from Original

The Colab version includes these modifications:
1. **Automatic environment detection** - Detects Colab and adjusts settings
2. **Optional Neptune logging** - Can be disabled if not needed
3. **Reduced num_workers** - Colab-friendly defaults
4. **TPU support** - Optional TPU training support
5. **Better error handling** - More informative error messages
6. **Path handling** - Works with Colab's file system

## Notes

- Colab sessions have time limits (12 hours for free, longer for Pro)
- Save checkpoints regularly (already configured)
- Download results before session ends
- Consider using Google Drive to persist data between sessions

## Saving Results

To save results to Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy results
!cp -r datamount/weights /content/drive/MyDrive/asl_fingerspelling_weights
```

## Support

For issues or questions:
1. Check the original README.md
2. Review error messages in the output
3. Check Colab-specific issues in this guide

